#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"
SYSTEMD_UNIT="${SERVICE_ROOT}/vllmtkb-server-autonomous.service"
SUPERVISOR_CONFIG="${SERVICE_ROOT}/supervisord.conf"
SUPERVISOR_VENV="${SERVICE_ROOT}/supervisor-venv"
SERVICE_NAME="vllmtkb-server-autonomous"

render() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/service_runtime.py" render \
    --repo-root "${REPO_ROOT}" --env-file "${SERVICE_ENV_FILE}" \
    --output-root "${SERVICE_ROOT}"
}

archive_stop_marker() {
  local marker="${RUNTIME_ROOT}/STOP_REQUESTED"
  if [[ -f "${marker}" ]]; then
    local archive="${RUNTIME_ROOT}/STOP_REQUESTED.authorized-$(date +%Y%m%d_%H%M%S)-$$"
    mv "${marker}" "${archive}"
    echo "Archived the stop marker as: ${archive}"
  fi
}

archive_terminal_state() {
  local state="${RUNTIME_ROOT}/state.json"
  [[ -f "${state}" ]] || { echo "No state.json exists; a new Session is already authorized."; return; }
  local summary
  summary=$("${PYTHON_BIN}" - "${state}" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
status = str(state.get("status", ""))
terminal = {"dry_run_complete", "completed_by_agent", "tuning_complete"}
has_task = bool(state.get("active_task_id"))
has_run = bool(state.get("active_run_id"))
operator_stopped = (
    status == "stopped_after_failed_round"
    and state.get("last_failure_classification") == "operator_stop_before_metrics"
    and not has_task
)
# Older dry-run states retained the simulated run id even though no task was
# submitted. Accept that one legacy shape, while keeping real terminal states
# and every dry-run task identity fail-closed. An explicitly operator-stopped
# round is also terminal once the Controller has cleared its active task.
legacy_dry_run = status == "dry_run_complete" and not has_task
archivable = status in terminal or operator_stopped
retained_audit_run = legacy_dry_run or operator_stopped
if not archivable or has_task or (has_run and not retained_audit_run):
    raise SystemExit(
        f"Refusing to archive non-terminal state: status={status!r}, "
        f"active_task={has_task}, active_run={has_run}"
    )
print(status)
PY
  )
  local archive="${RUNTIME_ROOT}/state.${summary}.archived-$(date +%Y%m%d_%H%M%S)-$$.json"
  mv "${state}" "${archive}"
  archive_stop_marker
  echo "Archived terminal state as: ${archive}"
}

supervisor_bin() {
  if [[ -x "${SUPERVISOR_VENV}/bin/$1" ]]; then
    printf '%s\n' "${SUPERVISOR_VENV}/bin/$1"
  else
    command -v "$1"
  fi
}

case "${1:-}" in
  prepare-env)
    mkdir -p "$(dirname -- "${SERVICE_ENV_FILE}")"
    if [[ -e "${SERVICE_ENV_FILE}" ]]; then
      echo "Environment file already exists; it was not overwritten: ${SERVICE_ENV_FILE}"
      exit 0
    fi
    install -m 600 "${SCRIPT_DIR}/controller.env.example" "${SERVICE_ENV_FILE}"
    echo "Edit the placeholder API key in: ${SERVICE_ENV_FILE}"
    ;;
  render)
    render
    ;;
  authorize-resume)
    archive_stop_marker
    ;;
  replay-unmeasured)
    [[ "${2:-}" =~ ^round_[0-9]+_a[0-9]+([a-z][0-9]+)?$ ]] || {
      echo "usage: $0 replay-unmeasured round_NNN_aN[rN]" >&2
      exit 2
    }
    request="${RUNTIME_ROOT}/REPLAY_UNMEASURED_REQUEST"
    [[ ! -e "${request}" ]] || {
      echo "A replay request already exists: ${request}" >&2
      exit 2
    }
    temporary="${request}.tmp-$$"
    printf '%s\n' "$2" > "${temporary}"
    mv "${temporary}" "${request}"
    echo "Authorized audited replay of: $2"
    ;;
  authorize-new-session)
    archive_terminal_state
    ;;
  systemd-install)
    command -v systemctl >/dev/null || { echo "systemctl is unavailable" >&2; exit 2; }
    render
    systemctl --user link "${SYSTEMD_UNIT}"
    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}.service"
    echo "Installed but not started. Run: $0 systemd-start"
    ;;
  systemd-start|systemd-stop|systemd-status)
    ACTION=${1#systemd-}
    systemctl --user "${ACTION}" "${SERVICE_NAME}.service"
    ;;
  systemd-restart)
    systemctl --user stop "${SERVICE_NAME}.service"
    archive_stop_marker
    systemctl --user start "${SERVICE_NAME}.service"
    ;;
  systemd-logs)
    journalctl --user -u "${SERVICE_NAME}.service" -n "${2:-100}" --no-pager
    ;;
  supervisor-install)
    "${PYTHON_BIN}" -m venv "${SUPERVISOR_VENV}"
    "${SUPERVISOR_VENV}/bin/python" -m pip install --no-cache-dir "supervisor==4.3.0"
    render
    echo "Supervisor 4.3.0 installed inside: ${SUPERVISOR_VENV}"
    ;;
  supervisor-start)
    SUPERVISORD=$(supervisor_bin supervisord) || { echo "Run first: $0 supervisor-install" >&2; exit 2; }
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Run first: $0 supervisor-install" >&2; exit 2; }
    render
    # `status` returns 3 when the managed program is EXITED even though the
    # supervisord daemon is healthy. Probe the daemon itself before deciding
    # whether to start a new daemon or restart the existing program.
    if (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" pid >/dev/null 2>&1); then
      (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}")
    else
      (cd "${SERVICE_ROOT}" && "${SUPERVISORD}" -c "${SUPERVISOR_CONFIG}")
    fi
    ;;
  supervisor-stop|supervisor-status)
    ACTION=${1#supervisor-}
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" "${ACTION}" "${SERVICE_NAME}")
    ;;
  supervisor-restart)
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" stop "${SERVICE_NAME}")
    archive_stop_marker
    (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}")
    ;;
  supervisor-shutdown)
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    (cd "${SERVICE_ROOT}" && "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" shutdown)
    ;;
  *)
    echo "usage: $0 {prepare-env|render|authorize-resume|replay-unmeasured ROUND|authorize-new-session|systemd-install|systemd-start|systemd-stop|systemd-restart|systemd-status|systemd-logs [N]|supervisor-install|supervisor-start|supervisor-stop|supervisor-restart|supervisor-status|supervisor-shutdown}" >&2
    exit 2
    ;;
esac
