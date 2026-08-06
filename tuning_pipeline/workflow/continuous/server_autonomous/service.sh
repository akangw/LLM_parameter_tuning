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
if state.get("active_task_id") or state.get("active_run_id") or status not in terminal:
    raise SystemExit(
        f"Refusing to archive non-terminal state: status={status!r}, "
        f"active_task={bool(state.get('active_task_id'))}, "
        f"active_run={bool(state.get('active_run_id'))}"
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
    if "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" status >/dev/null 2>&1; then
      "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}"
    else
      "${SUPERVISORD}" -c "${SUPERVISOR_CONFIG}"
    fi
    ;;
  supervisor-stop|supervisor-status)
    ACTION=${1#supervisor-}
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" "${ACTION}" "${SERVICE_NAME}"
    ;;
  supervisor-restart)
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" stop "${SERVICE_NAME}"
    archive_stop_marker
    "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}"
    ;;
  supervisor-shutdown)
    SUPERVISORCTL=$(supervisor_bin supervisorctl) || { echo "Supervisor is not installed" >&2; exit 2; }
    "${SUPERVISORCTL}" -c "${SUPERVISOR_CONFIG}" shutdown
    ;;
  *)
    echo "usage: $0 {prepare-env|render|authorize-resume|authorize-new-session|systemd-install|systemd-start|systemd-stop|systemd-restart|systemd-status|systemd-logs [N]|supervisor-install|supervisor-start|supervisor-stop|supervisor-restart|supervisor-status|supervisor-shutdown}" >&2
    exit 2
    ;;
esac
