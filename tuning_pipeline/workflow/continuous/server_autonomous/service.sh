#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"
SYSTEMD_UNIT="${SERVICE_ROOT}/vllmtkb-server-autonomous.service"
SUPERVISOR_CONFIG="${SERVICE_ROOT}/supervisord.conf"
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
  supervisor-start)
    command -v supervisord >/dev/null || { echo "Install Supervisor first: python3 -m pip install supervisor" >&2; exit 2; }
    render
    if supervisorctl -c "${SUPERVISOR_CONFIG}" status >/dev/null 2>&1; then
      supervisorctl -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}"
    else
      supervisord -c "${SUPERVISOR_CONFIG}"
    fi
    ;;
  supervisor-stop|supervisor-status)
    ACTION=${1#supervisor-}
    supervisorctl -c "${SUPERVISOR_CONFIG}" "${ACTION}" "${SERVICE_NAME}"
    ;;
  supervisor-restart)
    supervisorctl -c "${SUPERVISOR_CONFIG}" stop "${SERVICE_NAME}"
    archive_stop_marker
    supervisorctl -c "${SUPERVISOR_CONFIG}" start "${SERVICE_NAME}"
    ;;
  supervisor-shutdown)
    supervisorctl -c "${SUPERVISOR_CONFIG}" shutdown
    ;;
  *)
    echo "usage: $0 {prepare-env|render|authorize-resume|systemd-install|systemd-start|systemd-stop|systemd-restart|systemd-status|systemd-logs [N]|supervisor-start|supervisor-stop|supervisor-restart|supervisor-status|supervisor-shutdown}" >&2
    exit 2
    ;;
esac
