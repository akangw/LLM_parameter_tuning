#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

RUNNING=false
for PID_FILE in controller.pid launcher.pid; do
  if [[ -f "${PROCESS_ROOT}/${PID_FILE}" ]]; then
    PID=$(tr -d '[:space:]' < "${PROCESS_ROOT}/${PID_FILE}")
    if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
      echo "Controller: running (PID=${PID}, source=${PID_FILE})"
      RUNNING=true
      break
    fi
  fi
done
if [[ "${RUNNING}" == false ]]; then
  echo "Controller: not running"
fi

if command -v systemctl >/dev/null 2>&1; then
  SYSTEMD_STATE=$(systemctl --user is-active vllmtkb-server-autonomous.service 2>/dev/null || true)
  echo "systemd user service: ${SYSTEMD_STATE:-unavailable}"
fi
if [[ -f "${SERVICE_ROOT}/supervisord.conf" ]] && command -v supervisorctl >/dev/null 2>&1; then
  SUPERVISOR_STATE=$(supervisorctl -c "${SERVICE_ROOT}/supervisord.conf" status \
    vllmtkb-server-autonomous 2>/dev/null || true)
  echo "Supervisor service: ${SUPERVISOR_STATE:-not running}"
fi

controller --status
echo
ktp-lab status --lease vllmtkb-server-auto-418bd627-32c8cf190-glm52-a3-32npu 2>&1 || true
