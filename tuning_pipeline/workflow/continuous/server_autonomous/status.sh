#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

if [[ -f "${PROCESS_ROOT}/launcher.pid" ]]; then
  PID=$(tr -d '[:space:]' < "${PROCESS_ROOT}/launcher.pid")
  if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
    echo "Controller: running (PID=${PID})"
  else
    echo "Controller: not running (recorded PID=${PID:-invalid})"
  fi
else
  echo "Controller: not started"
fi

controller --status
echo
ktp-lab status --lease vllmtkb-server-auto-418bd627-32c8cf190-glm52-a3-32npu 2>&1 || true
