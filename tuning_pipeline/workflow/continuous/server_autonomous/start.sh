#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

MODE="${1:-auto}"
case "${MODE}" in
  auto)
    if [[ -f "${RUNTIME_ROOT}/state.json" ]]; then
      ACTION=--resume
    else
      ACTION=--start
    fi
    ;;
  new) ACTION=--start ;;
  resume) ACTION=--resume ;;
  *) echo "usage: $0 [auto|new|resume]" >&2; exit 2 ;;
esac

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
  echo "DEEPSEEK_API_KEY is not set." >&2
  exit 2
}
if [[ -f "${RUNTIME_ROOT}/controller.lock" ]]; then
  echo "A controller lock already exists; run status.sh before starting." >&2
  exit 2
fi

if "${PYTHON_BIN}" "${TOPOLOGY_CAMPAIGN}" \
  --config "${CONFIG}" --campaign-root "${CAMPAIGN_ROOT}" --is-enabled; then
  nohup "${PYTHON_BIN}" "${TOPOLOGY_CAMPAIGN}" \
    --config "${CONFIG}" --campaign-root "${CAMPAIGN_ROOT}" \
    --run --mode "${MODE}" \
    >"${PROCESS_ROOT}/stdout.log" \
    2>"${PROCESS_ROOT}/stderr.log" &
  PID=$!
  printf '%s\n' "${PID}" > "${PROCESS_ROOT}/launcher.pid"
  echo "Topology Campaign started with PID=${PID}, mode=${MODE}."
  exit 0
fi

nohup "${PYTHON_BIN}" "${CONTROLLER}" \
  --config "${CONFIG}" \
  --runtime-root "${RUNTIME_ROOT}" \
  "${ACTION}" \
  >"${PROCESS_ROOT}/stdout.log" \
  2>"${PROCESS_ROOT}/stderr.log" &
PID=$!
printf '%s\n' "${PID}" > "${PROCESS_ROOT}/launcher.pid"
echo "Server-autonomous controller started with PID=${PID}, action=${ACTION}."
