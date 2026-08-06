#!/bin/bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

MODE="${1:-auto}"
case "${MODE}" in auto|new|resume) ;; *) echo "usage: $0 [auto|new|resume]" >&2; exit 2 ;; esac

if [[ -f "${SERVICE_ENV_FILE}" ]]; then
  ENV_MODE=$(stat -c '%a' "${SERVICE_ENV_FILE}")
  ENV_OWNER=$(stat -c '%u' "${SERVICE_ENV_FILE}")
  if [[ "${ENV_OWNER}" != "$(id -u)" ]] || (( (8#${ENV_MODE} & 8#077) != 0 )); then
    echo "Refusing insecure service env file; it must be owned by this user and mode 600: ${SERVICE_ENV_FILE}" >&2
    exit 78
  fi
  set -a
  # This is an operator-owned, mode-600 shell environment file.
  source "${SERVICE_ENV_FILE}"
  set +a
fi
PYTHON_BIN="${VLLMTKB_PYTHON:-${PYTHON_BIN}}"

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
  echo "DEEPSEEK_API_KEY is not set; configure ${SERVICE_ENV_FILE}." >&2
  exit 78
}

set +e
DECISION_OUTPUT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/service_runtime.py" decide \
  --runtime-root "${RUNTIME_ROOT}" --mode "${MODE}")
DECISION_CODE=$?
set -e
if [[ ${DECISION_CODE} -ne 0 ]]; then
  exit "${DECISION_CODE}"
fi
ACTION=$(printf '%s\n' "${DECISION_OUTPUT}" | head -n 1)
if [[ "${ACTION}" == "complete" ]]; then
  echo "Server-autonomous Session is already complete; service will remain stopped."
  exit 0
fi

CHILD_PID=""
request_graceful_stop() {
  touch "${RUNTIME_ROOT}/STOP_REQUESTED"
  echo "Service stop requested; waiting for the active round to archive cleanly."
}
trap request_graceful_stop TERM INT

"${PYTHON_BIN}" "${CONTROLLER}" \
  --config "${CONFIG}" \
  --runtime-root "${RUNTIME_ROOT}" \
  "${ACTION}" &
CHILD_PID=$!
printf '%s\n' "$$" > "${PROCESS_ROOT}/service-wrapper.pid"
printf '%s\n' "${CHILD_PID}" > "${PROCESS_ROOT}/controller.pid"

set +e
while kill -0 "${CHILD_PID}" 2>/dev/null; do
  wait "${CHILD_PID}"
  CHILD_CODE=$?
done
wait "${CHILD_PID}" 2>/dev/null
FINAL_CODE=$?
if [[ ${FINAL_CODE} -eq 127 ]]; then
  FINAL_CODE=${CHILD_CODE:-0}
fi
exit "${FINAL_CODE}"
