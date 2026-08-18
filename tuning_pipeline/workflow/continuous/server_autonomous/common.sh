#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
CONTROLLER="${REPO_ROOT}/tuning_pipeline/workflow/continuous/continuous_tuning.py"
TOPOLOGY_CAMPAIGN="${REPO_ROOT}/tuning_pipeline/workflow/continuous/topology_campaign.py"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"
LOCAL_CONFIG="${SCRIPT_DIR}/config.local.yaml"
if [[ -n "${VLLMTKB_CONFIG:-}" ]]; then
  CONFIG="${VLLMTKB_CONFIG}"
elif [[ -f "${LOCAL_CONFIG}" ]]; then
  CONFIG="${LOCAL_CONFIG}"
else
  CONFIG="${DEFAULT_CONFIG}"
fi
[[ -f "${CONFIG}" ]] || {
  echo "Server-autonomous config does not exist: ${CONFIG}" >&2
  exit 2
}
# The generic service/status entrypoints follow the checked-in Search-V4
# default. Historical dispatchers always set their own isolated runtime root,
# and an operator may still override this path for a controlled resume.
RUNTIME_ROOT="${VLLMTKB_RUNTIME_ROOT:-${SCRIPT_DIR}/runtime_fixed_dp4_tp8_search_v4_live}"
PROCESS_ROOT="${RUNTIME_ROOT}/process"
SERVICE_ROOT="${RUNTIME_ROOT}/service"
SERVICE_ENV_FILE="${VLLMTKB_ENV_FILE:-${SCRIPT_DIR}/.secrets/controller.env}"
CAMPAIGN_ROOT="${RUNTIME_ROOT}/topology_campaign"
PYTHON_BIN="${VLLMTKB_PYTHON:-python3}"

mkdir -p "${PROCESS_ROOT}" "${SERVICE_ROOT}"

controller() {
  "${PYTHON_BIN}" "${CONTROLLER}" \
    --config "${CONFIG}" \
    --runtime-root "${RUNTIME_ROOT}" \
    "$@"
}
