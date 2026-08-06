#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
CONTROLLER="${REPO_ROOT}/tuning_pipeline/workflow/continuous/continuous_tuning.py"
CONFIG="${SCRIPT_DIR}/config.yaml"
RUNTIME_ROOT="${SCRIPT_DIR}/runtime"
PROCESS_ROOT="${RUNTIME_ROOT}/process"
PYTHON_BIN="${VLLMTKB_PYTHON:-python3}"

mkdir -p "${PROCESS_ROOT}"

controller() {
  "${PYTHON_BIN}" "${CONTROLLER}" \
    --config "${CONFIG}" \
    --runtime-root "${RUNTIME_ROOT}" \
    "$@"
}
