#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_runtime_loop.sh"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1

echo "EXPERIMENT_RUN_ID=${EXPERIMENT_RUN_ID}"
echo "ROLE=worker NODE_IP=${NODE_IP} MASTER_IP=${MASTER_IP} NIC_NAME=${NIC_NAME}"
vllm serve "${MODEL_PATH}" --headless --data-parallel-start-rank 1 "${VLLM_COMMON_ARGS[@]}" &
VLLM_PID=$!

stop_worker() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap stop_worker EXIT TERM INT

while kill -0 "${VLLM_PID}" 2>/dev/null; do
  if [[ -f "${RUN_DIR}/MASTER_DONE" ]]; then
    echo "Master finished; stopping worker."
    exit 0
  fi
  sleep 10
done
wait "${VLLM_PID}"
