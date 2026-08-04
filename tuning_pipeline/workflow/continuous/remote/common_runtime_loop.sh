#!/bin/bash
set -eo pipefail

AUTO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${AUTO_DIR}/../.." && pwd)
MODEL_PATH=/models/share/GLM-5.2-w8a8
if [[ -z "${EXPERIMENT_RUN_ID:-}" ]]; then
  LAB_ACTIVE_RUN="${AUTO_DIR}/lab_active_run.env"
  [[ -f "${LAB_ACTIVE_RUN}" ]] || {
    echo "Missing lab run pointer: ${LAB_ACTIVE_RUN}" >&2
    exit 2
  }
  source "${LAB_ACTIVE_RUN}"
fi
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:?EXPERIMENT_RUN_ID is required}"
RUN_DIR="${AUTO_DIR}/runs/${EXPERIMENT_RUN_ID}"
PARAM_FILE="${RUN_DIR}/candidate.env"
mkdir -p "${RUN_DIR}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Missing candidate parameter file: ${PARAM_FILE}" >&2
  exit 2
fi

# candidate.env is generated locally from a strict whitelist and copied into
# this unique run directory before task submission.
source "${PARAM_FILE}"
BENCHMARK_MODE="${BENCHMARK_MODE:-legacy_random_32k1k}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-192}"
ENABLE_EPLB="${ENABLE_EPLB:-false}"
EPLB_NUM_REDUNDANT_EXPERTS="${EPLB_NUM_REDUNDANT_EXPERTS:-0}"
COMPILATION_ENABLE_SP="${COMPILATION_ENABLE_SP:-false}"
CUDAGRAPH_CAPTURE_SIZES_JSON="${CUDAGRAPH_CAPTURE_SIZES_JSON:-null}"
DECODE_CONTEXT_PARALLEL_SIZE="${DECODE_CONTEXT_PARALLEL_SIZE:-1}"
ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1="${ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1:-false}"
ADDITIONAL_CONFIG_ENABLE_MLAPO="${ADDITIONAL_CONFIG_ENABLE_MLAPO:-true}"
ADDITIONAL_CONFIG_ENABLE_FUSED_MC2="${ADDITIONAL_CONFIG_ENABLE_FUSED_MC2:-0}"
source /models/share/init_env.sh
source /usr/local/Ascend/cann/set_env.sh

unset LOCAL_RANK
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# This customized glm5.2-a3 image carries source builds whose package version
# is 0.1.dev*. vLLM Ascend uses VLLM_VERSION to select its compatible upstream
# API branch; the deployment owner's known-good environment pins it to 0.21.0.
export VLLM_VERSION=0.21.0
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export NIC_NAME=bond4.3000
export HCCL_IF_IP="${NODE_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset VLLM_ASCEND_ENABLE_MLAPO
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_ENABLE_FUSED_MC2
export ASCEND_LAUNCH_BLOCKING=0

bool_flag() {
  [[ "${1}" == "true" ]]
}

VLLM_COMMON_ARGS=(
  --data-parallel-size 2
  --data-parallel-size-local 1
  --data-parallel-address "${MASTER_IP}"
  --data-parallel-rpc-port 12980
  --tensor-parallel-size 16
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}"
  --seed 1024
  --served-model-name glm-5
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --quantization ascend
)

bool_flag "${ENABLE_EXPERT_PARALLEL}" && VLLM_COMMON_ARGS+=(--enable-expert-parallel)
bool_flag "${ENABLE_PREFIX_CACHING}" && VLLM_COMMON_ARGS+=(--enable-prefix-caching)
bool_flag "${ASYNC_SCHEDULING}" && VLLM_COMMON_ARGS+=(--async-scheduling)
if bool_flag "${ENABLE_CHUNKED_PREFILL}"; then
  VLLM_COMMON_ARGS+=(--enable-chunked-prefill)
else
  VLLM_COMMON_ARGS+=(--no-enable-chunked-prefill)
fi
COMPILATION_CONFIG_JSON=$(
  python - "${COMPILATION_MODE}" "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
    "${COMPILATION_ENABLE_SP}" "${CUDAGRAPH_CAPTURE_SIZES_JSON}" <<'PY'
import json
import sys

mode, maximum, enable_sp, sizes_json = sys.argv[1:]
config = {
    "cudagraph_mode": mode,
    "max_cudagraph_capture_size": int(maximum),
    "pass_config": {"enable_sp": enable_sp == "true"},
}
sizes = json.loads(sizes_json)
if sizes is not None:
    config["cudagraph_capture_sizes"] = sizes
print(json.dumps(config, separators=(",", ":")))
PY
)
VLLM_COMMON_ARGS+=(--compilation-config "${COMPILATION_CONFIG_JSON}")
ADDITIONAL_CONFIG_JSON=$(
  python - "${ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1}" \
    "${ADDITIONAL_CONFIG_ENABLE_MLAPO}" \
    "${ADDITIONAL_CONFIG_ENABLE_FUSED_MC2}" <<'PY'
import json
import sys

flashcomm1, mlapo, fused_mc2 = sys.argv[1:]
config = {
    "enable_npugraph_ex": True,
    "fuse_muls_add": True,
    "multistream_overlap_shared_expert": True,
    "enable_flashcomm1": flashcomm1 == "true",
    "enable_mlapo": mlapo == "true",
    "enable_fused_mc2": int(fused_mc2),
}
print(json.dumps(config, separators=(",", ":")))
PY
)
VLLM_COMMON_ARGS+=(--additional-config "${ADDITIONAL_CONFIG_JSON}")
if bool_flag "${ENABLE_EPLB}"; then
  VLLM_COMMON_ARGS+=(--enable-eplb)
  VLLM_COMMON_ARGS+=(--eplb-config "{\"num_redundant_experts\":${EPLB_NUM_REDUNDANT_EXPERTS}}")
elif (( EPLB_NUM_REDUNDANT_EXPERTS > 0 )); then
  echo "EPLB_NUM_REDUNDANT_EXPERTS requires ENABLE_EPLB=true" >&2
  exit 2
fi

if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  MTP_DRAFT_MODEL_PATH="${MTP_DRAFT_MODEL_PATH:?MTP_DRAFT_MODEL_PATH is required when speculative decoding is enabled}"
  VLLM_COMMON_ARGS+=(--speculative-config "{\"model\":\"${MTP_DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"method\":\"mtp\"}")
fi
if (( LONG_PREFILL_TOKEN_THRESHOLD > 0 )); then
  VLLM_COMMON_ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
fi

{
  printf 'vllm serve %q ' "${MODEL_PATH}"
  printf '%q ' "${VLLM_COMMON_ARGS[@]}"
  printf '\n'
} > "${RUN_DIR}/vllm_common_command.txt"
