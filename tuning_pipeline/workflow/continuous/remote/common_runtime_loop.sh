#!/bin/bash
set -eo pipefail

AUTO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${AUTO_DIR}/../.." && pwd)
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
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${SERVICE_PORT:?SERVICE_PORT is required}"
: "${MODEL_QUANTIZATION:?MODEL_QUANTIZATION is required}"
: "${NIC_NAME:?NIC_NAME is required}"
: "${VLLM_COMPAT_VERSION:?VLLM_COMPAT_VERSION is required}"
: "${INIT_ENV_SCRIPT:?INIT_ENV_SCRIPT is required}"
: "${CANN_ENV_SCRIPT:?CANN_ENV_SCRIPT is required}"
: "${LAB_OUTPUT_ROOT:?LAB_OUTPUT_ROOT is required}"
: "${EXECUTOR_REMOTE_CONTRACT:?EXECUTOR_REMOTE_CONTRACT is required}"
[[ -f "${INIT_ENV_SCRIPT}" ]] || { echo "Missing ${INIT_ENV_SCRIPT}" >&2; exit 2; }
[[ -f "${CANN_ENV_SCRIPT}" ]] || { echo "Missing ${CANN_ENV_SCRIPT}" >&2; exit 2; }
LAUNCH_PROFILE="${LAUNCH_PROFILE:-explicit_candidate}"
case "${LAUNCH_PROFILE}" in
  official_source_defaults_deployable|explicit_candidate) ;;
  *)
    echo "Unsupported LAUNCH_PROFILE=${LAUNCH_PROFILE}" >&2
    exit 2
    ;;
esac
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
ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING="${ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING:-false}"
ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE="${ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE:-false}"
SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON="${SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON:-null}"
RUNTIME_INJECTION_MODE="${RUNTIME_INJECTION_MODE:-native_v1}"
RUNTIME_INJECTION_PAYLOAD_B64="${RUNTIME_INJECTION_PAYLOAD_B64:-}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-}"
FIXED_CLI_ARGS_JSON="${FIXED_CLI_ARGS_JSON:-[]}"
FIXED_ADDITIONAL_CONFIG_JSON="${FIXED_ADDITIONAL_CONFIG_JSON:-}"
[[ -n "${FIXED_ADDITIONAL_CONFIG_JSON}" ]] || FIXED_ADDITIONAL_CONFIG_JSON='{}'
FIXED_ENVIRONMENT_JSON="${FIXED_ENVIRONMENT_JSON:-}"
[[ -n "${FIXED_ENVIRONMENT_JSON}" ]] || FIXED_ENVIRONMENT_JSON='{}'
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
SAFETENSORS_PREFETCH_NUM_THREADS="${SAFETENSORS_PREFETCH_NUM_THREADS:-8}"
SAFETENSORS_PREFETCH_BLOCK_SIZE="${SAFETENSORS_PREFETCH_BLOCK_SIZE:-16777216}"
source "${INIT_ENV_SCRIPT}"
source "${CANN_ENV_SCRIPT}"

if [[ -n "${RUNTIME_CACHE_ROOT}" ]]; then
  export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
  export HF_HOME="${RUNTIME_CACHE_ROOT}/huggingface"
  export TORCH_HOME="${RUNTIME_CACHE_ROOT}/torch"
  export TRITON_CACHE_DIR="${RUNTIME_CACHE_ROOT}/triton"
  export TORCH_EXTENSIONS_DIR="${RUNTIME_CACHE_ROOT}/torch-extensions"
  mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
fi

unset LOCAL_RANK
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800
# This customized glm5.2-a3 image carries source builds whose package version
# is 0.1.dev*. vLLM Ascend uses VLLM_VERSION to select its compatible upstream
# API branch; the deployment owner's known-good environment pins it to 0.21.0.
export VLLM_VERSION="${VLLM_COMPAT_VERSION}"
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_IF_IP="${NODE_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
unset VLLM_ASCEND_ENABLE_MLAPO
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_ENABLE_FUSED_MC2

# B0 deliberately keeps only the compatibility, logging, timeout and network
# contract needed to launch the pinned image on the fixed two-node topology.
# The explicit-candidate path retains the already validated deployment settings
# used by Agent rounds.  Keeping these branches separate prevents inherited
# expert settings from silently contaminating the official-default baseline.
if [[ "${LAUNCH_PROFILE}" == "official_source_defaults_deployable" ]]; then
  unset HCCL_OP_EXPANSION_MODE
  unset VLLM_ASCEND_BALANCE_SCHEDULING
  unset OMP_PROC_BIND
  unset OMP_NUM_THREADS
  unset HCCL_BUFFSIZE
  unset PYTORCH_NPU_ALLOC_CONF
  unset ASCEND_LAUNCH_BLOCKING
else
  export HCCL_OP_EXPANSION_MODE=AIV
  export VLLM_ASCEND_BALANCE_SCHEDULING=0
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export HCCL_BUFFSIZE=400
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export ASCEND_LAUNCH_BLOCKING=0
fi

FIXED_ENVIRONMENT_FILE="${RUN_DIR}/fixed_environment.sh"
python3 - "${FIXED_ENVIRONMENT_JSON}" "${FIXED_ENVIRONMENT_FILE}" <<'PY'
import json
import re
import shlex
import sys
from pathlib import Path

value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit("FIXED_ENVIRONMENT_JSON must be an object")
lines = []
for name, setting in value.items():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise SystemExit(f"invalid fixed environment name: {name!r}")
    if setting is None:
        lines.append(f"unset {name}")
    else:
        lines.append(f"export {name}={shlex.quote(str(setting))}")
Path(sys.argv[2]).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
PY
source "${FIXED_ENVIRONMENT_FILE}"

bool_flag() {
  [[ "${1}" == "true" ]]
}

VLLM_COMMON_ARGS=(
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --quantization "${MODEL_QUANTIZATION}"
)

case "${EXECUTOR_REMOTE_CONTRACT}" in
  legacy_two_role_v1)
    VLLM_COMMON_ARGS+=(
      --data-parallel-address "${MASTER_IP}"
      --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}"
    )
    ;;
  single_node_local_dp_v1)
    if [[ "${WORKER_REPLICAS}" != 0 || "${DATA_PARALLEL_SIZE}" != "${DATA_PARALLEL_SIZE_LOCAL}" ]]; then
      echo "Invalid single-node local-DP topology contract" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported EXECUTOR_REMOTE_CONTRACT=${EXECUTOR_REMOTE_CONTRACT}" >&2
    exit 2
    ;;
esac

while IFS= read -r -d '' fixed_arg; do
  VLLM_COMMON_ARGS+=("${fixed_arg}")
done < <(python3 - "${FIXED_CLI_ARGS_JSON}" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise SystemExit("FIXED_CLI_ARGS_JSON must be a JSON string array")
for item in value:
    if "\x00" in item:
        raise SystemExit("fixed CLI arguments cannot contain NUL")
    sys.stdout.buffer.write(item.encode("utf-8") + b"\0")
PY
)

# DTFS loading is a deployment transport contract, not an Agent-tuned serving
# parameter. Apply it equally to B0 and later candidates so the source-default
# baseline does not fall back to high-latency lazy mmap across all TP workers.
VLLM_COMMON_ARGS+=(
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"
  --safetensors-prefetch-num-threads "${SAFETENSORS_PREFETCH_NUM_THREADS}"
  --safetensors-prefetch-block-size "${SAFETENSORS_PREFETCH_BLOCK_SIZE}"
)

# B0-deployable preserves the official source-default serving configuration and
# injects exactly one compatibility override. The pinned model resolves a
# 1,048,576-token context, which needs 107.25 GiB of KV cache on this topology;
# the measured deployment exposes 28.82 GiB. Keep the 64k exception explicit
# in the sole B0 launch identity.
if [[ "${LAUNCH_PROFILE}" == "official_source_defaults_deployable" ]]; then
  VLLM_COMMON_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ "${LAUNCH_PROFILE}" == "explicit_candidate" ]]; then
  GENERATED_JSON_CONFIGS="${RUN_DIR}/generated_json_configs.json"
  GENERATED_ARGS_FILE="${RUN_DIR}/generated_cli_args.nul"
  GENERATED_ENV_FILE="${RUN_DIR}/generated_environment.sh"
  : > "${GENERATED_ARGS_FILE}"
  printf '{}\n' > "${GENERATED_JSON_CONFIGS}"
  : > "${GENERATED_ENV_FILE}"
  if [[ "${RUNTIME_INJECTION_MODE}" == "generated_v1" ]]; then
    [[ -n "${RUNTIME_INJECTION_PAYLOAD_B64}" ]] || {
      echo "generated_v1 requires RUNTIME_INJECTION_PAYLOAD_B64" >&2
      exit 2
    }
    python - "${RUNTIME_INJECTION_PAYLOAD_B64}" "${GENERATED_JSON_CONFIGS}" \
      "${GENERATED_ARGS_FILE}" "${GENERATED_ENV_FILE}" <<'PY'
import base64
import json
import re
import shlex
import sys
from pathlib import Path

encoded, json_path, args_path, env_path = sys.argv[1:]
payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
if payload.get("schema_version") != 1:
    raise SystemExit("unsupported generated runtime payload")
configs = payload.get("json_configs", {})
root_flags = {
    "attention_config": "--attention-config",
    "eplb_config": "--eplb-config",
    "kv_transfer_config": "--kv-transfer-config",
    "offload_config": "--offload-config",
}
args = [str(item) for item in payload.get("cli_args", [])]
for root in sorted(set(configs) - {"compilation_config", "additional_config", "speculative_config"}):
    if root not in root_flags:
        raise SystemExit(f"unsupported generated JSON config root: {root}")
    args.extend([root_flags[root], json.dumps(configs[root], separators=(",", ":"))])
Path(json_path).write_text(json.dumps(configs, separators=(",", ":")) + "\n", encoding="utf-8")
Path(args_path).write_bytes(b"".join(item.encode("utf-8") + b"\0" for item in args))
lines = []
for name, value in payload.get("environment", {}).items():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise SystemExit(f"invalid generated environment name: {name!r}")
    if value is None:
        lines.append(f"unset {name}")
    else:
        lines.append(f"export {name}={shlex.quote(str(value))}")
Path(env_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
PY
    source "${GENERATED_ENV_FILE}"
  elif [[ "${RUNTIME_INJECTION_MODE}" != "native_v1" ]]; then
    echo "Unsupported RUNTIME_INJECTION_MODE=${RUNTIME_INJECTION_MODE}" >&2
    exit 2
  fi

  VLLM_COMMON_ARGS+=(
    --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}"
    --seed 1024
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
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
    "${COMPILATION_ENABLE_SP}" "${CUDAGRAPH_CAPTURE_SIZES_JSON}" \
    "${GENERATED_JSON_CONFIGS}" <<'PY'
import json
import sys

mode, maximum, enable_sp, sizes_json, generated_path = sys.argv[1:]
config = {
    "cudagraph_mode": mode,
    "max_cudagraph_capture_size": int(maximum),
    "pass_config": {"enable_sp": enable_sp == "true"},
}
sizes = json.loads(sizes_json)
if sizes is not None:
    config["cudagraph_capture_sizes"] = sizes
def merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value
with open(generated_path, encoding="utf-8") as handle:
    generated = json.load(handle)
merge(config, generated.get("compilation_config", {}))
print(json.dumps(config, separators=(",", ":")))
PY
  )
  VLLM_COMMON_ARGS+=(--compilation-config "${COMPILATION_CONFIG_JSON}")
  ADDITIONAL_CONFIG_JSON=$(
  python - "${ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1}" \
    "${ADDITIONAL_CONFIG_ENABLE_MLAPO}" \
    "${ADDITIONAL_CONFIG_ENABLE_FUSED_MC2}" \
    "${ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING}" \
    "${ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE}" \
    "${GENERATED_JSON_CONFIGS}" "${FIXED_ADDITIONAL_CONFIG_JSON}" <<'PY'
import json
import sys

flashcomm1, mlapo, fused_mc2, balance, reduce_sample, generated_path, fixed_json = sys.argv[1:]
config = {
    "enable_npugraph_ex": True,
    "fuse_muls_add": True,
    "multistream_overlap_shared_expert": True,
    "enable_flashcomm1": flashcomm1 == "true",
    "enable_mlapo": mlapo == "true",
    "enable_fused_mc2": int(fused_mc2),
    "enable_balance_scheduling": balance == "true",
    "enable_reduce_sample": reduce_sample == "true",
}
fixed = json.loads(fixed_json)
if not isinstance(fixed, dict):
    raise SystemExit("FIXED_ADDITIONAL_CONFIG_JSON must be an object")
config.update(fixed)
def merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value
with open(generated_path, encoding="utf-8") as handle:
    generated = json.load(handle)
merge(config, generated.get("additional_config", {}))
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
    SPECULATIVE_CONFIG_JSON=$(
    python - "${MTP_DRAFT_MODEL_PATH}" "${NUM_SPECULATIVE_TOKENS}" \
      "${GENERATED_JSON_CONFIGS}" "${RUNTIME_INJECTION_MODE}" \
      "${SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON}" <<'PY'
import json
import sys

model, tokens, generated_path, injection_mode, enforce_eager_json = sys.argv[1:]
config = {"model": model, "num_speculative_tokens": int(tokens)}
if injection_mode == "native_v1":
    config["method"] = "mtp"
enforce_eager = json.loads(enforce_eager_json)
if enforce_eager is not None:
    config["enforce_eager"] = enforce_eager
with open(generated_path, encoding="utf-8") as handle:
    generated = json.load(handle)
config.update(generated.get("speculative_config", {}))
if not config.get("method"):
    raise SystemExit("speculative decoding requires an explicit generated method")
print(json.dumps(config, separators=(",", ":")))
PY
    )
    VLLM_COMMON_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG_JSON}")
  fi
  if (( LONG_PREFILL_TOKEN_THRESHOLD > 0 )); then
    VLLM_COMMON_ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
  fi
  while IFS= read -r -d '' generated_arg; do
    VLLM_COMMON_ARGS+=("${generated_arg}")
  done < "${GENERATED_ARGS_FILE}"
fi

{
  printf 'vllm serve %q ' "${MODEL_PATH}"
  printf '%q ' "${VLLM_COMMON_ARGS[@]}"
  printf '\n'
} > "${RUN_DIR}/vllm_common_command.txt"
