#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_runtime_loop.sh"
exec > >(tee -a "${RUN_DIR}/master.log") 2>&1

VLLM_PID=""
COMPLETED=0
write_status() {
  local phase="$1"
  local outcome="${2:-running}"
  printf '{"run_id":"%s","phase":"%s","outcome":"%s","updated_at":"%s"}\n' \
    "${EXPERIMENT_RUN_ID}" "${phase}" "${outcome}" "$(date -Iseconds)" > "${RUN_DIR}/run_status.json"
}
write_startup_event() {
  local event="$1"
  printf '{"event":"%s","at":"%s","safetensors_load_strategy":"%s","prefetch_threads":%s,"prefetch_block_size":%s}\n' \
    "${event}" "$(date -Iseconds)" "${SAFETENSORS_LOAD_STRATEGY}" \
    "${SAFETENSORS_PREFETCH_NUM_THREADS}" "${SAFETENSORS_PREFETCH_BLOCK_SIZE}" \
    >> "${RUN_DIR}/startup_timeline.jsonl"
}
finish_experiment() {
  local code=$?
  touch "${RUN_DIR}/MASTER_DONE"
  if [[ "${COMPLETED}" != 1 ]]; then
    write_status "terminated" "failed"
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
  return "${code}"
}
trap finish_experiment EXIT TERM INT

echo "EXPERIMENT_RUN_ID=${EXPERIMENT_RUN_ID}"
echo "ROLE=master NODE_IP=${NODE_IP} MASTER_IP=${MASTER_IP} NIC_NAME=${NIC_NAME}"
write_status "starting_vllm"

case "${LAUNCH_PROFILE}" in
  official_source_defaults_deployable)
    SERVICE_VALUE_ORIGIN="source_resolved_with_max_model_len_override"
    ;;
  explicit_candidate)
    SERVICE_VALUE_ORIGIN="explicit_candidate_env"
    ;;
esac

cat > "${RUN_DIR}/effective_config.yaml" <<EOF
experiment_id: ${EXPERIMENT_RUN_ID}
launch_profile: ${LAUNCH_PROFILE}
model: ${MODEL_PATH}
topology: {pods: 2, npu_per_pod: 16, data_parallel_size: 2, tensor_parallel_size: 16}
service:
  port: ${SERVICE_PORT}
  runtime_injection_mode: ${RUNTIME_INJECTION_MODE}
  value_origin: ${SERVICE_VALUE_ORIGIN}
  max_num_seqs: ${MAX_NUM_SEQS}
  max_model_len: ${MAX_MODEL_LEN}
  max_num_batched_tokens: ${MAX_NUM_BATCHED_TOKENS}
  gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}
  enable_prefix_caching: ${ENABLE_PREFIX_CACHING}
  async_scheduling: ${ASYNC_SCHEDULING}
  enable_expert_parallel: ${ENABLE_EXPERT_PARALLEL}
  compilation_mode: ${COMPILATION_MODE}
  num_speculative_tokens: ${NUM_SPECULATIVE_TOKENS}
  long_prefill_token_threshold: ${LONG_PREFILL_TOKEN_THRESHOLD}
  enable_chunked_prefill: ${ENABLE_CHUNKED_PREFILL}
  max_cudagraph_capture_size: ${MAX_CUDAGRAPH_CAPTURE_SIZE}
  enable_eplb: ${ENABLE_EPLB}
  eplb_num_redundant_experts: ${EPLB_NUM_REDUNDANT_EXPERTS}
  compilation_enable_sp: ${COMPILATION_ENABLE_SP}
  cudagraph_capture_sizes: ${CUDAGRAPH_CAPTURE_SIZES_JSON}
  decode_context_parallel_size: ${DECODE_CONTEXT_PARALLEL_SIZE}
  flashcomm1: ${ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1}
  mlapo: ${ADDITIONAL_CONFIG_ENABLE_MLAPO}
  fused_mc2: ${ADDITIONAL_CONFIG_ENABLE_FUSED_MC2}
  enable_balance_scheduling: ${ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING}
  enable_reduce_sample: ${ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE}
  speculative_config__enforce_eager: ${SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON}
  safetensors_load_strategy: ${SAFETENSORS_LOAD_STRATEGY}
  safetensors_prefetch_num_threads: ${SAFETENSORS_PREFETCH_NUM_THREADS}
  safetensors_prefetch_block_size: ${SAFETENSORS_PREFETCH_BLOCK_SIZE}
benchmark:
  mode: ${BENCHMARK_MODE}
  temperature: 0
EOF

cat > "${RUN_DIR}/server_run_manifest.yaml" <<EOF
schema_version: 1
run_id: ${EXPERIMENT_RUN_ID}
run_root: ${RUN_DIR}
naming: <candidate-label>_<YYYYMMDD_HHMMSS>
authoritative_server_artifacts:
  configuration:
    - candidate.env
    - effective_config.yaml
    - vllm_common_command.txt
    - task.yaml
  generated_configuration_if_explicit_candidate:
    - generated_json_configs.json
    - generated_cli_args.nul
    - generated_environment.sh
  service_logs:
    - master.log
    - worker.log
    - run_status.json
    - startup_timeline.jsonl
  benchmark_logs:
    - benchmark_runner.log
    - benchmark_watchdog.log
    - servebench/
  result:
    - metrics.json
    - SERVICE_READY
    - BENCHMARK_STARTED
    - BENCHMARK_DONE
    - BENCHMARK_FAILED
    - MASTER_DONE
ktp_outer_logs: ${LAB_OUTPUT_ROOT}/${EXPERIMENT_RUN_ID}/service/rank-<000|001>.log
local_policy: core_logs_and_metrics_only
EOF

write_startup_event "vllm_process_start"
vllm serve "${MODEL_PATH}" --host 0.0.0.0 --port "${SERVICE_PORT}" "${VLLM_COMMON_ARGS[@]}" &
VLLM_PID=$!

write_status "waiting_for_api"
READY=0
READY_MAX_ATTEMPTS=1080
for attempt in $(seq 1 "${READY_MAX_ATTEMPTS}"); do
  if curl --fail --silent "http://127.0.0.1:${SERVICE_PORT}/v1/models" > "${RUN_DIR}/models_response.json"; then
    READY=1
    echo "vLLM API is ready."
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready."
    wait "${VLLM_PID}"
    exit 1
  fi
  if (( attempt % 6 == 0 )); then
    echo "Still waiting for vLLM API (${attempt}/${READY_MAX_ATTEMPTS})..."
  fi
  sleep 10
done
[[ "${READY}" == 1 ]] || { echo "Timed out waiting for vLLM API."; exit 1; }
write_startup_event "api_ready"

if [[ "${BENCHMARK_MODE}" == "legacy_random_32k1k" ]]; then
  write_status "legacy_warmup"
  vllm bench serve --served-model-name "${SERVED_MODEL_NAME}" --port "${SERVICE_PORT}" --backend vllm \
    --dataset-name random --random-input-len 32000 --random-output-len 1000 \
    --random-range-ratio 0.5 --request-rate 0.2 --num-prompts 8 --ignore-eos \
    --temperature 0 --trust-remote-code --seed 24 2>&1 | tee "${RUN_DIR}/warmup.log"

  write_status "legacy_formal_benchmark"
  vllm bench serve --served-model-name "${SERVED_MODEL_NAME}" --port "${SERVICE_PORT}" --backend vllm \
    --dataset-name random --random-input-len 32000 --random-output-len 1000 \
    --random-range-ratio 0.5 --request-rate 0.2 --num-prompts 8 --ignore-eos \
    --temperature 0 --trust-remote-code --seed 42 2>&1 | tee "${RUN_DIR}/formal.log"

  python "${SCRIPT_DIR}/extract_metrics.py" "${RUN_DIR}/formal.log" "${RUN_DIR}/metrics.json"
elif [[ "${BENCHMARK_MODE}" == "aligned_l1" ]]; then
  write_status "waiting_for_aligned_l1"
  touch "${RUN_DIR}/SERVICE_READY"
  BENCHMARK_WAIT_SECONDS="${BENCHMARK_WAIT_SECONDS:-39600}"
  for _ in $(seq 1 "${BENCHMARK_WAIT_SECONDS}"); do
    if [[ -f "${RUN_DIR}/BENCHMARK_DONE" ]]; then
      break
    fi
    if [[ -f "${RUN_DIR}/BENCHMARK_FAILED" ]]; then
      echo "Aligned L1 runner reported failure."
      exit 1
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited while the aligned L1 benchmark was running."
      wait "${VLLM_PID}"
      exit 1
    fi
    sleep 1
  done
  [[ -f "${RUN_DIR}/BENCHMARK_DONE" ]] || {
    echo "Timed out waiting for aligned L1 benchmark completion."
    exit 1
  }
  [[ -f "${RUN_DIR}/metrics.json" ]] || {
    echo "Aligned L1 completed without consolidated metrics."
    exit 1
  }
elif [[ "${BENCHMARK_MODE}" == "vllm_bench_serve" || "${BENCHMARK_MODE}" == "custom_adapter" ]]; then
  write_status "${BENCHMARK_MODE}"
  touch "${RUN_DIR}/BENCHMARK_STARTED"
  if ! python3 "${SCRIPT_DIR}/benchmark_driver.py" \
    --mode "${BENCHMARK_MODE}" --run-dir "${RUN_DIR}" --project-dir "${PROJECT_DIR}"; then
    touch "${RUN_DIR}/BENCHMARK_FAILED"
    exit 1
  fi
  touch "${RUN_DIR}/BENCHMARK_DONE"
else
  echo "Unsupported BENCHMARK_MODE=${BENCHMARK_MODE}" >&2
  exit 2
fi

write_status "completed" "success"
COMPLETED=1
echo "EXPERIMENT_COMPLETE RESULT_DIR=${RUN_DIR}"
