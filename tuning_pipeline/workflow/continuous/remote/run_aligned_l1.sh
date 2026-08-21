#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${1:?usage: run_aligned_l1.sh RUN_ID LEASE_NAME}"
LEASE_NAME="${2:?usage: run_aligned_l1.sh RUN_ID LEASE_NAME}"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"

failure_marker() {
  local code="${1:-$?}"
  mkdir -p "${RUN_DIR}"
  printf '{"run_id":"%s","status":"failed","exit_code":%s,"updated_at":"%s"}\n' \
    "${RUN_ID}" "${code}" "$(date -Iseconds)" > "${RUN_DIR}/BENCHMARK_FAILED"
  exit "${code}"
}
trap 'failure_marker "$?"' ERR

# `set +e` does not suppress an ERR trap.  Every command below whose exit code
# is intentionally inspected must temporarily disarm the top-level terminal
# failure trap, otherwise the trap exits before the bounded retry branch runs.
begin_captured_failure() {
  trap - ERR
  set +e
}

end_captured_failure() {
  set -e
  trap 'failure_marker "$?"' ERR
}

PARAM_FILE="${RUN_DIR}/candidate.env"
[[ -f "${PARAM_FILE}" ]] || {
  echo "Missing candidate environment: ${PARAM_FILE}" >&2
  failure_marker 2
}
source "${PARAM_FILE}"

: "${SERVEBENCH_ROOT:?SERVEBENCH_ROOT is required}"
: "${SERVEBENCH_WORKSPACE:?SERVEBENCH_WORKSPACE is required}"
: "${SERVEBENCH_DOCKER_IMAGE:?SERVEBENCH_DOCKER_IMAGE is required}"
: "${GUIDELLM_ACTIVATION:?GUIDELLM_ACTIVATION is required}"
: "${BENCHMARK_SPEC_ROOT:?BENCHMARK_SPEC_ROOT is required}"
: "${BENCHMARK_DATASET_ROOT:?BENCHMARK_DATASET_ROOT is required}"
: "${BENCHMARK_TOKENIZER:?BENCHMARK_TOKENIZER is required}"
: "${BENCHMARK_SERVED_MODEL:?BENCHMARK_SERVED_MODEL is required}"
: "${BENCHMARK_EXPECTED_FINGERPRINT_JSON:?BENCHMARK_EXPECTED_FINGERPRINT_JSON is required}"

REPETITIONS="${BENCHMARK_REPETITIONS:-3}"
PRIMARY_CONCURRENCY="${BENCHMARK_PRIMARY_CONCURRENCY:-32}"
CASE_RETRY_LIMIT="${BENCHMARK_CASE_RETRY_LIMIT:-2}"
RUNTIME_RETRY_LIMIT="${BENCHMARK_RUNTIME_RETRY_LIMIT:-2}"
METRICS_RETRY_LIMIT="${BENCHMARK_METRICS_RETRY_LIMIT:-2}"
TOTAL_FULL_RETRY_LIMIT="${BENCHMARK_TOTAL_FULL_RETRY_LIMIT:-2}"
EXPECTED_FORMAL_CASES="${BENCHMARK_EXPECTED_FORMAL_CASES:-12}"
BENCHMARK_BUDGET_STARTED_EPOCH=$(date +%s)
BENCH_ROOT="${RUN_DIR}/servebench"
mkdir -p "${BENCH_ROOT}"

export SERVEBENCH_ROOT BENCHMARK_SPEC_ROOT BENCHMARK_DATASET_ROOT
export BENCHMARK_TOKENIZER BENCHMARK_EXPECTED_FINGERPRINT_JSON
export RUN_DIR SERVEBENCH_WORKSPACE SERVEBENCH_DOCKER_IMAGE GUIDELLM_ACTIVATION
export BENCHMARK_SUITE BENCHMARK_PHASE BENCHMARK_SUITE_ID
export BENCHMARK_PROFILE BENCHMARK_IDENTITY_JSON
python3 "${SCRIPT_DIR}/validate_aligned_l1_inputs.py"

KTP_JOB_ID=$(
  ktp-lab lease list | awk -v lease="${LEASE_NAME}" '
    $1 == "LEASE" && $2 == lease { found=1; next }
    found && /job-id=/ {
      if (match($0, /job-id=[0-9]+/)) {
        value=substr($0, RSTART + 7, RLENGTH - 7)
        print value
        exit
      }
    }
    found && $1 == "LEASE" { exit }
  '
)
[[ "${KTP_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "Unable to resolve KTP job ID for lease ${LEASE_NAME}" >&2
  exit 2
}

RUN_PATHS=()
RESOLVED_TARGETS=()
full_retries_used=0
for repetition in $(seq 1 "${REPETITIONS}"); do
  CONFIG_DIR="${BENCH_ROOT}/config-rep${repetition}"
  RESULT_ROOT="${BENCH_ROOT}/results"
  RESULT_NAME="l1-rep${repetition}"
  RESULT_PATH="${RESULT_ROOT}/${RESULT_NAME}"
  mkdir -p "${CONFIG_DIR}" "${RESULT_ROOT}"
  cat > "${CONFIG_DIR}/target.yaml" <<EOF
目标配置版本: servebench-target-v1
KTP任务号: "${KTP_JOB_ID}"
服务:
  发现方式: KTP
  端口: ${BENCHMARK_SERVICE_PORT:-8000}
  模型: ${BENCHMARK_SERVED_MODEL}
  Tokenizer: ${BENCHMARK_TOKENIZER}
  指标地址: 自动
部署:
  框架: vLLM
  服务控制文件: ${SERVEBENCH_WORKSPACE}/tools/ktp-lab/runtime/leases/${LEASE_NAME}/control/service.json
资源采样:
  模式: auto
  间隔秒: 5
EOF
  echo "Starting aligned L1 repetition ${repetition}/${REPETITIONS}"
  RESOLVED_TARGET="${CONFIG_DIR}/target-resolved.yaml"
  "${SERVEBENCH_ROOT}/servebench" target-resolve \
    --target-profile "${CONFIG_DIR}/target.yaml" \
    --output "${RESOLVED_TARGET}"
  begin_captured_failure
  docker run --rm --network host \
    --volume "${SERVEBENCH_WORKSPACE}:${SERVEBENCH_WORKSPACE}:ro" \
    --volume "${SERVEBENCH_WORKSPACE}/tools:/tools:ro" \
    --volume "${SERVEBENCH_ROOT}:${SERVEBENCH_ROOT}:ro" \
    --volume "${RUN_DIR}:${RUN_DIR}" \
    --volume "${BENCHMARK_SPEC_ROOT}:${BENCHMARK_SPEC_ROOT}:ro" \
    --volume "${BENCHMARK_DATASET_ROOT}:${BENCHMARK_DATASET_ROOT}:ro" \
    --volume "${BENCHMARK_TOKENIZER}:${BENCHMARK_TOKENIZER}:ro" \
    --workdir "${SERVEBENCH_ROOT}" \
    --env "SB_ENV_SCRIPT=${GUIDELLM_ACTIVATION}" \
    --env "SB_TEST_CONFIG=${BENCHMARK_SUITE:-tuning-fixed}" \
    --env "SB_TEST_PHASE=${BENCHMARK_PHASE:-}" \
    --env "SB_TARGET_PATH=${RESOLVED_TARGET}" \
    --env "SB_RESULT_PATH=${RESULT_PATH}" \
    --env "SB_SPEC_ROOT=${BENCHMARK_SPEC_ROOT}" \
    --env "SB_DATASET_ROOT=${BENCHMARK_DATASET_ROOT}" \
    --env "TORCH_DEVICE_BACKEND_AUTOLOAD=0" \
    "${SERVEBENCH_DOCKER_IMAGE}" \
    bash -lc '
      set -eo pipefail
      source "$SB_ENV_SCRIPT"
      phase_args=()
      if [[ -n "$SB_TEST_PHASE" && "$SB_TEST_PHASE" != all ]]; then
        phase_args=(--phase "$SB_TEST_PHASE")
      fi
      python -B -c '"'"'from importlib.metadata import version; actual=version("guidellm"); assert actual == "0.7.2", f"需要guidellm 0.7.2，实际{actual}"'"'"'
      python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" plan "$SB_TEST_CONFIG" \
        --target-profile "$SB_TARGET_PATH" \
        --output-dir "$SB_RESULT_PATH" \
        --dataset-root "$SB_DATASET_ROOT" \
        "${phase_args[@]}"
      python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" run "$SB_RESULT_PATH"
      python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" report "$SB_RESULT_PATH"
    '
  runtime_code=$?
  end_captured_failure
  if (( runtime_code != 0 )); then
    for runtime_attempt in $(seq 1 "${RUNTIME_RETRY_LIMIT}"); do
      if (( full_retries_used >= TOTAL_FULL_RETRY_LIMIT )); then
        break
      fi
      ((full_retries_used += 1))
      RESULT_PATH="${RESULT_ROOT}/${RESULT_NAME}-runtime-retry${runtime_attempt}"
      echo "Retrying the complete aligned-L1 repetition on the still-running service after a case runtime failure (attempt ${runtime_attempt}/${RUNTIME_RETRY_LIMIT})."
      begin_captured_failure
      bash "${SCRIPT_DIR}/run_servebench_attempt.sh" \
        "${RESULT_PATH}" "${RESOLVED_TARGET}"
      runtime_code=$?
      end_captured_failure
      printf '{"run_id":"%s","attempt":%s,"exit_code":%s,"result_path":"%s","updated_at":"%s"}\n' \
        "${RUN_ID}" "${runtime_attempt}" "${runtime_code}" "${RESULT_PATH}" "$(date -Iseconds)" \
        > "${RUN_DIR}/BENCHMARK_RUNTIME_RETRY_STATE"
      if (( runtime_code == 0 )); then
        cp "${RUN_DIR}/BENCHMARK_RUNTIME_RETRY_STATE" \
          "${RUN_DIR}/BENCHMARK_RUNTIME_RETRY_RECOVERED"
        break
      fi
    done
  fi
  (( runtime_code == 0 )) || {
    echo "Aligned L1 runtime failed after bounded same-service retries." >&2
    failure_marker 2
  }
  RUN_PATHS+=(--run "${RESULT_PATH}")
  RESOLVED_TARGETS+=("${RESOLVED_TARGET}")
done

RETRY_PLAN="${RUN_DIR}/benchmark_case_retry_plan.json"
run_metrics_gate() {
  begin_captured_failure
  python3 "${SCRIPT_DIR}/aligned_l1_metrics.py" \
    "${RUN_PATHS[@]}" \
    --primary-concurrency "${PRIMARY_CONCURRENCY}" \
    --expected-formal-cases "${EXPECTED_FORMAL_CASES}" \
    --output "${RUN_DIR}/metrics.pending.json" \
    --retry-plan-output "${RETRY_PLAN}"
  local code=$?
  end_captured_failure
  return "${code}"
}

metrics_ready=false
if run_metrics_gate; then
  metrics_ready=true
else
  for retry_number in $(seq 1 "${CASE_RETRY_LIMIT}"); do
    retryable=$(python3 -c \
      'import json,sys; print(str(json.load(open(sys.argv[1]))["retryable"]).lower())' \
      "${RETRY_PLAN}")
    [[ "${retryable}" == "true" ]] || break
    retry_run_dir=$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
      "${RETRY_PLAN}")
    retry_config=$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["config"])' \
      "${RETRY_PLAN}")
    retry_config_path="${retry_run_dir}/${retry_config}"
    retry_case_dir=$(dirname -- "${retry_config_path}")
    retry_archive="${RUN_DIR}/benchmark_case_retries/attempt_${retry_number}"
    mkdir -p "${retry_archive}"
    cp -a "${retry_case_dir}" "${retry_archive}/original_case"
    for report_name in result.json report.json RESULT.md; do
      if [[ -f "${retry_run_dir}/${report_name}" ]]; then
        cp -a "${retry_run_dir}/${report_name}" \
          "${retry_archive}/${report_name}"
      fi
    done
    cp -a "${RETRY_PLAN}" "${retry_archive}/retry_plan.json"
    echo "Retrying one incomplete aligned-L1 case in-place before service teardown: ${retry_config} (attempt ${retry_number}/${CASE_RETRY_LIMIT})"
    begin_captured_failure
    docker run --rm --network host \
      --volume "${SERVEBENCH_WORKSPACE}:${SERVEBENCH_WORKSPACE}:ro" \
      --volume "${SERVEBENCH_WORKSPACE}/tools:/tools:ro" \
      --volume "${SERVEBENCH_ROOT}:${SERVEBENCH_ROOT}:ro" \
      --volume "${RUN_DIR}:${RUN_DIR}" \
      --volume "${BENCHMARK_SPEC_ROOT}:${BENCHMARK_SPEC_ROOT}:ro" \
      --volume "${BENCHMARK_DATASET_ROOT}:${BENCHMARK_DATASET_ROOT}:ro" \
      --volume "${BENCHMARK_TOKENIZER}:${BENCHMARK_TOKENIZER}:ro" \
      --workdir "${SERVEBENCH_ROOT}" \
      --env "SB_ENV_SCRIPT=${GUIDELLM_ACTIVATION}" \
      --env "SB_CASE_CONFIG=${retry_config_path}" \
      --env "TORCH_DEVICE_BACKEND_AUTOLOAD=0" \
      "${SERVEBENCH_DOCKER_IMAGE}" \
      bash -lc '
        set -eo pipefail
        source "$SB_ENV_SCRIPT"
        python -B -c '\''import json,os; print(json.dumps({"id": 1, "config": os.environ["SB_CASE_CONFIG"]}))'\'' |
          python -B adapters/guidellm/session_worker.py
      ' | tee "${retry_archive}/retry.log"
    case_retry_code=${PIPESTATUS[0]}
    if (( case_retry_code == 0 )); then
      grep -q '"exit_code":0' "${retry_archive}/retry.log"
      case_retry_code=$?
    fi
    if (( case_retry_code == 0 )); then
      "${SERVEBENCH_ROOT}/servebench" report "${retry_run_dir}"
      case_retry_code=$?
    fi
    end_captured_failure
    if (( case_retry_code != 0 )); then
      echo "Targeted case retry failed; escalating to a full same-service metrics recovery."
      break
    fi
    if run_metrics_gate; then
      metrics_ready=true
      printf '{"run_id":"%s","status":"recovered","attempt":%s,"config":"%s","updated_at":"%s"}\n' \
        "${RUN_ID}" "${retry_number}" "${retry_config}" "$(date -Iseconds)" \
        > "${RUN_DIR}/BENCHMARK_CASE_RETRY_RECOVERED"
      break
    fi
  done
fi

# A clean ServeBench exit can still leave an unreadable/incomplete metrics tree
# (for example, GuideLLM can fail while compiling one report).  Do not tear down
# the expensive vLLM service for that packaging failure.  Re-run every repetition
# into fresh directories and re-apply the strict metrics gate on the same service.
if [[ "${metrics_ready}" != "true" ]]; then
  for metrics_attempt in $(seq 1 "${METRICS_RETRY_LIMIT}"); do
    if (( full_retries_used >= TOTAL_FULL_RETRY_LIMIT )); then
      break
    fi
    ((full_retries_used += 1))
    recovered_paths=()
    recovery_failed=false
    for repetition in $(seq 1 "${REPETITIONS}"); do
      recovery_path="${BENCH_ROOT}/results/l1-metrics-retry${metrics_attempt}-rep${repetition}"
      resolved_target="${RESOLVED_TARGETS[$((repetition - 1))]}"
      echo "Re-running aligned-L1 repetition ${repetition}/${REPETITIONS} on the still-running service after metrics-gate failure (attempt ${metrics_attempt}/${METRICS_RETRY_LIMIT})."
      begin_captured_failure
      bash "${SCRIPT_DIR}/run_servebench_attempt.sh" \
        "${recovery_path}" "${resolved_target}"
      recovery_code=$?
      end_captured_failure
      if (( recovery_code != 0 )); then
        recovery_failed=true
        break
      fi
      recovered_paths+=(--run "${recovery_path}")
    done
    printf '{"run_id":"%s","attempt":%s,"runtime_failed":%s,"updated_at":"%s"}\n' \
      "${RUN_ID}" "${metrics_attempt}" "${recovery_failed}" "$(date -Iseconds)" \
      > "${RUN_DIR}/BENCHMARK_METRICS_RETRY_STATE"
    if [[ "${recovery_failed}" == "false" ]]; then
      RUN_PATHS=("${recovered_paths[@]}")
      if run_metrics_gate; then
        metrics_ready=true
        cp "${RUN_DIR}/BENCHMARK_METRICS_RETRY_STATE" \
          "${RUN_DIR}/BENCHMARK_METRICS_RETRY_RECOVERED"
        break
      fi
    fi
  done
fi
[[ "${metrics_ready}" == "true" ]] || {
  echo "Aligned L1 metrics gate still failed after bounded targeted and full same-service retries." >&2
  failure_marker 2
}
BENCHMARK_WALL_SECONDS=$(( $(date +%s) - BENCHMARK_BUDGET_STARTED_EPOCH ))
python3 - "${RUN_DIR}/metrics.pending.json" \
  "${BENCHMARK_WALL_SECONDS}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("metrics", {})["benchmark_wall_time_seconds"] = int(sys.argv[2])
temporary = path.with_name(f".{path.name}.timing-{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
cp "${RUN_DIR}/metrics.pending.json" "${RUN_DIR}/metrics.json"

printf '{"run_id":"%s","status":"completed","repetitions":%s,"updated_at":"%s"}\n' \
  "${RUN_ID}" "${REPETITIONS}" "$(date -Iseconds)" > "${RUN_DIR}/BENCHMARK_DONE"
echo "Aligned L1 complete: ${BENCH_ROOT}"
