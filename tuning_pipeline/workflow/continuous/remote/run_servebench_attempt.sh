#!/bin/bash
set -eo pipefail

RESULT_PATH="${1:?usage: run_servebench_attempt.sh RESULT_PATH RESOLVED_TARGET}"
RESOLVED_TARGET="${2:?usage: run_servebench_attempt.sh RESULT_PATH RESOLVED_TARGET}"

: "${SERVEBENCH_WORKSPACE:?SERVEBENCH_WORKSPACE is required}"
: "${SERVEBENCH_ROOT:?SERVEBENCH_ROOT is required}"
: "${SERVEBENCH_DOCKER_IMAGE:?SERVEBENCH_DOCKER_IMAGE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${BENCHMARK_SPEC_ROOT:?BENCHMARK_SPEC_ROOT is required}"
: "${BENCHMARK_DATASET_ROOT:?BENCHMARK_DATASET_ROOT is required}"
: "${BENCHMARK_TOKENIZER:?BENCHMARK_TOKENIZER is required}"
: "${GUIDELLM_ACTIVATION:?GUIDELLM_ACTIVATION is required}"

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
    python -B -c '"'"'from importlib.metadata import version; actual=version("guidellm"); assert actual == "0.7.2", f"GuideLLM 0.7.2 required, got {actual}"'"'"'
    python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" plan "$SB_TEST_CONFIG" \
      --target-profile "$SB_TARGET_PATH" \
      --output-dir "$SB_RESULT_PATH" \
      --dataset-root "$SB_DATASET_ROOT" \
      "${phase_args[@]}"
    python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" run "$SB_RESULT_PATH"
    python -B servebench_cli.py --spec-root "$SB_SPEC_ROOT" report "$SB_RESULT_PATH"
  '
