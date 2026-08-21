#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

SOURCE_VENDOR="/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190/workflow/auto/vendor"
TARGET_VENDOR="${REPO_ROOT}/workflow/auto/vendor"

[[ -d "${SOURCE_VENDOR}" ]] || {
  echo "Approved source vendor directory is missing: ${SOURCE_VENDOR}" >&2
  exit 2
}
mkdir -p "${TARGET_VENDOR}"
cp --archive --no-clobber "${SOURCE_VENDOR}/." "${TARGET_VENDOR}/"
python3 "${SCRIPT_DIR}/prepare_decode_only_benchmark.py" \
  --allowed-root "${REPO_ROOT}" \
  --source-spec-root "${TARGET_VENDOR}/benchmark-tuning-fast-c32-v2/spec" \
  --target-spec-root "${TARGET_VENDOR}/benchmark-tuning-decode-only-c32-v2/spec" \
  --suite-overlay "${REPO_ROOT}/tuning_pipeline/workflow/continuous/benchmark_assets/decode-only-c32-v2/spec/suites/01_调优_Decode单场景-v2.yaml"
echo "Immutable Benchmark assets seeded without changing the existing project."
