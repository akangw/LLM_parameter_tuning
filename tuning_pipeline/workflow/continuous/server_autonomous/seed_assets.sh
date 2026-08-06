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
echo "Immutable Benchmark assets seeded without changing the existing project."
