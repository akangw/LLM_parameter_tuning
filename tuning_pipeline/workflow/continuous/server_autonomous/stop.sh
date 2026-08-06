#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

touch "${RUNTIME_ROOT}/STOP_REQUESTED"
echo "Graceful stop requested. The controller will archive the active round and submit no next round."
