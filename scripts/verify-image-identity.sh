#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${VLLMTKB_PYTHON:-python3}" "${ROOT}/tuning_pipeline/workflow/continuous/image_identity_cli.py" "$@"
