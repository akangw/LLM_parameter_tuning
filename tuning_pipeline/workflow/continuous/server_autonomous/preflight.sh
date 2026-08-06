#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
  echo "DEEPSEEK_API_KEY is not set." >&2
  exit 2
}
command -v ktp-lab >/dev/null || { echo "ktp-lab is not available." >&2; exit 2; }
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10); import jsonschema, packaging, pydantic, yaml'

"${PYTHON_BIN}" - <<'PY'
import json, os, urllib.request
request = urllib.request.Request(
    "https://api.deepseek.com/models",
    headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
models = {item.get("id") for item in payload.get("data", [])}
if "deepseek-v4-flash" not in models:
    raise SystemExit("deepseek-v4-flash is not available for this API key")
print("DeepSeek API/model preflight: OK")
PY

controller --check-only
