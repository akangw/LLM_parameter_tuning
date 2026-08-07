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

AGENT_PROVIDER=$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream) or {}
print((config.get("agent") or {}).get("provider", "deepseek"))
PY
)

if [[ "${AGENT_PROVIDER}" == "codex" ]]; then
  mapfile -d '' CODEX_SETTINGS < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream) or {}
settings = (((config.get("agent") or {}).get("providers") or {}).get("codex") or {})
for value in (
    settings.get("command", "codex"),
    settings.get("codex_home", ""),
    settings.get("profile", ""),
    "true" if settings.get("use_user_config", False) else "false",
    settings.get("tmp_dir", ""),
):
    print(str(value), end="\0")
PY
  )
  CODEX_COMMAND=${CODEX_SETTINGS[0]:-codex}
  CODEX_HOME_VALUE=${CODEX_SETTINGS[1]:-}
  CODEX_PROFILE=${CODEX_SETTINGS[2]:-}
  CODEX_USE_USER_CONFIG=${CODEX_SETTINGS[3]:-false}
  CODEX_TMP_DIR=${CODEX_SETTINGS[4]:-}
  [[ -x "${CODEX_COMMAND}" ]] || {
    echo "Configured Codex executable is unavailable: ${CODEX_COMMAND}" >&2
    exit 2
  }
  [[ -z "${CODEX_HOME_VALUE}" || -d "${CODEX_HOME_VALUE}" ]] || {
    echo "Configured CODEX_HOME is unavailable: ${CODEX_HOME_VALUE}" >&2
    exit 2
  }
  if [[ -n "${CODEX_HOME_VALUE}" ]]; then
    export CODEX_HOME="${CODEX_HOME_VALUE}"
  fi
  if [[ -n "${CODEX_TMP_DIR}" ]]; then
    [[ -d "${CODEX_TMP_DIR}" ]] || {
      echo "Configured Codex tmp_dir is unavailable: ${CODEX_TMP_DIR}" >&2
      exit 2
    }
    export TMPDIR="${CODEX_TMP_DIR}"
  fi
  CODEX_CONFIG_ARGS=()
  if [[ -n "${CODEX_PROFILE}" ]]; then
    CODEX_CONFIG_ARGS+=(--profile "${CODEX_PROFILE}")
  elif [[ "${CODEX_USE_USER_CONFIG}" != "true" ]]; then
    CODEX_CONFIG_ARGS+=(--ignore-user-config)
  fi
  CODEX_SMOKE_LOG="${SERVICE_ROOT}/codex-preflight-$(date -u +%Y%m%dT%H%M%SZ).jsonl"
  timeout 180 "${CODEX_COMMAND}" exec \
    "${CODEX_CONFIG_ARGS[@]}" \
    --ephemeral \
    --sandbox read-only \
    --skip-git-repo-check \
    -C "${REPO_ROOT}" \
    --json \
    'Return exactly the word OK and do not use tools.' \
    </dev/null >"${CODEX_SMOKE_LOG}" 2>&1
  "${PYTHON_BIN}" - "${CODEX_SMOKE_LOG}" <<'PY'
import json, sys
messages = []
completed = False
for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("type") == "turn.completed":
        completed = True
    item = event.get("item") or {}
    if item.get("type") == "agent_message":
        messages.append(str(item.get("text", "")).strip())
if not completed or "OK" not in messages:
    raise SystemExit("Codex/DeepSeek smoke test did not return a completed OK result")
print("Codex Agent over DeepSeek preflight: OK")
PY
fi

controller --check-only
