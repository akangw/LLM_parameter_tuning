#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONTROLLER="${ROOT}/tuning_pipeline/workflow/continuous/continuous_tuning.py"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

CONFIG="${VLLMTKB_CONFIG:-}"
if [[ -z "${CONFIG}" ]]; then
  if [[ -f "${ROOT}/tuning_pipeline/workflow/continuous/config.local.yaml" ]]; then
    CONFIG="${ROOT}/tuning_pipeline/workflow/continuous/config.local.yaml"
  else
    CONFIG="${ROOT}/tuning_pipeline/workflow/continuous/config.yaml"
  fi
fi
RUNTIME_ROOT="${VLLMTKB_RUNTIME_ROOT:-${ROOT}/.runtime/controller}"
PYTHON="${VLLMTKB_PYTHON:-python3}"

common=(--runtime-root "${RUNTIME_ROOT}")
case "${ACTION}" in
  new)       exec "${PYTHON}" "${CONTROLLER}" --start --config "${CONFIG}" "${common[@]}" "$@" ;;
  resume)    exec "${PYTHON}" "${CONTROLLER}" --resume "${common[@]}" "$@" ;;
  retry)     exec "${PYTHON}" "${CONTROLLER}" --retry-paused-current "${common[@]}" "$@" ;;
  reanalyze) exec "${PYTHON}" "${CONTROLLER}" --reanalyze-current "${common[@]}" "$@" ;;
  check)     exec "${PYTHON}" "${CONTROLLER}" --check-only --config "${CONFIG}" "${common[@]}" "$@" ;;
  check-session) exec "${PYTHON}" "${CONTROLLER}" --check-only --use-frozen-session "${common[@]}" "$@" ;;
  prepare)   exec "${PYTHON}" "${CONTROLLER}" --prepare-lab --config "${CONFIG}" "${common[@]}" "$@" ;;
  status)    exec "${PYTHON}" "${CONTROLLER}" --status "${common[@]}" "$@" ;;
  offline-check) exec "${PYTHON}" "${CONTROLLER}" --offline-dry-run --config "${CONFIG}" "${common[@]}" "$@" ;;
  help|-h|--help)
    cat <<'EOF'
Usage: scripts/controller.sh ACTION [controller options]

Actions: new, resume, retry, reanalyze, check, check-session, prepare,
         status, offline-check

Environment:
  VLLMTKB_CONFIG        Local YAML override (defaults to config.local.yaml)
  VLLMTKB_RUNTIME_ROOT  Mutable state root (defaults to .runtime/controller)
  VLLMTKB_PYTHON        Python executable (defaults to python3)

Examples:
  scripts/controller.sh check --agent-provider deepseek
  scripts/controller.sh new --benchmark-profile vllm_bench_public_v1
EOF
    ;;
  *) echo "Unknown action: ${ACTION}" >&2; exit 2 ;;
esac
