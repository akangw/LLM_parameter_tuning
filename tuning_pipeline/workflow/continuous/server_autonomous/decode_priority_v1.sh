#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRACKED_CONFIG="${SCRIPT_DIR}/config.dp4_tp8.decode_priority_v1.yaml"
LOCAL_CONFIG="${SCRIPT_DIR}/config.dp4_tp8.decode_priority_v1.local.yaml"
if [[ -n "${VLLMTKB_DECODE_CONFIG:-}" ]]; then
  export VLLMTKB_CONFIG="${VLLMTKB_DECODE_CONFIG}"
elif [[ -f "${LOCAL_CONFIG}" ]]; then
  export VLLMTKB_CONFIG="${LOCAL_CONFIG}"
else
  export VLLMTKB_CONFIG="${TRACKED_CONFIG}"
fi
export VLLMTKB_RUNTIME_ROOT="${SCRIPT_DIR}/runtime_decode_priority_v1_live"

COMMAND="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${COMMAND}" in
  dry-run) TARGET="${SCRIPT_DIR}/dry_run.sh" ;;
  seed-assets) TARGET="${SCRIPT_DIR}/seed_assets.sh" ;;
  prepare-lease) TARGET="${SCRIPT_DIR}/prepare_lease.sh" ;;
  preflight) TARGET="${SCRIPT_DIR}/preflight.sh" ;;
  start) TARGET="${SCRIPT_DIR}/start.sh" ;;
  foreground) TARGET="${SCRIPT_DIR}/run_foreground.sh" ;;
  service) TARGET="${SCRIPT_DIR}/service.sh" ;;
  status) TARGET="${SCRIPT_DIR}/status.sh" ;;
  stop) TARGET="${SCRIPT_DIR}/stop.sh" ;;
  *)
    echo "usage: $0 {dry-run|seed-assets|prepare-lease|preflight|start|foreground|service|status|stop} [args...]" >&2
    exit 2
    ;;
esac

exec bash "${TARGET}" "$@"
