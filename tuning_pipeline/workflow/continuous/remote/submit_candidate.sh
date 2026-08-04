#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARAM_SOURCE="${1:?usage: submit_candidate.sh CANDIDATE_ENV [--dry-run]}"
MODE="${2:-}"
case "${PARAM_SOURCE}" in
  "${SCRIPT_DIR}/candidates/"*.env) ;;
  *) echo "Candidate must be inside ${SCRIPT_DIR}/candidates" >&2; exit 2 ;;
esac
[[ -f "${PARAM_SOURCE}" ]] || { echo "Candidate not found: ${PARAM_SOURCE}" >&2; exit 2; }

source "${PARAM_SOURCE}"
ROUND_LABEL="${ROUND_LABEL:?ROUND_LABEL is required}"
RUN_ID="${ROUND_LABEL}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
GENERATED_DIR="${SCRIPT_DIR}/generated"
GENERATED_YAML="${GENERATED_DIR}/experiment_${RUN_ID}.yaml"
mkdir -p "${RUN_DIR}" "${GENERATED_DIR}"
cp "${PARAM_SOURCE}" "${RUN_DIR}/candidate.env"
sed "s/__RUN_ID__/${RUN_ID}/g" "${SCRIPT_DIR}/experiment_loop.yaml" > "${GENERATED_YAML}"
cp "${GENERATED_YAML}" "${RUN_DIR}/task.yaml"

echo "Validating ${GENERATED_YAML}"
ktp submit --dry-run -f "${GENERATED_YAML}"
if [[ "${MODE}" == "--dry-run" ]]; then
  echo "DRY_RUN_COMPLETE"
else
  echo "Submitting ${RUN_ID}"
  ktp submit -f "${GENERATED_YAML}"
fi
echo "EXPERIMENT_RUN_ID=${RUN_ID}"
echo "GENERATED_YAML=${GENERATED_YAML}"
echo "RESULT_DIR=${RUN_DIR}"
