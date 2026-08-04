#!/bin/bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID="${1:?usage: benchmark_watchdog.sh RUN_ID LEASE_NAME}"
LEASE_NAME="${2:?usage: benchmark_watchdog.sh RUN_ID LEASE_NAME}"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
WAIT_SECONDS="${BENCHMARK_WATCHDOG_WAIT_SECONDS:-14400}"
LOCK_DIR="${RUN_DIR}/BENCHMARK_START_LOCK"

for _ in $(seq 1 "${WAIT_SECONDS}"); do
  if [[ -f "${RUN_DIR}/BENCHMARK_DONE" || -f "${RUN_DIR}/MASTER_DONE" ]]; then
    exit 0
  fi
  if [[ -f "${RUN_DIR}/SERVICE_READY" ]]; then
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      touch "${RUN_DIR}/BENCHMARK_STARTED"
      runner_pid="${RUN_DIR}/benchmark_runner.pid"
      runner_log="${RUN_DIR}/benchmark_runner.log"
      detached_runner="echo \$\$ > $(printf '%q' "${runner_pid}"); exec bash $(printf '%q' "${SCRIPT_DIR}/run_aligned_l1.sh") $(printf '%q' "${RUN_ID}") $(printf '%q' "${LEASE_NAME}") > $(printf '%q' "${runner_log}") 2>&1 < /dev/null"
      setsid -f bash -c "${detached_runner}"
      printf '{"run_id":"%s","started_by":"remote_watchdog","updated_at":"%s"}\n' \
        "${RUN_ID}" "$(date -Iseconds)" > "${RUN_DIR}/BENCHMARK_WATCHDOG_STARTED"
    fi
    exit 0
  fi
  sleep 1
done

printf '{"run_id":"%s","status":"ready_timeout","updated_at":"%s"}\n' \
  "${RUN_ID}" "$(date -Iseconds)" > "${RUN_DIR}/BENCHMARK_WATCHDOG_TIMEOUT"
exit 1
