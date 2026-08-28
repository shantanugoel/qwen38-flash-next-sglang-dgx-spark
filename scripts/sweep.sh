#!/usr/bin/env bash
# Run several serve configs back to back, unattended.
#
#   nohup ./scripts/sweep.sh > results/sweep.log 2>&1 &
#
# Each entry is "TAG|env assignments". A config that fails to boot is recorded
# and the sweep moves on - a kernel backend with no sm_121 cubins is a result,
# not a reason to stop. Never run this in the foreground (AGENTS.md).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CONFIGS_FILE="${CONFIGS_FILE:-${ROOT}/results/sweep.configs}"
[[ -f "${CONFIGS_FILE}" ]] || { echo "no ${CONFIGS_FILE}"; exit 1; }

while IFS= read -r line; do
  line="${line%%#*}"
  [[ -z "${line// }" ]] && continue
  tag="${line%%|*}"
  envs="${line#*|}"
  echo ""
  echo "######## ${tag} :: ${envs} ########"
  if env ${envs} TAG="${tag}" "${SCRIPT_DIR}/run_config.sh"; then
    echo "SWEEP ${tag}: ok"
  else
    echo "SWEEP ${tag}: FAILED (see above)"
  fi
done < "${CONFIGS_FILE}"
echo "SWEEP: all configs done"
