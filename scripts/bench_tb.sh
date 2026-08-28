#!/usr/bin/env bash
# Terminal-Bench 2.1 fixed 8-task subset against the local server.
#
#   nohup ./scripts/bench_tb.sh > results/tb.log 2>&1 &
#
# Never run this in the foreground: each task boots its own container and the
# whole subset takes tens of minutes. Poll results/tb.log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

PORT="${PORT:-30000}"
SERVED_NAME="${SERVED_NAME:-qwen38-flash-next-nvfp4-mtp}"
DATASET="${TB_DATASET:-terminal-bench==2.1}"
AGENT="${TB_AGENT:-terminus-2}"
RUN_ID="${RUN_ID:-tb21-$(date -u +%Y%m%d-%H%M%S)}"
OUTDIR="${OUTDIR:-${ROOT}/results/tb}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_BASE_URL="${OPENAI_API_BASE}"

mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "${ROOT}/bench/subsets/tb21_8.txt")
args=()
for t in "${TASKS[@]}"; do args+=(-t "$t"); done

mkdir -p "${OUTDIR}"
echo "tb run dataset=${DATASET} agent=${AGENT} model=openai/${SERVED_NAME} tasks=${#TASKS[@]}"
tb run \
  -d "${DATASET}" \
  -a "${AGENT}" \
  -m "openai/${SERVED_NAME}" \
  "${args[@]}" \
  --n-concurrent 1 \
  --output-path "${OUTDIR}" \
  --run-id "${RUN_ID}" \
  --no-upload-results
echo "RESULT tb: run-id ${RUN_ID} under ${OUTDIR}"
