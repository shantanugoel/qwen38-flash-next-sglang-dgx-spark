#!/usr/bin/env bash
# Terminal-Bench fixed 8-task subset against the local server, via Harbor.
#
#   nohup ./scripts/bench_tb.sh > results/tb.log 2>&1 &
#
# Never run this in the foreground: each task builds and boots its own
# container and the subset takes tens of minutes. Poll results/tb.log.
#
# Dataset note: the public Harbor registry carries terminal-bench@2.0 (89
# tasks). 2.1 is the 2.0 point-release and is not separately registered, so we
# pin 2.0 and record that in RESEARCH_LOG.md. All 8 subset tasks exist in it.
#
# The agent runs inside a container, so it reaches the server through
# scripts/host_proxy.py on the docker bridge, not 127.0.0.1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

SERVED_NAME="${SERVED_NAME:-qwen38-flash-next-nvfp4-mtp}"
DATASET="${TB_DATASET:-terminal-bench@2.0}"
AGENT="${TB_AGENT:-terminus-2}"
JOB="${JOB:-tb-$(date -u +%Y%m%d-%H%M%S)}"
OUTDIR="${OUTDIR:-${ROOT}/results/tb}"
BRIDGE="${BRIDGE:-172.17.0.1}"
PROXY_PORT="${PROXY_PORT:-30001}"
AGENT_BASE="http://${BRIDGE}:${PROXY_PORT}/v1"

# The bridge proxy must be up: the local server binds loopback only.
if ! curl -sf -m 5 "http://${BRIDGE}:${PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "bridge proxy not answering on ${BRIDGE}:${PROXY_PORT}." >&2
  echo "start it: nohup python3 ${SCRIPT_DIR}/host_proxy.py > results/proxy.log 2>&1 &" >&2
  exit 1
fi

mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "${ROOT}/bench/subsets/tb21_8.txt")
args=()
for t in "${TASKS[@]}"; do args+=(-i "$t"); done

mkdir -p "${OUTDIR}"
echo "harbor run dataset=${DATASET} agent=${AGENT} model=openai/${SERVED_NAME} tasks=${#TASKS[@]}"
harbor run \
  -d "${DATASET}" \
  -a "${AGENT}" \
  -m "openai/${SERVED_NAME}" \
  "${args[@]}" \
  --ae "OPENAI_API_KEY=local" \
  --ae "OPENAI_BASE_URL=${AGENT_BASE}" \
  --ae "OPENAI_API_BASE=${AGENT_BASE}" \
  --allow-agent-host "${BRIDGE}" \
  -k 1 \
  -n 1 \
  --job-name "${JOB}" \
  --jobs-dir "${OUTDIR}" \
  -y
echo "RESULT tb: job ${JOB} under ${OUTDIR}"
