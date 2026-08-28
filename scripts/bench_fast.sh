#!/usr/bin/env bash
# Fast bench suite for one loaded config. Never run in the foreground.
#
#   TAG=baseline nohup ./scripts/bench_fast.sh > results/fast-baseline.log 2>&1 &
#
# Writes results/<TAG>/{decode,quality,longctx}.json and prints RESULT lines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAG="${TAG:-run}"
OUTDIR="${ROOT}/results/${TAG}"
mkdir -p "${OUTDIR}"

# Two suites against one server measure each other, not the server. A stale
# chained "wait for /health then bench" job is easy to leave behind, so refuse
# to start a second one.
LOCK="${ROOT}/results/.bench.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another bench run holds ${LOCK}; refusing to run ${TAG} concurrently" >&2
  exit 1
fi
echo "$$ ${TAG} $(date -u +%FT%TZ)" >&9
export BASE="${BASE:-http://127.0.0.1:30000}"
export MODEL="${MODEL:-qwen38-flash-next-nvfp4-mtp}"

echo "=== ${TAG} $(date -u +%FT%TZ) ==="
"${SCRIPT_DIR}/smoke.sh"

echo "=== quality (math/tools/code/multiturn/vision + effort) ==="
EFFORT=1 OUT="${OUTDIR}/quality.json" python3 "${ROOT}/bench/quality.py" || true

echo "=== decode (thinking off + on) ==="
N="${N:-3}" THINKING=both OUT="${OUTDIR}/decode.json" python3 "${ROOT}/bench/decode.py" || true

echo "=== longctx (8k/32k needle + prefix cache) ==="
SIZES="${SIZES:-8k,32k}" OUT="${OUTDIR}/longctx.json" python3 "${ROOT}/bench/longctx.py" || true

echo "=== agentic session (${TURNS:-40} turns, tools) ==="
TURNS="${TURNS:-40}" OUT="${OUTDIR}/agentic.json" python3 "${ROOT}/bench/agentic.py" || true

echo "=== server metrics ==="
python3 - <<'PY'
import sys, json
sys.path.insert(0, "bench")
from client import server_metrics
print("RESULT metrics:", json.dumps(server_metrics()))
PY

echo "RESULT bench_fast: ${TAG} done"
