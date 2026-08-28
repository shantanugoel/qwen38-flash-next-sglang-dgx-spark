#!/usr/bin/env bash
# One experiment: restart the server with a set of env knobs, wait for /health,
# run the fast bench suite, and leave everything under results/<TAG>/.
#
#   TAG=packA MEMFRAC=0.85 PREFILL=2048 MAX_RUNNING=2 CUDA_GRAPH_MAX_BS=8 \
#     nohup ./scripts/run_config.sh > results/run-packA.log 2>&1 &
#
# Never run this in the foreground: it contains a 10-20 min boot. Poll
# results/run-<TAG>.log. See AGENTS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

TAG="${TAG:?set TAG}"
BOOT_WAIT="${BOOT_WAIT:-2400}"

echo "=== ${TAG} restart $(date -u +%FT%TZ) ==="
env | grep -E '^(MEMFRAC|PREFILL|MAX_RUNNING|CONTEXT|MAX_TOTAL|SPEC|SPEC_STEPS|SPEC_TOPK|SPEC_DRAFT|CUDA_GRAPH_MAX_BS|EXTRA_ARGS)=' || true

docker rm -f qwen38-flash-next >/dev/null 2>&1 || true
"${SCRIPT_DIR}/serve.sh"

start=$SECONDS
until curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; do
  if (( SECONDS - start > BOOT_WAIT )); then
    echo "RESULT ${TAG}: BOOT TIMEOUT after $((SECONDS-start))s"
    docker logs --tail 40 qwen38-flash-next 2>&1 | tr '\r' '\n' | tail -20
    exit 1
  fi
  if ! docker ps -q -f name=qwen38-flash-next | grep -q .; then
    echo "RESULT ${TAG}: CONTAINER DIED after $((SECONDS-start))s"
    docker logs --tail 40 qwen38-flash-next 2>&1 | tr '\r' '\n' | tail -20
    exit 1
  fi
  sleep 15
done
boot=$((SECONDS - start))
echo "RESULT ${TAG}: boot ${boot}s"

docker logs qwen38-flash-next 2>&1 | tr '\r' '\n' \
  | grep -oE "PLE table: [0-9]+/[0-9]+ shards[^,]*|max_total_num_tokens=[0-9]+|KV Cache is allocated[^|]{0,90}" \
  | sed 's/^/RESULT '"${TAG}"' boot-fact: /' | sort -u

echo "RESULT ${TAG}: host_mem $(free -g | awk '/^Mem:/{print "used="$3"G free="$4"G cache="$6"G"}') gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tail -1)"

TAG="${TAG}" "${SCRIPT_DIR}/bench_fast.sh"
