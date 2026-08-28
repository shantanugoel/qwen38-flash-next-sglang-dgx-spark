#!/usr/bin/env bash
# Poll OpenAI-compat /health. Never use docker logs -f.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
WAIT="${WAIT:-1500}"
INTERVAL="${INTERVAL:-15}"
BASE="http://${HOST}:${PORT}"

echo "waiting for ${BASE}/health up to ${WAIT}s (interval ${INTERVAL}s)"
deadline=$((SECONDS + WAIT))
while (( SECONDS < deadline )); do
  if curl -sf -m 5 "${BASE}/health" >/dev/null 2>&1; then
    echo "ready after ${SECONDS}s"
    exit 0
  fi
  echo "not ready (${SECONDS}s) container=$(docker inspect -f '{{.State.Status}}' "${CONTAINER}" 2>/dev/null || echo missing)"
  sleep "${INTERVAL}"
done
echo "not ready after ${WAIT}s" >&2
exit 1
