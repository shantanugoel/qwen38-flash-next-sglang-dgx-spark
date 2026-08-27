#!/usr/bin/env bash
# Health + numeric smoke. Optional: wait for /health (first boot is slow).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
BASE="http://${HOST}:${PORT}"
WAIT="${WAIT:-0}"

if [[ "${WAIT}" != 0 ]]; then
  echo "waiting for ${BASE}/health (up to ${WAIT}s)..."
  deadline=$((SECONDS + WAIT))
  until curl -sf -m 5 "${BASE}/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "not ready" >&2
      exit 1
    fi
    sleep 5
  done
fi

echo ">> health"
curl -sf -m 10 "${BASE}/health" >/dev/null && echo "   OK"

echo ">> 12*17"
python3 - "${BASE}" "${SERVED_NAME}" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 2048,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=300))
text = (r["choices"][0]["message"].get("content") or "").strip()
print("  ", repr(text))
if "204" not in text:
    raise SystemExit("expected 204 in content")
print("   OK")
PY
