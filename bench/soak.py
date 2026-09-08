#!/usr/bin/env python3
"""Mixed-load soak against a live server. Sequential short chats, tools, prose."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

SECONDS = float(os.environ.get("SOAK_SECONDS", "3600"))
OUT = os.environ.get("OUT", "")

WEATHER = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

JOBS = (
    ("arith_off", {"messages": [{"role": "user", "content": "12*17"}],
                   "max_tokens": 64, "thinking": False, "temperature": 0.0}),
    ("prose_on", {"messages": [{"role": "user",
                                "content": "Say hello in one short Spanish sentence."}],
                  "max_tokens": 128, "thinking": True, "temperature": 0.7}),
    ("tool_off", {"messages": [{"role": "user",
                                "content": "What is the weather in Oslo?"}],
                  "max_tokens": 256, "thinking": False, "temperature": 0.0,
                  "tools": WEATHER}),
    ("code_off", {"messages": [{"role": "user",
                                "content": "Write a one-line Python identity function. Code only."}],
                  "max_tokens": 128, "thinking": False, "temperature": 0.7}),
)


def main() -> int:
    deadline = time.monotonic() + SECONDS
    rows = []
    errors = 0
    start = time.monotonic()
    i = 0
    while time.monotonic() < deadline:
        name, kwargs = JOBS[i % len(JOBS)]
        i += 1
        row = {"i": i, "name": name, "pass": False}
        try:
            result = chat(**kwargs)
            row["pass"] = True
            row["completion_tokens"] = result["completion_tokens"]
            row["seconds"] = round(result["seconds"], 3)
            row["finish"] = result["finish"]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            row["error"] = type(exc).__name__ + ": " + str(exc)[:200]
        rows.append(row)
        print("RESULT soak item:", name, "ok" if row["pass"] else "fail", flush=True)
    elapsed = round(time.monotonic() - start, 3)
    ok = sum(1 for r in rows if r["pass"])
    report = {
        "results": rows,
        "failed": errors,
        "summary": {
            "ok": ok,
            "errors": errors,
            "requests": len(rows),
            "seconds": elapsed,
            "target_seconds": SECONDS,
        },
    }
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
    print("RESULT soak:", ok, "/", len(rows), "ok in", elapsed, "s", flush=True)
    return 0 if errors == 0 and ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
