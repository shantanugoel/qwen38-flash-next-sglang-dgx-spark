#!/usr/bin/env python3
"""Repeat one deterministic-sampling case to measure boot-to-boot instability.

    N=20 python3 bench/effort_probe.py

The `effort_thinking_off` quality case (a $50 change question whose answer is
24) has passed on some boots of the same accepted configuration and failed on
others with `28` or `25`. A single sample cannot tell a regression from an
unstable case, so this probe resends the identical request N times at
temperature 0 and reports the answer distribution. It is descriptive: it never
decides the quality gate.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

OUT = os.environ.get("OUT", "")
N = int(os.environ.get("N", os.environ.get("EFFORT_PROBE_N", "20")))
THINKING = os.environ.get("THINKING", "off") == "on"
QUESTION = (
    "A shop sells pens at $3 and notebooks at $7. "
    "If I buy 4 pens and 2 notebooks and pay with a $50 bill, "
    "what is my change? Reply with the integer dollar amount only."
)
EXPECTED = "24"


def main() -> int:
    rows = []
    for i in range(N):
        r = chat(
            [{"role": "user", "content": QUESTION}],
            max_tokens=2048,
            temperature=0,
            thinking=THINKING,
        )
        text = (r["content"] or "").strip()
        passed = EXPECTED in text
        rows.append({
            "i": i,
            "pass": passed,
            "answer": text[:40],
            "completion_tokens": r["completion_tokens"],
            "seconds": r["seconds"],
        })
        print(f"RESULT effort_probe_{i}: {'PASS' if passed else 'FAIL'} answer={text[:20]!r}",
              flush=True)
    answers = collections.Counter(r["answer"] for r in rows)
    passed = sum(1 for r in rows if r["pass"])
    summary = {
        "n": N,
        "thinking": "on" if THINKING else "off",
        "expected": EXPECTED,
        "passed": passed,
        "answer_counts": dict(answers),
    }
    report = {"results": rows, "summary": summary, "failed": N - passed}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT effort_probe: {passed}/{N} answered {EXPECTED}; "
          f"answers={dict(answers)}", flush=True)
    # Descriptive probe: a stable-but-wrong or unstable case is data, not an
    # exit failure. Only a transport/API error (raised above) fails the run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
