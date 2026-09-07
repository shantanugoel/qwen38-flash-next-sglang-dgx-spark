#!/usr/bin/env python3
"""Long-context smokes: needle recall + prefix-cache TTFT at 8k and 32k.

    SIZES=8k,32k python3 bench/longctx.py
Do not run 128k/240k as a default — sequential long prefills have wedged this box.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, model  # noqa: E402
import urllib.request

OUT = os.environ.get("OUT", "")
SIZES = os.environ.get("SIZES", "8k,32k")
NEEDLE = "NIGHTINGALE-ACCESS-CODE=7K-QUARTZ-19"
FILLER = (
    "The quarterly operations log continues with routine warehouse notes, "
    "shift handovers, pallet counts, and weather asides. Nothing in this "
    "paragraph is the secret. "
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def haystack(target_tokens: int) -> str:
    body = []
    n = 0
    while n < target_tokens // 2:
        body.append(FILLER)
        n = approx_tokens(" ".join(body))
    mid = (
        f" IGNORE previous noise. The invented fact is: {NEEDLE}. "
        "Do not mention it until asked. "
    )
    body.append(mid)
    while n < target_tokens - 80:
        body.append(FILLER)
        n = approx_tokens(" ".join(body))
    return "".join(body)


def completions(messages, max_tokens=64, temperature=0, thinking=False, timeout=1800):
    # reuse chat()
    return chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        timeout=timeout,
    )


def needle_and_cache(label: str, tokens: int) -> dict:
    hs = haystack(tokens)
    ask = (
        f"{hs}\n\nQuestion: What is the NIGHTINGALE access code? "
        "Reply with the code only."
    )
    msgs = [{"role": "user", "content": ask}]
    print(f">> needle {label} first (~{tokens} tok)", flush=True)
    r1 = completions(msgs, max_tokens=64, thinking=False)
    text = (r1["content"] or "") + " " + (r1["reasoning"] or "")
    hit = "7K-QUARTZ-19" in text.replace(" ", "")
    print(
        f"RESULT needle_{label}: {'PASS' if hit else 'FAIL'} "
        f"ttft={r1['seconds']:.2f}s prompt={r1['prompt_tokens']} "
        f"content={r1['content']!r}",
        flush=True,
    )
    print(f">> prefix-cache {label} resend", flush=True)
    r2 = completions(msgs, max_tokens=16, thinking=False)
    speedup = (r1["seconds"] / r2["seconds"]) if r2["seconds"] else 0.0
    print(
        f"RESULT cache_{label}: first={r1['seconds']:.2f}s "
        f"second={r2['seconds']:.2f}s speedup={speedup:.1f}x "
        f"prompt={r2['prompt_tokens']}",
        flush=True,
    )
    return {
        "size": label,
        "target_tokens": tokens,
        "needle_pass": hit,
        "first_s": r1["seconds"],
        "second_s": r2["seconds"],
        "speedup": round(speedup, 2),
        "prompt_tokens": r1["prompt_tokens"],
        "answer": r1["content"],
    }


def main() -> int:
    mapping = {"8k": 8000, "32k": 32000, "40k": 40000, "120k": 120000, "128k": 128000, "190k": 190000, "210k": 210000}
    rows = []
    for part in SIZES.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in mapping:
            print(f"unknown size {part}", file=sys.stderr)
            return 1
        rows.append(needle_and_cache(part, mapping[part]))
    failed = [r for r in rows if not r["needle_pass"]]
    report = {"results": rows, "failed": len(failed)}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(
        f"RESULT longctx: {len(rows)-len(failed)}/{len(rows)} needles pass",
        flush=True,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
