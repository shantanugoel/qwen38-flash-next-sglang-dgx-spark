#!/usr/bin/env python3
"""Decode speed: code EN + prose, thinking on/off. Hashd1ve-shaped protocol.

    N=3 python3 bench/decode.py
    THINKING=off N=3 python3 bench/decode.py
    THINKING=both python3 bench/decode.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, tok_s  # noqa: E402

N = int(os.environ.get("N", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
MODE = os.environ.get("THINKING", "both").lower()  # on | off | both
OUT = os.environ.get("OUT", "")

PROMPTS = [
    ("code_en", "Write a Python function that merges two sorted lists. Code only."),
    (
        "prose_es",
        "Explica en tres frases que es la prescripcion de una sancion administrativa.",
    ),
]


def run_one(prompt: str, thinking: bool):
    r = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        thinking=thinking,
    )
    return {
        "tok_s": tok_s(r),
        "completion_tokens": r["completion_tokens"],
        "reasoning_tokens": r["reasoning_tokens"],
        "seconds": round(r["seconds"], 3),
        "finish": r["finish"],
        "content_preview": (r["content"] or r["reasoning"])[:160].replace("\n", " "),
    }


def sweep(thinking: bool) -> dict:
    label = "thinking_on" if thinking else "thinking_off"
    out = {"mode": label, "n": N, "tasks": {}}
    for name, prompt in PROMPTS:
        print(f">> warmup {label} {name}", flush=True)
        run_one(prompt, thinking)
        samples = []
        for i in range(N):
            print(f">> {label} {name} {i+1}/{N}", flush=True)
            samples.append(run_one(prompt, thinking))
        rates = [s["tok_s"] for s in samples]
        out["tasks"][name] = {
            "median_tok_s": round(statistics.median(rates), 2),
            "min_tok_s": round(min(rates), 2),
            "max_tok_s": round(max(rates), 2),
            "samples": samples,
        }
        print(
            f"RESULT {label} {name}: median {out['tasks'][name]['median_tok_s']} "
            f"tok/s ({out['tasks'][name]['min_tok_s']}-{out['tasks'][name]['max_tok_s']})",
            flush=True,
        )
    return out


def main() -> int:
    modes = []
    if MODE in ("off", "both"):
        modes.append(False)
    if MODE in ("on", "both"):
        modes.append(True)
    if not modes:
        print("THINKING must be on|off|both", file=sys.stderr)
        return 1
    report = {"results": [sweep(t) for t in modes]}
    text = json.dumps(report, indent=2)
    if OUT:
        Path(OUT).write_text(text)
        print(f"wrote {OUT}", flush=True)
    print("RESULT decode: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
