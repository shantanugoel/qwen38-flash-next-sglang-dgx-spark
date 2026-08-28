#!/usr/bin/env python3
"""GSM8K sanity slice. Not a leaderboard number - a regression gate.

    N=20 python3 bench/gsm8k.py
    N=20 THINKING=on OUT=results/x/gsm8k.json python3 bench/gsm8k.py

Pulls the first N of the official test split (fixed order, no sampling) into
data/gsm8k/test.jsonl on first run.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat_stream  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "gsm8k"
LOCAL = DATA / "test.jsonl"
URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
N = int(os.environ.get("N", "20"))
THINKING = os.environ.get("THINKING", "off").lower() == "on"
MAX_TOKENS = int(os.environ.get("GSM8K_MAX_TOKENS", "2048"))
OUT = os.environ.get("OUT", "")

PROMPT = (
    "{q}\n\n"
    "Solve it, then give the final answer on its own last line as "
    "'#### <number>'."
)


def ensure_data() -> list[dict]:
    if not LOCAL.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            LOCAL.write_bytes(r.read())
    rows = []
    with open(LOCAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold(answer: str) -> str:
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


def predicted(text: str) -> str | None:
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> int:
    rows = ensure_data()[:N]
    hits = 0
    recs, ttfts, rates = [], [], []
    pt = ct = cached = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        r = chat_stream(
            [{"role": "user", "content": PROMPT.format(q=row["question"])}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            thinking=THINKING,
        )
        text = r["content"] or r["reasoning"]
        want = gold(row["answer"])
        got = predicted(text)
        good = same(got, want)
        hits += good
        pt += r["prompt_tokens"]
        ct += r["completion_tokens"]
        cached += r["cached_tokens"]
        ttfts.append(r["ttft_s"])
        rates.append(r["decode_tps"])
        recs.append({"i": i, "pass": good, "want": want, "got": got})
        print(
            f"{'PASS' if good else 'FAIL'} gsm8k[{i}] want={want} got={got} "
            f"tok={r['completion_tokens']} {r['decode_tps']} tok/s",
            flush=True,
        )
    summary = {
        "n": len(rows),
        "correct": hits,
        "accuracy": round(100 * hits / max(1, len(rows)), 1),
        "thinking": "on" if THINKING else "off",
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cached_tokens": cached,
        "cache_hit_pct": round(100 * cached / max(1, pt), 1),
        "median_ttft_s": round(statistics.median(ttfts), 3) if ttfts else 0,
        "median_decode_tps": round(statistics.median(rates), 2) if rates else 0,
    }
    print("RESULT gsm8k: " + json.dumps(summary), flush=True)
    if OUT:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps({"summary": summary, "results": recs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
