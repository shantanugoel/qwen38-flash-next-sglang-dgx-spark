#!/usr/bin/env python3
"""Concurrent 1/2/4-stream decode. Same code prompt as bench/decode.py.

    CONCURRENCIES=1,2,4 N=3 python3 bench/streams.py

Each concurrency warms once, then N parallel batches. Aggregate tok/s uses the
batch wall clock; per-stream tok/s uses each request's own duration.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, tok_s  # noqa: E402

N = int(os.environ.get("N", os.environ.get("STREAMS_N", "3")))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
THINKING = os.environ.get("THINKING", "off").lower() in ("1", "on", "true", "yes")
OUT = os.environ.get("OUT", "")
PROMPT = os.environ.get(
    "STREAMS_PROMPT",
    "Write a Python function that merges two sorted lists. Code only.",
)
CONCURRENCIES = [
    int(part.strip())
    for part in os.environ.get("CONCURRENCIES", "1,2,4").split(",")
    if part.strip()
]


def one(i: int) -> dict:
    try:
        r = chat(
            [{"role": "user", "content": f"{PROMPT} [stream {i}]"}],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            thinking=THINKING,
        )
    except (Exception, SystemExit) as exc:
        return {
            "tok_s": 0.0,
            "completion_tokens": 0,
            "seconds": 0.0,
            "finish": None,
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc)[:300],
        }
    return {
        "tok_s": tok_s(r),
        "completion_tokens": r["completion_tokens"],
        "seconds": round(r["seconds"], 3),
        "finish": r["finish"],
        "ok": r["completion_tokens"] > 0 and r["seconds"] > 0,
    }


def batch(concurrency: int) -> dict:
    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, i) for i in range(concurrency)]
        for fut in as_completed(futs):
            rows.append(fut.result())
    wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in rows)
    return {
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(toks / wall, 2) if wall else 0.0,
        "per_stream_tok_s": [r["tok_s"] for r in rows],
        "completion_tokens": toks,
        "ok": all(r["ok"] for r in rows) and len(rows) == concurrency,
        "streams": rows,
    }


def sweep(concurrency: int) -> dict:
    print(f">> warmup streams c={concurrency}", flush=True)
    warm = batch(concurrency)
    samples = []
    for i in range(N):
        print(f">> streams c={concurrency} {i + 1}/{N}", flush=True)
        samples.append(batch(concurrency))
    rates = [s["aggregate_tok_s"] for s in samples]
    per = [x for s in samples for x in s["per_stream_tok_s"]]
    out = {
        "concurrency": concurrency,
        "n": N,
        "thinking": THINKING,
        "warmup_ok": warm["ok"],
        "median_aggregate_tok_s": round(statistics.median(rates), 2) if rates else 0.0,
        "min_aggregate_tok_s": round(min(rates), 2) if rates else 0.0,
        "max_aggregate_tok_s": round(max(rates), 2) if rates else 0.0,
        "median_per_stream_tok_s": round(statistics.median(per), 2) if per else 0.0,
        "samples": samples,
        "ok": warm["ok"] and all(s["ok"] for s in samples) and bool(rates),
    }
    print(
        f"RESULT streams c={concurrency}: aggregate median "
        f"{out['median_aggregate_tok_s']} tok/s "
        f"({out['min_aggregate_tok_s']}-{out['max_aggregate_tok_s']}) "
        f"per-stream {out['median_per_stream_tok_s']}",
        flush=True,
    )
    return out


def main() -> int:
    if not CONCURRENCIES or any(c < 1 for c in CONCURRENCIES):
        print("CONCURRENCIES must be positive integers", file=sys.stderr)
        return 1
    results = [sweep(c) for c in CONCURRENCIES]
    failed = sum(1 for r in results if not r["ok"])
    report = {"results": results, "failed": failed}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT streams: {len(results) - failed}/{len(results)} concurrencies ok", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
