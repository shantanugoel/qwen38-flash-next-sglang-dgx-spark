#!/usr/bin/env python3
"""64k (or sized) prefill arriving while two decode streams are in flight.

    N=3 python3 bench/mixedload.py

Reports cold prefill TTFT, decode streamed-chunk gaps during the prefill, and
tokens per chunk (chunk gaps are not token gaps). Decode prompts stay short;
the prefill haystack is salted so the radix prefix cache misses.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import base, model, chat_stream  # noqa: E402
from prefill import haystack  # noqa: E402

OUT = os.environ.get("OUT", "")
N = int(os.environ.get("N", os.environ.get("MIXEDLOAD_N", "3")))
PREFILL_TOKENS = int(os.environ.get("MIXEDLOAD_TOKENS", os.environ.get("PREFILL_TOKENS", "64000")))
DECODE_TOKENS = int(os.environ.get("MIXEDLOAD_DECODE_TOKENS", "1500"))
DECODE_STREAMS = int(os.environ.get("MIXEDLOAD_DECODE_STREAMS", "2"))
THINKING = False

DECODE_PROMPTS = [
    "Write a Python function that merges two sorted lists. Code only. Keep going with helpers and tests.",
    "Explica con detalle que es la prescripcion de una sancion administrativa. Varios parrafos.",
]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 6)


def start_stream(prompt: str, max_tokens: int, holder: dict) -> None:
    ttft_event = threading.Event()
    chunks: list[dict] = []

    def on_delta(rec: dict) -> None:
        chunks.append({"t_abs": time.perf_counter(), **rec})
        ttft_event.set()

    def run() -> None:
        try:
            result = chat_stream(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                thinking=THINKING,
                timeout=1800,
                on_delta=on_delta,
            )
            holder["result"] = result
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            holder["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        finally:
            ttft_event.set()
            holder["done"].set()

    holder["ttft_event"] = ttft_event
    holder["chunks"] = chunks
    holder["done"] = threading.Event()
    thread = threading.Thread(target=run, daemon=True)
    holder["thread"] = thread
    thread.start()


def gaps_in_window(chunks: list[dict], start: float, end: float) -> list[float]:
    times = [c["t_abs"] for c in chunks]
    times.sort()
    return [times[i] - times[i - 1] for i in range(1, len(times))
            if times[i] >= start and times[i - 1] <= end]


def tokens_per_chunk(result: dict | None) -> float | None:
    if not result:
        return None
    n_chunks = len(result.get("chunks") or [])
    ct = result.get("completion_tokens") or 0
    if n_chunks <= 0 or ct <= 0:
        return None
    return ct / n_chunks


def run_one(i: int) -> dict:
    salt = f"mixed-{i}-{os.getpid()}-{i * 10007}"
    hs = haystack(PREFILL_TOKENS, salt)
    prefill_msgs = [{
        "role": "user",
        "content": (
            f"{hs}\n\nQuestion: What is the NIGHTINGALE access code? "
            "Reply with the code only."
        ),
    }]
    # Size using the serving tokenizer; character estimates undershoot by ~25%.
    estimate = PREFILL_TOKENS
    for _ in range(6):
        req = urllib.request.Request(base() + "/tokenize", data=json.dumps({
            "model": model(), "messages": prefill_msgs,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            tokenized = json.load(response)
        count = tokenized.get("count", tokenized.get("num_tokens"))
        if count is None:
            count = len(tokenized["tokens"])
        if abs(count - PREFILL_TOKENS) <= 128:
            break
        estimate = round(estimate * PREFILL_TOKENS / count)
        prefill_msgs[0]["content"] = haystack(estimate, salt) + (
            "\n\nQuestion: What is the NIGHTINGALE access code? Reply with the code only.")
    else:
        raise RuntimeError("could not size mixed prefill within 128 tokens")
    load_start = time.perf_counter()
    holders = []
    for s in range(DECODE_STREAMS):
        holder: dict = {}
        prompt = DECODE_PROMPTS[s % len(DECODE_PROMPTS)] + f" [mix {i}.{s} {salt}]"
        start_stream(prompt, DECODE_TOKENS, holder)
        holders.append(holder)
    for holder in holders:
        if not holder["ttft_event"].wait(120):
            return {"i": i, "ok": False, "error": "decode TTFT timeout"}
        if holder.get("error"):
            return {"i": i, "ok": False, "error": holder["error"]}

    if any(h["done"].is_set() for h in holders):
        raise RuntimeError("decode finished before prefill arrival")

    print(f">> mixedload {i + 1}/{N} 64k-class prefill during {DECODE_STREAMS} decodes", flush=True)
    prefill_start = time.perf_counter()
    prefill = chat_stream(
        prefill_msgs, max_tokens=32, temperature=0, thinking=False, timeout=1800,
    )
    prefill_end = prefill_start + prefill["ttft_s"]
    for holder in holders:
        holder["done"].wait(1800)
        holder["thread"].join(timeout=5)

    text = (prefill["content"] or "") + " " + (prefill["reasoning"] or "")
    needle = "7K-QUARTZ-19" in text.replace(" ", "")
    overlap_gaps = []
    tpc = []
    decode_ok = True
    decode_rows = []
    for holder in holders:
        result = holder.get("result")
        if holder.get("error") or not result or result.get("completion_tokens", 0) <= 0:
            decode_ok = False
        overlap_gaps.extend(gaps_in_window(holder["chunks"], prefill_start, prefill_end))
        tpc.append(tokens_per_chunk(result))
        decode_rows.append({
            "completion_tokens": (result or {}).get("completion_tokens", 0),
            "seconds": (result or {}).get("seconds"),
            "n_chunks": len((result or {}).get("chunks") or []),
            "n_chunks_overlap": sum(
                1 for c in holder["chunks"] if prefill_start <= c["t_abs"] <= prefill_end
            ),
            "tokens_per_chunk": tokens_per_chunk(result),
            "error": holder.get("error"),
        })
    tpc_ok = [x for x in tpc if x is not None]
    row = {
        "i": i,
        "aggregate_output_tok_s": round((prefill["completion_tokens"] + sum(
            (h.get("result") or {}).get("completion_tokens", 0) for h in holders
        )) / (time.perf_counter() - load_start), 3),
        "prefill_prompt_tokens": prefill["prompt_tokens"],
        "prefill_ttft_s": round(prefill["ttft_s"], 3),
        "prefill_seconds": round(prefill["seconds"], 3),
        "prefill_tps": prefill["prefill_tps"],
        "prefill_needle_pass": needle,
        "decode_streams": DECODE_STREAMS,
        "decode": decode_rows,
        "n_chunk_gaps": len(overlap_gaps),
        "chunk_gap_s": {
            "p50": percentile(overlap_gaps, 50),
            "p95": percentile(overlap_gaps, 95),
            "p99": percentile(overlap_gaps, 99),
        },
        "tokens_per_chunk": {
            "median": round(statistics.median(tpc_ok), 3) if tpc_ok else None,
            "min": round(min(tpc_ok), 3) if tpc_ok else None,
            "max": round(max(tpc_ok), 3) if tpc_ok else None,
        },
        "ok": needle and decode_ok and prefill["ttft_s"] > 0 and prefill["prompt_tokens"] > 0,
    }
    print(
        f"RESULT mixedload {i + 1}: ttft={row['prefill_ttft_s']:.2f}s "
        f"prompt={row['prefill_prompt_tokens']} needle={'PASS' if needle else 'FAIL'} "
        f"gap_p50={row['chunk_gap_s']['p50']} gap_p95={row['chunk_gap_s']['p95']} "
        f"tok/chunk={row['tokens_per_chunk']['median']}",
        flush=True,
    )
    return row


def main() -> int:
    if DECODE_STREAMS < 1 or PREFILL_TOKENS < 1000:
        print("MIXEDLOAD_DECODE_STREAMS>=1 and MIXEDLOAD_TOKENS>=1000 required", file=sys.stderr)
        return 1
    print("Warm-up (excluded from measured repeats)", flush=True)
    warmup = run_one(-1)
    if not warmup.get("ok"):
        raise RuntimeError("mixed-load warm-up failed")
    rows = [run_one(i) for i in range(N)]
    failed = sum(1 for r in rows if not r.get("ok"))
    ttfts = [r["prefill_ttft_s"] for r in rows if r.get("ok")]
    p50s = [r["chunk_gap_s"]["p50"] for r in rows if r.get("ok") and r["chunk_gap_s"]["p50"] is not None]
    p95s = [r["chunk_gap_s"]["p95"] for r in rows if r.get("ok") and r["chunk_gap_s"]["p95"] is not None]
    p99s = [r["chunk_gap_s"]["p99"] for r in rows if r.get("ok") and r["chunk_gap_s"]["p99"] is not None]
    tpcs = [r["tokens_per_chunk"]["median"] for r in rows if r.get("ok") and r["tokens_per_chunk"]["median"] is not None]
    report = {
        "results": rows,
        "failed": failed,
        "summary": {
            "n": N,
            "prefill_tokens_target": PREFILL_TOKENS,
            "decode_streams": DECODE_STREAMS,
            "median_aggregate_output_tok_s": statistics.median([r["aggregate_output_tok_s"] for r in rows if r.get("ok")]) if ttfts else None,
            "median_prefill_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
            "median_chunk_gap_p50_s": round(statistics.median(p50s), 6) if p50s else None,
            "median_chunk_gap_p95_s": round(statistics.median(p95s), 6) if p95s else None,
            "median_chunk_gap_p99_s": round(statistics.median(p99s), 6) if p99s else None,
            "median_tokens_per_chunk": round(statistics.median(tpcs), 3) if tpcs else None,
        },
    }
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(
        f"RESULT mixedload: {len(rows) - failed}/{len(rows)} ok "
        f"median_ttft={report['summary']['median_prefill_ttft_s']} "
        f"tok/chunk={report['summary']['median_tokens_per_chunk']}",
        flush=True,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
