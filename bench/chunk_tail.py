#!/usr/bin/env python3
"""A1 probe: forwards with fewer than 4 token rows (sglang#38346 path).

A QSA extend forward with 1..3 token rows reads past `token_k` on the stock
pin. Three ways a real server produces one, each sent through native
/generate with exact token ids (salted per run so nothing is a stale cache hit):

  tail    prompt of k*CHUNK + r tokens: the last chunked-prefill piece is r
  resend  prompt of 64*m + r tokens sent twice: the radix hit is page-aligned,
          so the resend extends only r tokens
  tiny    a 1..3 token prompt

    BASE=http://127.0.0.1:30000 OUT=results/x/chunk_tail.json python3 bench/chunk_tail.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import base, chat  # noqa: E402

OUT = os.environ.get("OUT", "")
CHUNK = int(os.environ.get("CHUNK", "4096"))
PAGE = int(os.environ.get("PAGE", "64"))
TAIL_KS = [int(x) for x in os.environ.get("TAIL_KS", "1,2").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "2"))
# Ordinary text-token ids, well clear of the special/added tokens at the top.
LOW, HIGH = 1000, 150000


def generate(ids, max_new_tokens=8, timeout=900):
    body = {"input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0}}
    req = urllib.request.Request(base() + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    meta = data.get("meta_info") or {}
    return {"seconds": round(time.perf_counter() - t0, 3),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "output_ids": data.get("output_ids") or [],
            "text": data.get("text", "")}


def health():
    try:
        with urllib.request.urlopen(base() + "/health", timeout=10) as r:
            return r.status == 200
    except OSError:
        return False


def main():
    rng = random.Random(f"{time.time_ns()}-{os.getpid()}")
    ids = lambda n: [rng.randrange(LOW, HIGH) for _ in range(n)]  # noqa: E731
    cases = []
    for rnd in range(ROUNDS):
        for r in (1, 2, 3):
            for k in TAIL_KS:
                cases.append(("tail", f"k{k}_r{r}", [ids(k * CHUNK + r)]))
            m = rng.randrange(8, 200)
            prompt = ids(PAGE * m + r)
            cases.append(("resend", f"m{m}_r{r}", [prompt, prompt]))
            cases.append(("tiny", f"n{r}", [ids(r)]))
    results, failed = [], 0
    for kind, label, sends in cases:
        row = {"kind": kind, "label": label, "sends": []}
        try:
            for prompt in sends:
                out = generate(prompt)
                out["ok"] = (out["completion_tokens"] or 0) > 0
                row["sends"].append({k: v for k, v in out.items() if k != "output_ids"})
                row.setdefault("outputs", []).append(out["output_ids"])
            row["ok"] = all(s["ok"] for s in row["sends"]) and health()
            if kind == "resend":
                second = row["sends"][1]
                row["extend_tokens"] = (second["prompt_tokens"] or 0) - (second["cached_tokens"] or 0)
                row["same_greedy_output"] = row["outputs"][0] == row["outputs"][1]
        except Exception as exc:  # noqa: BLE001
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row.pop("outputs", None)
        failed += not row["ok"]
        print(f"RESULT {kind} {label}: {'PASS' if row['ok'] else 'FAIL'} "
              f"{row.get('extend_tokens', '')} {row.get('error', '')}", flush=True)
        results.append(row)
        if not health():
            print("server unhealthy; stopping", flush=True)
            break
    # The server must still answer a normal chat request correctly afterwards.
    try:
        r = chat([{"role": "user", "content": "12*17"}], max_tokens=256, temperature=0,
                 thinking=False)
        after_ok = "204" in r["content"]
    except Exception as exc:  # noqa: BLE001
        after_ok = False
        print("after-check error", exc, flush=True)
    resend = [r for r in results if r["kind"] == "resend" and r["ok"]]
    summary = {"cases": len(cases), "run": len(results), "failed": failed,
               "after_chat_ok": after_ok,
               "resend_short_extend": sum(1 for r in resend if 0 < r["extend_tokens"] < 4),
               "resend_same_greedy": sum(1 for r in resend if r["same_greedy_output"]),
               "resend_ok": len(resend)}
    report = {"summary": summary, "results": results,
              "failed": failed + (len(cases) - len(results)) + (not after_ok)}
    print("SUMMARY", json.dumps(summary), flush=True)
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
