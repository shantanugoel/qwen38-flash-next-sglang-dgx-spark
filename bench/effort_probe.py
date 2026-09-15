#!/usr/bin/env python3
"""Repeat one deterministic-sampling case to measure boot-to-boot instability.

    N=20 python3 bench/effort_probe.py
    CACHE_MODES=miss,hit N=10 python3 bench/effort_probe.py

CACHE_MODES (C4) switches to exact token ids: the chat prompt is rendered with
/tokenize and sent to /generate greedy, and the full output id sequences are
compared. `miss` calls /flush_cache before every repeat, so each answer is
computed from an empty radix cache; `hit` never flushes, so repeats reuse the
cached prefix (and its mamba checkpoints). Divergence within `miss` points at
kernels; divergence only within `hit`, or between the two, points at cached
state (sglang#34820-style precision).

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
CACHE_MODES = [m.strip() for m in os.environ.get("CACHE_MODES", "").split(",") if m.strip()]
# The radix cache stores whole 64-token pages, so a ~40 token prompt can never
# be a cache hit. Prepend a fixed document so `hit` really reuses cached pages.
CACHE_PREFIX_TOKENS = int(os.environ.get("CACHE_PREFIX_TOKENS", "4096"))
LONGBENCH_JSON = os.environ.get("LONGBENCH_JSON", os.path.expanduser(
    "~/ai/cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots/"
    "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"))


def shared_prefix() -> str:
    if CACHE_PREFIX_TOKENS <= 0:
        return ""
    docs = json.load(open(LONGBENCH_JSON))
    doc = next(d["context"] for d in docs if len(d["context"]) > CACHE_PREFIX_TOKENS * 6)
    return doc[: CACHE_PREFIX_TOKENS * 3] + "\n\n"
PROMPTS = {
    "effort": QUESTION,
    "code": "Write a Python function that merges two sorted lists. Code only.",
}


def post(path, body=None, timeout=300):
    import urllib.request
    from client import base
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base() + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if body is not None else raw


def flush_cache():
    import time
    import urllib.error
    for _ in range(30):
        try:
            post("/flush_cache")
            return True
        except urllib.error.HTTPError:
            time.sleep(2)
    return False


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def token_probe() -> int:
    from client import model
    report = {"modes": {}, "summary": {}}
    prefix = shared_prefix()
    for name, prompt in PROMPTS.items():
        tok = post("/tokenize", {"model": model(), "messages": [{"role": "user", "content": prefix + prompt}],
                                 "chat_template_kwargs": {"enable_thinking": THINKING}})
        ids = tok.get("tokens") or tok.get("input_ids")
        max_new = 64 if name == "effort" else 256
        seqs_by_mode = {}
        for mode in CACHE_MODES:
            seqs = []
            for i in range(N):
                if mode == "miss" and not flush_cache():
                    raise RuntimeError("flush_cache kept failing")
                out = post("/generate", {"input_ids": ids, "sampling_params": {
                    "max_new_tokens": max_new, "temperature": 0}})
                meta = out.get("meta_info") or {}
                seqs.append(out.get("output_ids") or [])
                print(f"RESULT token_probe {name} {mode} {i}: len={len(seqs[-1])} "
                      f"cached={meta.get('cached_tokens')} text={out.get('text', '')[:30]!r}", flush=True)
            seqs_by_mode[mode] = seqs
            distinct = collections.Counter(tuple(x) for x in seqs)
            ref = seqs[0]
            report["modes"][f"{name}/{mode}"] = {
                "n": N, "prompt_tokens": len(ids),
                "cached_tokens_last": meta.get("cached_tokens"), "distinct_sequences": len(distinct),
                "largest_class": distinct.most_common(1)[0][1],
                "first_divergence_vs_first": [first_divergence(ref, x) for x in seqs],
            }
        if "miss" in seqs_by_mode and "hit" in seqs_by_mode:
            miss = collections.Counter(tuple(x) for x in seqs_by_mode["miss"])
            hit = collections.Counter(tuple(x) for x in seqs_by_mode["hit"])
            report["summary"][name] = {
                "miss_distinct": len(miss), "hit_distinct": len(hit),
                "modal_equal": miss.most_common(1)[0][0] == hit.most_common(1)[0][0],
                "first_divergence_modal": first_divergence(list(miss.most_common(1)[0][0]),
                                                           list(hit.most_common(1)[0][0])),
            }
    report["results"] = [dict(key=k, **v) for k, v in report["modes"].items()]
    report["summary"]["n"] = len(report["results"])
    report["summary"]["repeats"] = N
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
    print("RESULT token_probe:", json.dumps(report["summary"]), flush=True)
    return 0


def main() -> int:
    if CACHE_MODES:
        return token_probe()
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
