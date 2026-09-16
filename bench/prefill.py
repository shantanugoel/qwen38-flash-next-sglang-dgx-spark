#!/usr/bin/env python3
"""Cold vs prefix-warm prefill with streamed TTFT.

    SIZES=8k,32k N=3 python3 bench/prefill.py
    PREFILL_TEXT=real SIZES=32k,128k,250k N=3 python3 bench/prefill.py

Each repeat uses a unique salt so the radix prefix cache misses. With the
default filler text the n-grams are shared, so later repeats are the PLE-warm /
prefix-cold case. PREFILL_TEXT=real builds every prompt from LongBench-v2
documents not used earlier in the run, so every repeat is PLE-cold as well
(real long-document traffic). The last prompt of each size is then resent as
prefix-warm. Decode-sized gathers are not the target; see bench/decode.py.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat_stream  # noqa: E402

OUT = os.environ.get("OUT", "")
SIZES = os.environ.get("SIZES", "8k,32k")
N = int(os.environ.get("N", os.environ.get("PREFILL_N", "3")))
NEEDLE = "NIGHTINGALE-ACCESS-CODE=7K-QUARTZ-19"
FILLER = (
    "The quarterly operations log continues with routine warehouse notes, "
    "shift handovers, pallet counts, and weather asides. Nothing in this "
    "paragraph is the secret. "
)
MAPPING = {"8k": 8000, "32k": 32000, "40k": 40000, "120k": 120000,
           "128k": 128000, "190k": 190000, "210k": 210000, "250k": 250000,
           "300k": 300000, "400k": 400000, "500k": 500000, "600k": 600000}
TEXT = os.environ.get("PREFILL_TEXT", "filler").lower()
LONGBENCH_JSON = os.environ.get("LONGBENCH_JSON", os.path.expanduser(
    "~/ai/cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots/"
    "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"))
QUESTION = "\n\nQuestion: What is the NIGHTINGALE access code? Reply with the code only."
_DOCS: list[str] = []


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def haystack(target_tokens: int, salt: str) -> str:
    body = [f"Session marker {salt}. "]
    n = approx_tokens(body[0])
    while n < target_tokens // 2:
        body.append(FILLER)
        n = approx_tokens("".join(body))
    body.append(
        f" IGNORE previous noise. The invented fact is: {NEEDLE}. "
        "Do not mention it until asked. "
    )
    while n < target_tokens - 80:
        body.append(FILLER)
        n = approx_tokens("".join(body))
    return "".join(body)


def next_docs(chars: int) -> str:
    """Consume whole LongBench-v2 documents, never reusing one within a run."""
    if not _DOCS:
        import json as _json
        import random
        docs = [d["context"] for d in _json.load(open(LONGBENCH_JSON))
                if 20000 <= len(d["context"]) <= 1500000]
        random.Random(os.environ.get("PREFILL_SEED", str(os.getpid()))).shuffle(docs)
        _DOCS.extend(docs)
    out, n = [], 0
    while n < chars:
        if not _DOCS:
            raise RuntimeError("ran out of unused LongBench-v2 documents")
        doc = _DOCS.pop()
        out.append(doc[: chars - n])
        n += len(out[-1])
    return "\n\n".join(out)


def count_tokens(content: str) -> int:
    import urllib.request
    from client import base, model
    body = {"model": model(), "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base() + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return int(data.get("count") or len(data.get("tokens") or []))


def real_prompt(target_tokens: int, salt: str) -> str:
    """Fresh real documents around a mid-prompt needle, sized by the tokenizer.

    The documents are drawn once; only the cut point moves while sizing, so the
    token count converges instead of changing with every draw."""
    budget = target_tokens - 120
    first, second = next_docs(budget * 3), next_docs(budget * 3)

    def build(chars: int) -> str:
        half = chars // 2
        return (f"Session marker {salt}.\n\n" + first[:half]
                + f"\n\nThe invented fact is: {NEEDLE}. Do not mention it until asked.\n\n"
                + second[:chars - half] + QUESTION)

    lo, hi = 1000, min(len(first), len(second)) * 2
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2
        content = build(mid)
        tokens = count_tokens(content)
        if tokens <= budget:
            best = content
            if tokens >= 0.98 * budget:
                break
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError(f"could not size a real-text prompt under {budget} tokens")
    return best


def scheduler_io() -> dict | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pid = None
    for line in raw.splitlines():
        if "scheduler" in line.lower():
            pid = int(line.split(",", 1)[0].strip())
            break
    if pid is None:
        return None
    try:
        text = Path(f"/proc/{pid}/io").read_text()
    except OSError:
        return None
    out = {"pid": pid}
    for line in text.splitlines():
        key, _, val = line.partition(":")
        if key in ("read_bytes", "write_bytes", "rchar", "wchar"):
            out[key] = int(val.strip())
    return out


def summarize(r: dict, needle_ok: bool) -> dict:
    return {
        "ttft_s": round(r["ttft_s"], 3),
        "seconds": round(r["seconds"], 3),
        "prompt_tokens": r["prompt_tokens"],
        "cached_tokens": r["cached_tokens"],
        "cache_hit_pct": r["cache_hit_pct"],
        "prefill_tps": r["prefill_tps"],
        "decode_tps": r["decode_tps"],
        "completion_tokens": r["completion_tokens"],
        "needle_pass": needle_ok,
        "content": (r["content"] or "")[:80],
    }


def ask(msgs, max_tokens: int) -> tuple[dict, bool]:
    r = chat_stream(
        msgs, max_tokens=max_tokens, temperature=0, thinking=False, timeout=1800,
    )
    text = (r["content"] or "") + " " + (r["reasoning"] or "")
    hit = "7K-QUARTZ-19" in text.replace(" ", "")
    return r, hit


def run_size(label: str, tokens: int) -> dict:
    rows = []
    last_msgs = None
    io0 = scheduler_io()
    for i in range(N):
        salt = f"{label}-{i}-{os.getpid()}-{i * 10007}"
        if TEXT == "real":
            msgs = [{"role": "user", "content": real_prompt(tokens, salt)}]
            kind = "cold_prefix" if i == 0 else "real_fresh_docs"
        else:
            hs = haystack(tokens, salt)
            msgs = [{"role": "user", "content": hs + QUESTION}]
            kind = "cold_prefix" if i == 0 else "ple_warm_prefix_cold"
        last_msgs = msgs
        print(f">> prefill {label} {kind} {i + 1}/{N}", flush=True)
        r, hit = ask(msgs, 32)
        row = {"size": label, "i": i, "kind": kind, **summarize(r, hit)}
        rows.append(row)
        print(
            f"RESULT prefill_{label}_{kind}: ttft={row['ttft_s']:.2f}s "
            f"prefill={row['prefill_tps']} tok/s prompt={row['prompt_tokens']} "
            f"cached={row['cached_tokens']} needle={'PASS' if hit else 'FAIL'}",
            flush=True,
        )
    print(f">> prefill {label} prefix_warm resend", flush=True)
    r, hit = ask(last_msgs, 16)
    warm = {"size": label, "i": N, "kind": "prefix_warm", **summarize(r, hit)}
    rows.append(warm)
    print(
        f"RESULT prefill_{label}_prefix_warm: ttft={warm['ttft_s']:.2f}s "
        f"prefill={warm['prefill_tps']} tok/s prompt={warm['prompt_tokens']} "
        f"cached={warm['cached_tokens']} needle={'PASS' if hit else 'FAIL'}",
        flush=True,
    )
    io1 = scheduler_io()
    io_delta = None
    if io0 and io1:
        io_delta = {k: io1[k] - io0[k] for k in io0 if k != "pid" and k in io1}
    cold = [x["ttft_s"] for x in rows if x["kind"] == "cold_prefix"]
    later = [x["ttft_s"] for x in rows if x["kind"] == "ple_warm_prefix_cold"]
    fresh = [x for x in rows if x["kind"] in ("cold_prefix", "real_fresh_docs")]
    return {
        "size": label,
        "target_tokens": tokens,
        "io_delta": io_delta,
        "median_cold_ttft_s": round(statistics.median(cold), 3) if cold else None,
        "median_ple_warm_ttft_s": round(statistics.median(later), 3) if later else None,
        "text": TEXT,
        "median_real_ttft_s": (round(statistics.median(x["ttft_s"] for x in fresh), 3)
                               if TEXT == "real" else None),
        "median_real_prefill_tps": (round(statistics.median(x["prefill_tps"] for x in fresh), 1)
                                    if TEXT == "real" else None),
        "prefix_warm_ttft_s": warm["ttft_s"],
        "rows": rows,
        "failed": sum(1 for x in rows if not x["needle_pass"]),
    }


def main() -> int:
    sizes = []
    for part in SIZES.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in MAPPING:
            print(f"unknown size {part}", file=sys.stderr)
            return 1
        sizes.append((part, MAPPING[part]))
    groups = [run_size(label, tokens) for label, tokens in sizes]
    rows = [r for g in groups for r in g["rows"]]
    failed = sum(g["failed"] for g in groups)
    report = {"results": rows, "groups": groups, "failed": failed}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT prefill: {len(rows) - failed}/{len(rows)} needles pass", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
