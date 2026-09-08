#!/usr/bin/env python3
"""Cold vs prefix-warm prefill with streamed TTFT.

    SIZES=8k,32k N=3 python3 bench/prefill.py

Each repeat uses a unique salt so the radix prefix cache misses. Filler n-grams
are shared, so later repeats are the PLE-warm / prefix-cold case. The last prompt
of each size is then resent as prefix-warm. Decode-sized gathers are not the
target; see bench/decode.py for that.
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
           "128k": 128000, "190k": 190000, "210k": 210000}


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
        hs = haystack(tokens, salt)
        msgs = [{
            "role": "user",
            "content": (
                f"{hs}\n\nQuestion: What is the NIGHTINGALE access code? "
                "Reply with the code only."
            ),
        }]
        last_msgs = msgs
        kind = "cold_prefix" if i == 0 else "ple_warm_prefix_cold"
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
    return {
        "size": label,
        "target_tokens": tokens,
        "io_delta": io_delta,
        "median_cold_ttft_s": round(statistics.median(cold), 3) if cold else None,
        "median_ple_warm_ttft_s": round(statistics.median(later), 3) if later else None,
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
