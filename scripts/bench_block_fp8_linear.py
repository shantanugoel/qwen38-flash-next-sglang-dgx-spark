#!/usr/bin/env python3
"""D2 step 1: block-FP8 vs BF16 linear at this checkpoint's side-layer shapes.

Runs inside the pinned image on the GPU, no model load. For each (N, K) of the
BF16 tensors that D2 would convert, times `F.linear` in BF16 against SGLang's
128x128 block-FP8 linear at decode (M=1..4) and prefill (M=4096) batch sizes,
and checks the relative error. Decode is bandwidth-bound, so the interesting
column is M=1..4.

    ./scripts/bench_block_fp8_linear.sh
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

BACKENDS = {
    "triton": "triton_w8a8_block_fp8_linear",
    "cutlass": "cutlass_w8a8_block_fp8_linear_with_fallback",
    "deepgemm": "deepgemm_w8a8_block_fp8_linear_with_fallback",
}

BLOCK = [128, 128]
SHAPES = [
    # (name, N, K, count in the checkpoint)
    ("linear_attn.in_proj_qkv", 10240, 2560, 36),
    ("linear_attn.in_proj_z", 6144, 2560, 36),
    ("linear_attn.out_proj", 2560, 6144, 36),
    ("self_attn.q_proj", 12288, 2560, 12),
    ("self_attn.o_proj", 2560, 6144, 12),
    ("hyper_connection.down", 320, 10240, 96),
    ("hyper_connection.up", 10240, 320, 96),
    ("lm_head", 248320, 2560, 1),
]
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,2,4,4096").split(",")]
ITERS = int(os.environ.get("ITERS", "50"))


def timed(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    dev = torch.device("cuda")
    which = os.environ.get("FP8_BACKEND", "triton")
    linear = getattr(fp8_utils, BACKENDS[which])
    print("backend:", which, getattr(linear, "__name__", linear), flush=True)
    rows = []
    for name, n, k, count in SHAPES:
        w_bf16 = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        w_fp8, w_scale = per_block_cast_to_fp8(w_bf16.float())
        for m in BATCHES:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.05
            bf16_ms = timed(lambda: F.linear(x, w_bf16))
            try:
                fp8_ms = timed(lambda: linear(x, w_fp8, BLOCK, w_scale, None, None))
                err = (linear(x, w_fp8, BLOCK, w_scale, None, None).float()
                       - F.linear(x, w_bf16).float())
                rel = (err.norm() / F.linear(x, w_bf16).float().norm()).item()
            except Exception as exc:  # noqa: BLE001
                rows.append({"name": name, "n": n, "k": k, "m": m,
                             "error": f"{type(exc).__name__}: {exc}"[:200]})
                print(f"{name:26s} N={n:6d} K={k:5d} M={m:5d} FP8 FAILED {exc}"[:150], flush=True)
                continue
            rows.append({"name": name, "n": n, "k": k, "m": m, "count": count,
                         "bf16_ms": round(bf16_ms, 4), "fp8_ms": round(fp8_ms, 4),
                         "speedup": round(bf16_ms / fp8_ms, 3), "rel_l2": round(rel, 5)})
            print(f"{name:26s} N={n:6d} K={k:5d} M={m:5d} bf16 {bf16_ms:8.4f} ms  "
                  f"fp8 {fp8_ms:8.4f} ms  x{bf16_ms / fp8_ms:5.2f}  rel_l2 {rel:.4f}",
                  flush=True)
        del w_bf16, w_fp8, w_scale
        torch.cuda.empty_cache()
    # Per-token decode cost of one full layer stack, weighted by tensor counts.
    for m in BATCHES:
        got = [r for r in rows if r.get("m") == m and "bf16_ms" in r]
        if not got:
            continue
        bf16 = sum(r["bf16_ms"] * r["count"] for r in got)
        fp8 = sum(r["fp8_ms"] * r["count"] for r in got)
        print(f"TOTAL M={m:5d}: bf16 {bf16:8.2f} ms  fp8 {fp8:8.2f} ms  "
              f"x{bf16 / fp8:5.2f} over all side-layer tensors", flush=True)
        rows.append({"total_for_m": m, "bf16_ms": round(bf16, 3), "fp8_ms": round(fp8, 3),
                     "speedup": round(bf16 / fp8, 3)})
    out = os.environ.get("OUT")
    if out:
        with open(out, "w") as f:
            json.dump({"results": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
