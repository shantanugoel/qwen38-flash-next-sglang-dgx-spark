#!/usr/bin/env python3
"""B2: `_ple_shard_matches_slice` against the real table and checkpoint.

CPU only, inside the pinned image, importing the patched `qwen4_exp.py`. For a
few real shards it checks that the slice verifier agrees with the whole-tensor
`_ple_shard_matches`, that single-byte corruptions inside sampled windows are
caught by both, and how long each takes.

    ./scripts/test_ple_slice_reuse.sh
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import torch
from safetensors import safe_open

SNAPSHOT, TABLE, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
SHARDS = [int(x) for x in os.environ.get("SHARDS", "0,63,127").split(",")]
ROWS_TOTAL = 320001536
NUM_SHARDS = 128


def load_model_module(path):
    spec = importlib.util.spec_from_file_location("patched_qwen4_exp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_model_module(MODEL)
    weight_map = json.load(open(os.path.join(SNAPSHOT, "model.safetensors.index.json")))["weight_map"]
    nbytes = os.path.getsize(TABLE)
    row_bytes = nbytes // ROWS_TOTAL
    table = torch.from_file(TABLE, shared=False, size=nbytes, dtype=torch.uint8)
    shard_size = (ROWS_TOTAL + NUM_SHARDS - 1) // NUM_SHARDS
    failures = 0

    def check(label, got, want):
        nonlocal failures
        ok = got == want
        failures += not ok
        print(f"  {label}: {got} (want {want}) {'PASS' if ok else 'FAIL'}", flush=True)

    for shard in SHARDS:
        key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        path = os.path.join(SNAPSHOT, weight_map[key])
        with safe_open(path, framework="pt", device="cpu") as st:
            source = st.get_slice(key)
            rows = source.get_shape()[0]
            start = shard * shard_size
            dst = table[start * row_bytes:(start + rows) * row_bytes].view(
                torch.float8_e4m3fn).view(rows, row_bytes)
            print(f"shard {shard}: rows={rows} file={weight_map[key]}", flush=True)
            t0 = time.perf_counter()
            slice_ok = mod._ple_shard_matches_slice(dst, source)
            t_slice = time.perf_counter() - t0
            check(f"slice verifier on disk table ({t_slice * 1e3:.1f} ms)", slice_ok, True)
            t0 = time.perf_counter()
            full = st.get_tensor(key)
            whole_ok = mod._ple_shard_matches(dst, full)
            t_whole = time.perf_counter() - t0
            check(f"whole-tensor verifier ({t_whole * 1e3:.1f} ms incl. get_tensor)", whole_ok, True)

            copy = dst.clone()
            flat = copy.view(-1).view(torch.uint8)
            n = flat.numel()
            # Offset 0 and the last byte are always inside a sampled window.
            for label, offset in (("first byte", 0), ("last byte", n - 1)):
                flat[offset] ^= 0xFF
                check(f"corrupt {label}: slice", mod._ple_shard_matches_slice(copy, source), False)
                check(f"corrupt {label}: whole", mod._ple_shard_matches(copy, full), False)
                flat[offset] ^= 0xFF
            check("restored copy: slice", mod._ple_shard_matches_slice(copy, source), True)
            # A different shape must never count as a match.
            check("short dst: slice", mod._ple_shard_matches_slice(copy[:-1], source), False)
            del full, copy
    os.environ["SGLANG_QWEN4_PLE_REUSE"] = "0"
    with safe_open(path, framework="pt", device="cpu") as st:
        check("SGLANG_QWEN4_PLE_REUSE=0 forces copy", mod._ple_shard_matches_slice(dst, st.get_slice(key)), False)
    print("RESULT", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
