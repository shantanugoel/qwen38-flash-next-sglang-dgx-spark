#!/usr/bin/env python3
"""D2: build a sibling snapshot with FP8 blockwise dense side layers.

The source snapshot is left untouched: every file is symlinked into the new
directory, and only the shards that actually hold converted tensors are
rewritten. The result is served with `--quantization modelopt_mixed`, whose
`hf_quant_config.json` lists the routed experts as NVFP4 (unchanged bytes) and
the converted projections as FP8_BLOCK_SCALES (128x128 `weight_scale_inv`).

What is converted, from the SM121 micro-benchmark
(`scripts/bench_block_fp8_linear.py`): the GDN projections and attention
projections, which are 1.5-3.2x faster in block FP8 at decode batch sizes.
What is deliberately left in BF16:

  hyper_connection.*  0.15x at decode (N=320), and K=320 cannot be block-quantized
  lm_head             the NEXTN draft slices arbitrary rows of it for the 64k
                      token map (U6); 128-row block scales cannot survive that
  embed_tokens, mtp.*, visual.*, mlp.gate, shared_expert*  small or excluded

    MODE=config-only python3 scripts/build_fp8_hybrid_snapshot.py <src> <dst>
    python3 scripts/build_fp8_hybrid_snapshot.py <src> <dst>
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

# Every shard of a packed module must be converted together: SGLang fuses
# q/k/v into `qkv_proj` and in_proj_qkv/in_proj_z into `in_proj_qkvz`, and
# ModelOptMixedPrecisionConfig._resolve_quant_algo applies ONE algo to the whole
# fused layer. Converting only some shards leaves BF16 bytes being read as FP8.
CONVERT = (
    re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$"),
    re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"),
)
# The NEXTN draft is excluded from the quant config (and slices lm_head rows for
# the 64k token map), so its tensors must stay BF16.
SKIP = re.compile(r"(^|\.)mtp\.|\.indexer\.|visual\.")
BLOCK = 128
MODE = os.environ.get("MODE", "convert")


def targets(weight_map: dict) -> list[str]:
    if MODE == "config-only":
        return []
    return [k for k in weight_map
            if any(p.search(k) for p in CONVERT) and not SKIP.search(k)]


def module_of(key: str) -> str:
    return key.rsplit(".weight", 1)[0]


def expert_modules(weight_map: dict) -> list[str]:
    # Language-model layers only: `mtp.layers.0.mlp.experts` belongs to the
    # NEXTN draft, whose experts are BF16 in this checkpoint. Marking them
    # NVFP4 makes the draft build packed params and fail the shape check.
    mods = {m.group(0) for k in weight_map
            if not SKIP.search(k)
            and (m := re.match(r"^.*\.layers\.\d+\.mlp\.experts", k))}
    return sorted(mods)


def ple_modules(weight_map: dict) -> list[str]:
    mods = {m.group(0) for k in weight_map
            if (m := re.match(r"^.*\.ple\.ple_embedding\.ngram_embedding", k))}
    return sorted(mods)


def quant_config(weight_map: dict, converted: list[str]) -> dict:
    layers = {}
    for mod in expert_modules(weight_map):
        layers[mod] = {"quant_algo": "NVFP4", "group_size": 16}
    for mod in ple_modules(weight_map):
        layers[mod] = {"quant_algo": "FP8"}
    for key in converted:
        layers[module_of(key)] = {"quant_algo": "FP8_BLOCK_SCALES"}
    quantized = set(layers)
    exclude = sorted({module_of(k) for k in weight_map
                      if not any(module_of(k).startswith(q) or q.startswith(module_of(k))
                                 for q in quantized)})
    return {"producer": {"name": "qwen38-flash-next-recipe", "version": "d2"},
            "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                             "group_size": 16, "exclude_modules": exclude,
                             "quantized_layers": layers}}


def convert_shard(src_path: Path, dst_path: Path, keys: set[str]) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors, metadata = {}, {}
    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            t = f.get_tensor(key)
            if key not in keys:
                tensors[key] = t
                continue
            w = t.to(torch.float32)
            n, k = w.shape
            pad_n, pad_k = -(-n // BLOCK) * BLOCK, -(-k // BLOCK) * BLOCK
            padded = torch.zeros(pad_n, pad_k, dtype=torch.float32)
            padded[:n, :k] = w
            view = padded.view(pad_n // BLOCK, BLOCK, pad_k // BLOCK, BLOCK)
            amax = view.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
            scale = (amax / 448.0).float()
            q = (view / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            tensors[key] = q.view(pad_n, pad_k)[:n, :k].contiguous()
            # SGLang/DeepSeek convention: weight_scale_inv multiplies back.
            tensors[key.replace(".weight", ".weight_scale_inv")] = (
                scale.view(pad_n // BLOCK, pad_k // BLOCK).contiguous())
            print(f"   converted {key} {tuple(w.shape)}", flush=True)
    save_file(tensors, str(dst_path), metadata=metadata)


def main(src: str, dst: str) -> int:
    src_dir, dst_dir = Path(src).resolve(), Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    index_name = "model.safetensors.index.json"
    weight_map = json.loads((src_dir / index_name).read_text())["weight_map"]
    converted = targets(weight_map)
    shards = {weight_map[k] for k in converted}
    print(f"converting {len(converted)} tensors in {len(shards)} shard(s)", flush=True)

    for entry in sorted(src_dir.iterdir()):
        if entry.name in (index_name, "hf_quant_config.json", "config.json") or entry.name in shards:
            continue
        link = dst_dir / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve())

    for shard in sorted(shards):
        keys = {k for k in converted if weight_map[k] == shard}
        print(f"rewriting {shard} ({len(keys)} tensors)", flush=True)
        convert_shard(src_dir / shard, dst_dir / shard, keys)

    new_map = dict(weight_map)
    for key in converted:
        new_map[key.replace(".weight", ".weight_scale_inv")] = weight_map[key]
    index = json.loads((src_dir / index_name).read_text())
    index["weight_map"] = new_map
    (dst_dir / index_name).write_text(json.dumps(index, indent=2))
    (dst_dir / "hf_quant_config.json").write_text(
        json.dumps(quant_config(weight_map, converted), indent=2))
    # `hf_quant_config.json` is only consulted when config.json carries no
    # `quantization_config` (ModelConfig._parse_quant_hf_config), so the copy
    # here drops the source's NVFP4 block. Without this the mixed config is
    # silently ignored and FP8 bytes land in BF16 parameters.
    config = json.loads((src_dir / "config.json").read_text())
    removed = config.pop("quantization_config", None)
    for section in ("text_config", "vision_config"):
        if isinstance(config.get(section), dict):
            removed = config[section].pop("quantization_config", removed) or removed
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2))
    print("config.json rewritten; quantization_config removed:", bool(removed), flush=True)
    print("wrote", dst_dir, flush=True)
    print("converted modules:", len({module_of(k) for k in converted}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
