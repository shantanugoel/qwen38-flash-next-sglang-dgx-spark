#!/usr/bin/env python3
"""B2: verify the reused PLE table from checkpoint slices, not whole files.

REJECTED 2026-09-14, not applied by prepare.sh. safetensors 0.8.0 `get_tensor`
is already lazy over mmap, so the stock loader only faults in the sampled
windows of the PLE files: they cost ~6 s of the main load, not ~83 s (that tail
is the four dense model-bf16 files). With this patch the main load was 402 s vs
396 s and the boot 515 s vs 504 s. Kept as the record; see RESEARCH_LOG B2.

`ple_reuse.py` already skips rewriting a PLE shard whose on-disk rows match
the checkpoint on 32+2 sampled 4 KiB windows. But the loader still hands
`load_weights` every tensor of the 10 `model-plefp8-*` files (47.7 GiB, 129
tensors), materialised with `get_tensor` on 8 threads, just so the patch can
compare ~140 KiB per shard. Measured: the last 17 shards of the main load
(which include the PLE files) take ~83 s, with up to ~38 GiB of transient RAM.

This patch keeps the same verification and reads only what it compares:

  loader.py     after `checkpoint_key_filter` drops files, the loader tells
                the model which files it skipped (`checkpoint_files_skipped`).
                Requires draft_mtp_files.py (B1) first.
  qwen4_exp.py  `Qwen4ExpForConditionalGeneration.checkpoint_key_filter`
                rejects `.ple_embedding.ngram_embedding.` keys, so files that
                hold only PLE table shards and their scale are skipped. After
                the normal weight loop, every key of a skipped file is loaded
                by `safe_open`: the scale buffer with `get_tensor`, each shard
                by sampling the same windows through `get_slice` against the
                table. A shard whose samples all match counts as already on
                disk; any mismatch (or an unexpected layout) loads that whole
                shard with `get_tensor` and takes the existing copy path.

`SGLANG_QWEN4_PLE_REUSE=0` (force a full copy) also turns the filter off, so
that path still reads every file like stock. So does
`SGLANG_QWEN4_PLE_SLICE_REUSE=0`, which restores the U4a behaviour exactly.
`SGLANG_QWEN4_PLE_SLICE_REUSE_FORCE_FULL=1` (test only) fails every slice check,
so each shard takes the whole-tensor fallback.

Usage: python3 ple_slice_reuse.py <loader.py> <qwen4_exp.py>
"""
from __future__ import annotations

import sys

MODULE_HELPER_ANCHOR = '''def _ple_reuse_report() -> None:
'''

MODULE_HELPER_NEW = '''def _ple_shard_matches_slice(dst: torch.Tensor, source) -> bool:
    \"\"\"_ple_shard_matches for a safetensors slice: compare the same sampled
    byte windows, reading only the rows that cover them from the checkpoint.\"\"\"
    import os

    if os.environ.get("SGLANG_QWEN4_PLE_REUSE", "1").strip() == "0":
        return False
    # Test knob: treat every slice sample as a mismatch (full-read fallback).
    if os.environ.get("SGLANG_QWEN4_PLE_SLICE_REUSE_FORCE_FULL", "0").strip() == "1":
        return False
    shape = tuple(source.get_shape())
    if dst.dtype != torch.float8_e4m3fn or source.get_dtype() != "F8_E4M3":
        return False
    if tuple(dst.shape) != shape or dst.dim() < 1 or shape[0] == 0:
        return False
    n_rows = shape[0]
    row_bytes = dst[0].numel()
    n = n_rows * row_bytes
    win = 4096
    if n <= win * 2:
        spans = [(0, n_rows)]
    else:
        try:
            k = int(os.environ.get("SGLANG_QWEN4_PLE_REUSE_WINDOWS", "32"))
        except ValueError:
            k = 32
        k = max(4, k)
        gen = torch.Generator().manual_seed(0x5150 ^ n)
        offs = (torch.randint(0, (n - win) // win, (k,), generator=gen) * win).tolist()
        offs = sorted(set(offs + [0, n - win]))
        spans = [
            (o // row_bytes, min(n_rows, (o + win + row_bytes - 1) // row_bytes))
            for o in offs
        ]
    try:
        for r0, r1 in spans:
            a = dst[r0:r1].reshape(-1).view(torch.uint8)
            if a.device.type != "cpu":
                a = a.cpu()
            b = source[r0:r1].reshape(-1).contiguous().view(torch.uint8)
            if not torch.equal(a, b):
                return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _ple_reuse_report() -> None:
'''

LOADER_SOURCE_OLD = '''        key_filter: Optional[Callable[[str], bool]] = None
        """Checkpoint key predicate; files holding no passing key are skipped."""
'''

LOADER_SOURCE_NEW = '''        key_filter: Optional[Callable[[str], bool]] = None
        """Checkpoint key predicate; files holding no passing key are skipped."""

        on_files_skipped: Optional[Callable[[str, List[str]], None]] = None
        """Told (folder, skipped files) when key_filter drops files."""
'''

LOADER_INIT_OLD = '''                key_filter=getattr(model, "checkpoint_key_filter", None),
'''

LOADER_INIT_NEW = '''                key_filter=getattr(model, "checkpoint_key_filter", None),
                on_files_skipped=getattr(model, "checkpoint_files_skipped", None),
'''

LOADER_ITER_OLD = '''        if use_safetensors and source.key_filter is not None:
            hf_weights_files = _files_with_matching_keys(
                hf_weights_files, hf_folder, source.key_filter
            )
'''

LOADER_ITER_NEW = '''        if use_safetensors and source.key_filter is not None:
            kept_files = _files_with_matching_keys(
                hf_weights_files, hf_folder, source.key_filter
            )
            if source.on_files_skipped is not None and len(kept_files) < len(
                hf_weights_files
            ):
                kept_set = set(kept_files)
                source.on_files_skipped(
                    hf_folder, [f for f in hf_weights_files if f not in kept_set]
                )
            hf_weights_files = kept_files
'''

MODEL_CLASS_OLD = '''    def _load_qwen4_exp_ple_buffer(
        self,
'''

MODEL_CLASS_NEW = '''    def checkpoint_key_filter(self, name: str) -> bool:
        # B2: PLE table shards (and their scale) are verified from checkpoint
        # slices after the weight loop instead of being read whole.
        import os

        if os.environ.get("SGLANG_QWEN4_PLE_REUSE", "1").strip() == "0":
            return True
        if os.environ.get("SGLANG_QWEN4_PLE_SLICE_REUSE", "1").strip() == "0":
            return True
        return ".ple_embedding.ngram_embedding." not in name

    def checkpoint_files_skipped(self, hf_folder: str, files) -> None:
        self._ple_skipped_files = list(files)

    def _load_qwen4_exp_ple_buffer(
        self,
'''

MODEL_SLICE_ANCHOR = '''        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())
'''

MODEL_SLICE_NEW = '''        def ple_shard_matches_slices(name: str, st, key: str) -> bool:
            """True when the table rows of a skipped shard match sampled rows
            read through get_slice; the same windows _ple_shard_matches uses."""
            import re

            match = re.search(r"\\.ngram_embedding\\.shard_(\\d+)\\.weight$", name)
            if not match:
                return False
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            source = st.get_slice(key)
            shape = source.get_shape()
            if (
                source.get_dtype() != "F8_E4M3"
                or emb.weight.dtype != torch.float8_e4m3fn
                or len(shape) != emb.weight.dim()
                or tuple(shape[1:]) != tuple(emb.weight.shape[1:])
            ):
                return False
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            row_start = int(match.group(1)) * shard_size
            n_rows = int(shape[0])
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            if n_rows == 0 or row_start < tp_start or row_start + n_rows > tp_end:
                return False
            local_start = row_start - tp_start
            dst = emb.weight.data[local_start : local_start + n_rows]
            if not _ple_shard_matches_slice(dst, source):
                return False
            _PLE_REUSE_STATS["skipped"] += 1
            _PLE_REUSE_STATS["rows_skipped"] += n_rows
            _PLE_REUSE_STATS["slice_verified"] = (
                _PLE_REUSE_STATS.get("slice_verified", 0) + 1
            )
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        def load_skipped_ple_files() -> None:
            from safetensors import safe_open

            skipped = getattr(self, "_ple_skipped_files", None) or []
            self._ple_skipped_files = []
            for path in skipped:
                with safe_open(path, framework="pt", device="cpu") as st:
                    for key in st.keys():
                        name = key.replace("model.language_model.", "model.")
                        if ".ngram_embedding.shard_" in name:
                            if ple_shard_matches_slices(name, st, key):
                                continue
                            if load_qwen4_exp_ple_shard(name, st.get_tensor(key)):
                                continue
                        elif self._load_qwen4_exp_ple_buffer(
                            name, st.get_tensor(key), buffers, loaded_buffers
                        ):
                            continue
                        raise ValueError(
                            f"checkpoint key filter skipped an unhandled weight: {key}"
                        )
            if skipped:
                logger.info(
                    "PLE table: read %d skipped checkpoint files by slice "
                    "(%d shards verified from sampled rows)",
                    len(skipped),
                    _PLE_REUSE_STATS.get("slice_verified", 0),
                )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())
'''

MODEL_CALL_OLD = '''        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)
'''

MODEL_CALL_NEW = '''        load_skipped_ple_files()
        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)
'''


def replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"ERROR: expected 1 {what}, found {src.count(old)}")
    return src.replace(old, new, 1)


def main(loader_path: str, model_path: str) -> int:
    with open(loader_path, encoding="utf-8") as f:
        loader = f.read()
    if "on_files_skipped" in loader:
        print("ALREADY PATCHED:", loader_path)
    else:
        if "_files_with_matching_keys" not in loader:
            raise SystemExit("ERROR: apply draft_mtp_files.py (B1) first")
        loader = replace_once(loader, LOADER_SOURCE_OLD, LOADER_SOURCE_NEW, "Source key_filter")
        loader = replace_once(loader, LOADER_INIT_OLD, LOADER_INIT_NEW, "init_new key_filter")
        loader = replace_once(loader, LOADER_ITER_OLD, LOADER_ITER_NEW, "key filter call")
        with open(loader_path, "w", encoding="utf-8") as f:
            f.write(loader)
        print("PATCHED:", loader_path)

    with open(model_path, encoding="utf-8") as f:
        model = f.read()
    if "load_skipped_ple_files" in model:
        print("ALREADY PATCHED:", model_path)
        return 0
    if "_ple_shard_matches" not in model:
        raise SystemExit("ERROR: apply ple_reuse.py first")
    model = replace_once(model, MODULE_HELPER_ANCHOR, MODULE_HELPER_NEW, "_ple_reuse_report")
    model = replace_once(model, MODEL_CLASS_OLD, MODEL_CLASS_NEW, "_load_qwen4_exp_ple_buffer")
    model = replace_once(model, MODEL_SLICE_ANCHOR, MODEL_SLICE_NEW, "load_weights params_dict")
    model = replace_once(model, MODEL_CALL_OLD, MODEL_CALL_NEW, "load_weights tail")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model)
    print("PATCHED:", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
