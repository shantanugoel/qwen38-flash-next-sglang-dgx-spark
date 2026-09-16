#!/usr/bin/env python3
"""B1: the NEXTN draft pass reads only the checkpoint files holding MTP keys.

The draft model (`Qwen4ExpForCausalLMMTP`) is loaded through the same
`DefaultModelLoader` as the target, which resolves every `*.safetensors` file
of the repo and materialises every tensor of all 206 files (125.9 GiB) with
`get_tensor`, only for `Qwen3_5ForCausalLMMTP.load_weights` to drop every name
that does not contain "mtp" as its first check. Measured: 88.7 s per boot.

Two hunks:
  loader.py          `Source` gains an optional `key_filter`, taken from the
                     model's `checkpoint_key_filter`; `_get_weights_iterator`
                     then keeps only files the safetensors index maps a
                     passing key to (plus any file the index does not list).
                     No index, or no passing key: the file list is unchanged.
  qwen4_exp_mtp.py   `checkpoint_key_filter` mirrors the load_weights guard.

The file set is derived from the index at runtime, never hard-coded.
SGLANG_CHECKPOINT_KEY_FILTER=0 restores stock file resolution.

Usage: python3 draft_mtp_files.py <loader.py> <qwen4_exp_mtp.py>
"""
from __future__ import annotations

import sys

SOURCE_FIELD = """        key_filter: Optional[Callable[[str], bool]] = None
        \"\"\"Checkpoint key predicate; files holding no passing key are skipped.\"\"\"

"""

INIT_FIELD = """                key_filter=getattr(model, "checkpoint_key_filter", None),
"""


def patch_default_loader_source(src: str) -> str:
    """Add `key_filter` to DefaultModelLoader.Source and its init_new.

    Anchored on the class region rather than on the exact field list, which
    upstream extends (e.g. allow_patterns_overrides).
    """
    begin = src.index("class DefaultModelLoader(BaseModelLoader):")
    end = src.index("\nclass ", begin + 10)
    region = src[begin:end]
    field_anchor = '        model_config: Optional[ModelConfig] = None\n'
    doc_anchor = '        """The model configuration (for checking architecture, etc)."""\n'
    i = region.index(field_anchor)
    j = region.index(doc_anchor, i) + len(doc_anchor)
    region = region[:j] + "\n" + SOURCE_FIELD.rstrip("\n") + "\n" + region[j:]
    init_anchor = "                model_config=model_config,\n"
    k = region.index(init_anchor) + len(init_anchor)
    region = region[:k] + INIT_FIELD + region[k:]
    return src[:begin] + region + src[end:]


ITER_OLD = '''        else:
            hf_folder = resolved_source.hf_folder
            hf_weights_files = list(resolved_source.weight_files)
            use_safetensors = resolved_source.use_safetensors

'''

ITER_NEW = '''        else:
            hf_folder = resolved_source.hf_folder
            hf_weights_files = list(resolved_source.weight_files)
            use_safetensors = resolved_source.use_safetensors

        if use_safetensors and source.key_filter is not None:
            hf_weights_files = _files_with_matching_keys(
                hf_weights_files, hf_folder, source.key_filter
            )

'''

TYPING_OLD = """from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
"""

TYPING_NEW = """from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
"""

HELPER_ANCHOR = "class DefaultModelLoader(BaseModelLoader):"

HELPER = '''def _files_with_matching_keys(
    hf_weights_files: List[str], hf_folder: str, key_filter: Callable[[str], bool]
) -> List[str]:
    """Keep the files the safetensors index maps a key passing key_filter to.

    Files the index does not list are kept (they cannot be ruled out). Without
    an index, when no key passes, or with SGLANG_CHECKPOINT_KEY_FILTER=0, the
    list is returned unchanged.
    """
    if os.environ.get("SGLANG_CHECKPOINT_KEY_FILTER", "1").strip() == "0":
        return hf_weights_files
    index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return hf_weights_files
    matching = {os.path.basename(v) for k, v in weight_map.items() if key_filter(k)}
    if not matching:
        return hf_weights_files
    indexed = {os.path.basename(v) for v in weight_map.values()}
    kept = [
        f
        for f in hf_weights_files
        if os.path.basename(f) in matching or os.path.basename(f) not in indexed
    ]
    logger.info(
        "Checkpoint key filter: reading %d of %d weight files (%d matching keys)",
        len(kept),
        len(hf_weights_files),
        sum(1 for k in weight_map if key_filter(k)),
    )
    return kept


'''

MTP_OLD = '''class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
'''

MTP_NEW = '''class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    @staticmethod
    def checkpoint_key_filter(name: str) -> bool:
        # Mirrors Qwen3_5ForCausalLMMTP.load_weights, which skips every name
        # without "mtp": the loader then reads only the files holding MTP keys.
        return "mtp" in name

'''


def replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"ERROR: expected 1 {what}, found {src.count(old)}")
    return src.replace(old, new, 1)


def main(loader_path: str, mtp_path: str) -> int:
    with open(loader_path, encoding="utf-8") as f:
        loader = f.read()
    if "_files_with_matching_keys" in loader:
        print("ALREADY PATCHED:", loader_path)
    else:
        loader = patch_default_loader_source(loader)
        loader = replace_once(loader, ITER_OLD, ITER_NEW, "resolved-source branch")
        loader = replace_once(loader, HELPER_ANCHOR, HELPER + HELPER_ANCHOR,
                              "DefaultModelLoader class")
        loader = replace_once(loader, TYPING_OLD, TYPING_NEW, "typing import block")
        with open(loader_path, "w", encoding="utf-8") as f:
            f.write(loader)
        print("PATCHED:", loader_path)

    with open(mtp_path, encoding="utf-8") as f:
        mtp = f.read()
    if "checkpoint_key_filter" in mtp:
        print("ALREADY PATCHED:", mtp_path)
    else:
        mtp = replace_once(mtp, MTP_OLD, MTP_NEW, "Qwen4ExpForCausalLMMTP class")
        with open(mtp_path, "w", encoding="utf-8") as f:
            f.write(mtp)
        print("PATCHED:", mtp_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
