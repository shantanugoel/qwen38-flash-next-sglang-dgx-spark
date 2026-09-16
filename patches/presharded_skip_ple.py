#!/usr/bin/env python3
"""Item 12: keep the PLE table out of the presharded weight cache.

REJECTED 2026-09-16, not applied by prepare.sh. The cache works (the model
serves normally from it) but is SLOWER here: two cached boots loaded the target
in 463.8 s and 483.1 s against 385-410 s for the normal path, i.e. boot 595-608 s
vs 500-530 s. On unified memory, re-reading 74 GB of cached tensors costs more
than re-running the per-tensor weight loaders. See RESEARCH_LOG item 12.

`--load-format presharded` (PreshardedModelLoader) dumps `model.state_dict()`
after post-processing and reloads it on later boots, which is exactly the
post-load cache the B3 profile pointed at: the main load is per-tensor
`weight_loader` work, not disk.

The PLE n-gram table must not go in it. It is a 47.7 GiB host mmap of a file
that already survives restarts, so dumping it would write the table a second
time and reloading it would materialise it as ordinary memory, undoing
`--ple-offload-backend file`. The table file *is* its own cache: by the time
weights load, `allocate_ple_host_table` has already mapped the correct rows
(our `ple_reuse.py` verifies them per shard).

Two hunks:
  dump  drop PLE table keys from the state dict before it is hashed/written
  load  drop the same keys from the missing-parameter check, since nothing in
        the cache fills them and the mapping already has the right bytes
  dispatch  let LoadFormat.PRESHARDED win over the ModelOpt loader, which
        otherwise claims every modelopt checkpoint and rejects presharded_path

Usage: python3 presharded_skip_ple.py <path to model_loader/loader.py>
"""
from __future__ import annotations

import sys

# get_model_loader returns ModelOptModelLoader for a modelopt checkpoint before
# it ever tests LoadFormat.PRESHARDED, so the weight cache is unreachable here
# (and `presharded_path` then trips DefaultModelLoader's extra-config check).
# ModelOpt already delegates to DefaultModelLoader for an already-quantized
# checkpoint, and PreshardedModelLoader builds the model through the same
# `_initialize_model` + quant config, so letting PRESHARDED win is safe.
DISPATCH_OLD = """    modelopt_config = load_config.modelopt_config
"""

DISPATCH_NEW = """    if load_config.load_format == LoadFormat.PRESHARDED:
        return PreshardedModelLoader(load_config)

    modelopt_config = load_config.modelopt_config
"""

# The speculative draft keeps the normal loader, but is handed the same
# --model-loader-extra-config; PreshardedModelLoader is the only consumer of
# these keys, so tolerating them elsewhere keeps the draft loadable.
VALIDATE_OLD = """    allowed_keys = {"enable_multithread_load", "num_threads"}
"""

VALIDATE_NEW = """    allowed_keys = {
        "enable_multithread_load",
        "num_threads",
        "presharded_path",
        "draft_presharded_path",
    }
"""

HELPER_ANCHOR = "class PreshardedModelLoader(DefaultModelLoader):"

HELPER = '''def _is_ple_table_key(name: str) -> bool:
    """The offloaded PLE n-gram table, which lives in its own backing file."""
    return name.endswith(".ngram_embedding.weight")


'''

DUMP_OLD = """            state_dict = dict(model.state_dict())
            extras = self._collect_extra_tensors(model)
            self._dump_state_to_disk(state_dict, extras, presharded_dir, shard_config)
"""

DUMP_NEW = """            state_dict = {
                k: v for k, v in model.state_dict().items()
                if not _is_ple_table_key(k)
            }
            extras = self._collect_extra_tensors(model)
            self._dump_state_to_disk(state_dict, extras, presharded_dir, shard_config)
"""

LOAD_OLD = """            state_dict = dict(model.state_dict())
            reads = plan.get("rank_to_reads", {}).get(str(rank), [])
"""

LOAD_NEW = """            # The PLE table is not in the cache; its backing file already holds
            # the verified rows, so it must not count as a missing parameter.
            state_dict = {
                k: v for k, v in model.state_dict().items()
                if not _is_ple_table_key(k)
            }
            reads = plan.get("rank_to_reads", {}).get(str(rank), [])
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "_is_ple_table_key" in src:
        print("ALREADY PATCHED:", path)
        return 0
    for old, new, what in ((DUMP_OLD, DUMP_NEW, "presharded dump state_dict"),
                           (LOAD_OLD, LOAD_NEW, "presharded load state_dict"),
                           (DISPATCH_OLD, DISPATCH_NEW, "get_model_loader dispatch"),
                           (VALIDATE_OLD, VALIDATE_NEW, "extra config validator")):
        if src.count(old) != 1:
            print(f"ERROR: expected 1 {what}, found {src.count(old)}")
            return 1
        src = src.replace(old, new, 1)
    if src.count(HELPER_ANCHOR) != 1:
        print("ERROR: could not locate PreshardedModelLoader")
        return 1
    src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
