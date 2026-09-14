#!/usr/bin/env bash
# B2: CPU check of the PLE slice verifier on the real table and checkpoint.
# Reads ~1.2 GiB of checkpoint; do not run it during a measured benchmark.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
[[ -f "${QWEN4_BACKEND}" ]] || { echo "run prepare.sh first" >&2; exit 1; }
table="$(ls "${PLE_DIR}"/ple_table_*.bin | head -1)"
UIDGID="$(docker_user)"
docker run --rm --init --user "${UIDGID}" --memory 16g --cpuset-cpus "${CPUSET:-0-19}" \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES= -e SHARDS \
  -v "${HF_CACHE}:/huggingface:ro" -v "${PLE_DIR}:/ple:ro" \
  -v "${QWEN4_BACKEND}:/work/patched_qwen4_exp.py:ro" \
  -v "${SCRIPT_DIR}/test_ple_slice_reuse.py:/work/test_ple_slice_reuse.py:ro" \
  --workdir /work --entrypoint python3 "${IMAGE}" /work/test_ple_slice_reuse.py \
  "/huggingface/hub/models--${MODEL//\//--}/snapshots/${REVISION}" \
  "/ple/$(basename "${table}")" /work/patched_qwen4_exp.py
