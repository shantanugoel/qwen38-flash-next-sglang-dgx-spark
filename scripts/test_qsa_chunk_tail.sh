#!/usr/bin/env bash
# A1: CPU-only check of the #38346 clamp against the stock indexer. No GPU, no
# server; safe beside a serving experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
[[ -f "${BUILD}/qsa/qsa_indexer.py" ]] || { echo "run prepare.sh first" >&2; exit 1; }
UIDGID="$(docker_user)"
docker run --rm --init --user "${UIDGID}" --memory 16g --cpuset-cpus "${CPUSET:-0-19}" \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES= \
  -v "${BUILD}/qsa/qsa_indexer.py:/work/patched_qsa_indexer.py:ro" \
  -v "${SCRIPT_DIR}/test_qsa_chunk_tail.py:/work/test_qsa_chunk_tail.py:ro" \
  --workdir /work --entrypoint python3 "${IMAGE}" \
  /work/test_qsa_chunk_tail.py /work/patched_qsa_indexer.py
