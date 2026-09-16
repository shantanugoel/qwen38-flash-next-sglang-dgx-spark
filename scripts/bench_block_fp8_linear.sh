#!/usr/bin/env bash
# D2 step 1: block-FP8 vs BF16 linear micro-benchmark. One short GPU container,
# no model load. Never run beside a serving experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
require_spark
OUT_DIR="${OUT_DIR:-${ROOT}/results/d2-microbench}"
mkdir -p "${OUT_DIR}"
UIDGID="$(docker_user)"
extra_gpu_groups
docker run --rm --init --user "${UIDGID}" "${extra_groups[@]}" --gpus all --ipc host \
  --shm-size 8g --memory 64g --memory-swap 64g --cpuset-cpus "${CPUSET:-0-19}" \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e BATCHES -e ITERS -e OUT=/out/microbench.json \
  -v "${OUT_DIR}:/out" \
  -v "${SCRIPT_DIR}/bench_block_fp8_linear.py:/work/bench.py:ro" \
  --workdir /work --entrypoint python3 "${IMAGE}" /work/bench.py
