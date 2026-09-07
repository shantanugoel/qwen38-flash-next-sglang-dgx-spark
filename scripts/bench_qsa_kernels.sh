#!/usr/bin/env bash
# Isolated SM121 QSA kernel check against a free GPU. No model load.
# Background for first compile: nohup ./scripts/bench_qsa_kernels.sh > results/qsa_kernel.log 2>&1 &
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
require_spark

[[ -f "${QSA_BACKEND}" && -f "${BUILD}/sm121_varlen.py" && -d "${BUILD}/kda_kernels" ]] || {
  echo "patches missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${BUILD}/path_qsa.txt" && -f "${BUILD}/path_sm121_varlen.txt" && -f "${BUILD}/path_kda_kernels.txt" ]] || {
  echo "in-image paths missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}

QSA_IN_IMAGE="$(cat "${BUILD}/path_qsa.txt")"
SM121_IN_IMAGE="$(cat "${BUILD}/path_sm121_varlen.txt")"
KDA_IN_IMAGE="$(cat "${BUILD}/path_kda_kernels.txt")"
UIDGID="$(docker_user)"
extra_gpu_groups
NAME="${CONTAINER}-qsa-kernel"

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run --rm --name "${NAME}" \
  --user "${UIDGID}" \
  "${extra_groups[@]}" \
  --gpus all \
  --ipc host \
  --workdir /tmp \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -v "${QSA_BACKEND}:${QSA_IN_IMAGE}:ro" \
  -v "${BUILD}/sm121_varlen.py:${SM121_IN_IMAGE}:ro" \
  -v "${BUILD}/kda_kernels:${KDA_IN_IMAGE}:ro" \
  -v "${ROOT}/bench/qsa_sm121.py:/tmp/qsa_sm121.py:ro" \
  --entrypoint python3 \
  "${IMAGE}" \
  /tmp/qsa_sm121.py
