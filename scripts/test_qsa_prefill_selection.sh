#!/usr/bin/env bash
# U8a: run sglang#38209's own kernel tests against the patched QSA sources.
# One short GPU container; no server. Never run it beside a serving experiment.
#   OUT_DIR=results/u8a-tests ./scripts/test_qsa_prefill_selection.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
require_spark

OUT_DIR="${OUT_DIR:-${ROOT}/results/u8a-kernel-tests}"
mkdir -p "${OUT_DIR}"
[[ -f "${BUILD}/qsa_prefill_selection.on" ]] || {
  echo "overlay is off; run QSA_PREFILL_SELECTION=1 ./scripts/prepare.sh first" >&2
  exit 1
}
QSA_DIR_IN_IMAGE="$(cat "${BUILD}/path_qsa_dir.txt")"
QSA_IN_IMAGE="$(cat "${BUILD}/path_qsa.txt")"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/test/registered/kernels/ops/attention"
cid="$(docker create "${IMAGE}")"
docker cp "${cid}:/sgl-workspace/sglang/test/registered/kernels/test_qsa.py" \
  "${work}/test/registered/kernels/test_qsa.py"
docker rm -f "${cid}" >/dev/null
git -C "${work}" apply -p1 "${ROOT}/patches/qsa_prefill_selection_tests.diff"

UIDGID="$(docker_user)"
extra_gpu_groups
docker run --rm --name "${CONTAINER}-u8a-tests" --init \
  --user "${UIDGID}" "${extra_groups[@]}" --gpus all --ipc host --shm-size 8g \
  --memory 64g --memory-swap 64g --cpuset-cpus "${CPUSET:-0-19}" \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -v "${QSA_BACKEND}:${QSA_IN_IMAGE}:ro" \
  -v "${BUILD}/qsa/kernel.py:${QSA_DIR_IN_IMAGE}/kernel.py:ro" \
  -v "${BUILD}/qsa/metadata.py:${QSA_DIR_IN_IMAGE}/metadata.py:ro" \
  -v "${BUILD}/qsa/qsa_indexer.py:${QSA_DIR_IN_IMAGE}/qsa_indexer.py:ro" \
  -v "${BUILD}/sm121_varlen.py:$(cat "${BUILD}/path_sm121_varlen.txt"):ro" \
  -v "${work}/test/registered/kernels/test_qsa.py:/sgl-workspace/sglang/test/registered/kernels/test_qsa.py:ro" \
  -v "${work}/test/registered/kernels/ops/attention/test_qsa_prefill_compressed_pack.py:/sgl-workspace/sglang/test/registered/kernels/ops/attention/test_qsa_prefill_compressed_pack.py:ro" \
  --workdir /sgl-workspace/sglang \
  --entrypoint python3 "${IMAGE}" -m pytest -q \
    test/registered/kernels/test_qsa.py \
    test/registered/kernels/ops/attention/test_qsa_prefill_compressed_pack.py
