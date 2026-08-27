#!/usr/bin/env bash
# Pull the SGLang image, extract the two files that need patching, patch them,
# download the NVFP4 checkpoint as the host user (never root).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_hf_token
require_spark

mkdir -p "${BUILD}" "${HF_CACHE}" "${PLE_DIR}" "${SGLANG_CACHE}"

echo "image ${IMAGE}"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}"

extract() {
  local src_in_image="$1" dest="$2"
  local cid
  cid="$(docker create "${IMAGE}")"
  docker cp "${cid}:${src_in_image}" "${dest}"
  docker rm -f "${cid}" >/dev/null
}

echo "extracting sources from image..."
# Resolve in-image paths from the running Python package when possible.
qwen4_path="$(docker run --rm --entrypoint python3 "${IMAGE}" -c \
  'import sglang.srt.models.qwen4_exp as m; print(m.__file__)' | tail -1)"
qsa_path="$(docker run --rm --entrypoint python3 "${IMAGE}" -c \
  'import sglang.srt.layers.attention.qwen_sparse_attn_backend as m; print(m.__file__)' | tail -1)"
printf '%s\n' "${qwen4_path}" > "${BUILD}/path_qwen4_exp.txt"
printf '%s\n' "${qsa_path}" > "${BUILD}/path_qsa.txt"
extract "${qwen4_path}" "${QWEN4_BACKEND}"
extract "${qsa_path}" "${QSA_BACKEND}"

python3 "${ROOT}/patches/ple_mmap.py" "${QWEN4_BACKEND}"
python3 "${ROOT}/patches/qsa_drop_sm121_sdpa.py" "${QSA_BACKEND}"
python3 "${ROOT}/patches/qsa_trtllm_sm120.py" "${QSA_BACKEND}"
python3 -m py_compile "${QWEN4_BACKEND}" "${QSA_BACKEND}"

python3 - <<PY
from pathlib import Path
qwen4 = Path("${QWEN4_BACKEND}").read_text()
qsa = Path("${QSA_BACKEND}").read_text()
assert "_alloc_ple_table" in qwen4, "PLE mmap helper missing"
assert "_alloc_ple_table(source_weight.shape" in qwen4
assert "if is_sm121():" not in qsa, "SM121 SDPA intercept still present"
assert "is_sm100_supported() or is_sm120_supported()" in qsa, "sm_120 gate missing"
print("patches ok")
PY

snapshot_ready() {
  [[ -f "${SNAPSHOT}/config.json" ]] || return 1
  [[ -f "${SNAPSHOT}/model.safetensors.index.json" ]] || return 1
  if find "${HF_CACHE}/hub/models--${MODEL//\//--}" -name '*.incomplete' -print -quit | grep -q .; then
    return 1
  fi
  return 0
}

if snapshot_ready; then
  echo "checkpoint ${REVISION} already in ${HF_CACHE}"
else
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is empty. export it or put it in ${ROOT}/.env" >&2
    exit 1
  fi
  echo "downloading ${MODEL} @ ${REVISION} into ${HF_CACHE} as uid $(id -u)"
  HF_HOME="${HF_CACHE}" HF_TOKEN="${HF_TOKEN}" hf download "${MODEL}" \
    --revision "${REVISION}" --max-workers 8
  snapshot_ready || { echo "download finished but snapshot is incomplete" >&2; exit 1; }
fi

echo "prepare complete. serve: ${SCRIPT_DIR}/serve.sh"
