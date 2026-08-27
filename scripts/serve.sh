#!/usr/bin/env bash
# Serve Flash-Next NVFP4. Detached docker run, non-root, PLE mmap + QSA sm_120
# + MTP 3/1/4 unquant + CUDA graphs. Default bind 127.0.0.1:30000.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_hf_token
require_spark

PORT="${PORT:-30000}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"

docker image inspect "${IMAGE}" >/dev/null
[[ -f "${QWEN4_BACKEND}" && -f "${QSA_BACKEND}" ]] || {
  echo "patches missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${SNAPSHOT}/config.json" ]] || {
  echo "checkpoint missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${BUILD}/path_qwen4_exp.txt" && -f "${BUILD}/path_qsa.txt" ]] || {
  echo "in-image paths missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}

QWEN4_IN_IMAGE="$(cat "${BUILD}/path_qwen4_exp.txt")"
QSA_IN_IMAGE="$(cat "${BUILD}/path_qsa.txt")"
UIDGID="$(docker_user)"
extra_gpu_groups
mkdir -p "${PLE_DIR}" "${SGLANG_CACHE}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER}" --init \
  --user "${UIDGID}" \
  "${extra_groups[@]}" \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  --memory 116g \
  --memory-swap 116g \
  --cpuset-cpus "${CPUSET:-0-19}" \
  --workdir /tmp \
  -p "${BIND_ADDR}:${PORT}:30000" \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -e HF_TOKEN \
  -e HF_HOME=/huggingface \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -v "${HF_CACHE}:/huggingface" \
  -v "${SGLANG_CACHE}:/tmp/.cache/sglang" \
  -v "${PLE_DIR}:/ple" \
  -v "${QWEN4_BACKEND}:${QWEN4_IN_IMAGE}:ro" \
  -v "${QSA_BACKEND}:${QSA_IN_IMAGE}:ro" \
  "${IMAGE}" \
  sglang serve \
    --model-path "${MODEL}" \
    --revision "${REVISION}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 20 \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --max-total-tokens 524288 \
    --context-length 262144 \
    --mem-fraction-static 0.95 \
    --allow-auto-truncate \
    --ple-offload-embedding \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --disable-prefill-cuda-graph \
    --disable-flashinfer-autotune \
    --enable-metrics \
    --enable-cache-report \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant

echo "started ${CONTAINER} on ${BIND_ADDR}:${PORT} as ${UIDGID}"
echo "first load ~8–20 min (PLE mmap fill is quiet). follow: docker logs -f ${CONTAINER}"
echo "ready: GET http://${BIND_ADDR}:${PORT}/health  then  ${SCRIPT_DIR}/smoke.sh"
