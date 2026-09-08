#!/usr/bin/env bash
# Serve Flash-Next NVFP4. Detached docker run, non-root, PLE mmap + QSA SM121
# Triton varlen (#36845, 2026-08-28) + MTP 3/1/4 unquant + CUDA graphs.
# Default bind 127.0.0.1:30000.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_hf_token
# Experiment identity was validated before launch; .env cannot change it later.
if [[ -n "${EXPERIMENT_IMAGE_ID:-}" ]]; then
  IMAGE="${EXPERIMENT_IMAGE_ID}"
  MODEL="${RECIPE_MODEL:?}"
  REVISION="${EXPERIMENT_REVISION:?}"
  PLE_DIR="${EXPERIMENT_PLE_DIR:?}"
fi
require_spark

PORT="${PORT:-30000}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"

# Tunables. Defaults are the shipped recipe; override per experiment, e.g.
#   MEMFRAC=0.85 PREFILL=2048 ./scripts/serve.sh
MEMFRAC="${MEMFRAC:-0.95}"
PREFILL="${PREFILL:-4096}"
MAX_RUNNING="${MAX_RUNNING:-4}"
CONTEXT="${CONTEXT:-262144}"
MAX_TOTAL="${MAX_TOTAL:-524288}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
MAMBA_STRATEGY="${MAMBA_STRATEGY:-extra_buffer}"
PAGE_SIZE="${PAGE_SIZE:-64}"
# Extra raw sglang flags, word-split on purpose: EXTRA_ARGS="--strip-thinking-cache"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

opt=()
[[ -n "${CUDA_GRAPH_MAX_BS}" ]] && opt+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS}")
if [[ "${SPEC:-nextn}" == "off" ]]; then
  SPEC_ARGS=()
else
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${SPEC_STEPS}"
    --speculative-eagle-topk "${SPEC_TOPK}"
    --speculative-num-draft-tokens "${SPEC_DRAFT}"
    --speculative-draft-model-quantization unquant
  )
fi

docker image inspect "${IMAGE}" >/dev/null
[[ -f "${QWEN4_BACKEND}" && -f "${QSA_BACKEND}" && -f "${SPEC_UTILS_BACKEND}" && -f "${BUILD}/sm121_varlen.py" ]] || {
  echo "patches missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${SNAPSHOT}/config.json" ]] || {
  echo "checkpoint missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${BUILD}/path_qwen4_exp.txt" && -f "${BUILD}/path_qsa.txt" && -f "${BUILD}/path_sm121_varlen.txt" && -f "${BUILD}/path_spec_utils.txt" ]] || {
  echo "in-image paths missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}

QWEN4_IN_IMAGE="$(cat "${BUILD}/path_qwen4_exp.txt")"
QSA_IN_IMAGE="$(cat "${BUILD}/path_qsa.txt")"
SM121_IN_IMAGE="$(cat "${BUILD}/path_sm121_varlen.txt")"
SPEC_UTILS_IN_IMAGE="$(cat "${BUILD}/path_spec_utils.txt")"
UIDGID="$(docker_user)"
extra_gpu_groups
mkdir -p "${PLE_DIR}" "${SGLANG_CACHE}"

MOUNTS=(
  -v "${QWEN4_BACKEND}:${QWEN4_IN_IMAGE}:ro"
  -v "${QSA_BACKEND}:${QSA_IN_IMAGE}:ro"
  -v "${SPEC_UTILS_BACKEND}:${SPEC_UTILS_IN_IMAGE}:ro"
  -v "${BUILD}/sm121_varlen.py:${SM121_IN_IMAGE}:ro"
)
if [[ -f "${BUILD}/path_ple_table.txt" && -f "${PLE_TABLE_BACKEND}" ]]; then
  MOUNTS+=(-v "${PLE_TABLE_BACKEND}:$(cat "${BUILD}/path_ple_table.txt"):ro")
fi

if [[ -f "${BUILD}/path_ple_table.txt" && -z "${PLE_OFFLOAD_BACKEND:-}" ]]; then
  # Native #37068 defaults to pinned host RAM, which OOMs the 48 GiB table on GB10.
  PLE_OFFLOAD_BACKEND=file
fi
if [[ -n "${PLE_OFFLOAD_BACKEND:-}" ]]; then
  opt+=(--ple-offload-backend "${PLE_OFFLOAD_BACKEND}")
  opt+=(--ple-offload-dir /ple)
fi

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
  -e SGLANG_QWEN4_PLE_FILE_DIR=/ple \
  -e SGLANG_QWEN4_PLE_FILE_PREFETCH="${SGLANG_QWEN4_PLE_FILE_PREFETCH:-0}" \
  -e SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB="${SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB:-0}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN \
  -v "${HF_CACHE}:/huggingface" \
  -v "${SGLANG_CACHE}:/tmp/.cache/sglang" \
  -v "${PLE_DIR}:/ple" \
  "${MOUNTS[@]}" \
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
    --page-size "${PAGE_SIZE:-64}" \
    --mamba-radix-cache-strategy "${MAMBA_STRATEGY:-extra_buffer}" \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 20 \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size "${PREFILL}" \
    --max-running-requests "${MAX_RUNNING}" \
    --max-total-tokens "${MAX_TOTAL}" \
    --context-length "${CONTEXT}" \
    --mem-fraction-static "${MEMFRAC}" \
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
    --enable-gdn-replayssm-spec \
    "${opt[@]}" \
    "${SPEC_ARGS[@]}" \
    "${EXTRA[@]}"

echo "started ${CONTAINER} on ${BIND_ADDR}:${PORT} as ${UIDGID}"
echo "first load ~8–20 min (PLE mmap fill is quiet). poll: ${SCRIPT_DIR}/wait_ready.sh"
echo "ready: GET http://${BIND_ADDR}:${PORT}/health  then  ${SCRIPT_DIR}/smoke.sh"
