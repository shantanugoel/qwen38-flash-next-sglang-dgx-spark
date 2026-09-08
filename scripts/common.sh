# Shared pins. Sourced by prepare/serve/stop/smoke. Not meant to be run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# U3 pin: SGLang 4ccff141 (MTP token-0 router #38290 + native PLE file backend).
# Hub tags move; clones pull this digest. Local alias:
#   lmsysorg/sglang:dev-qwen38-next-local-4ccff14
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6}"
CONTAINER="${CONTAINER:-qwen38-flash-next}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
REVISION="${REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
SERVED_NAME="${SERVED_NAME:-qwen38-flash-next-nvfp4-mtp}"
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
PLE_DIR="${PLE_DIR:-$ROOT/data/ple}"
SGLANG_CACHE="${SGLANG_CACHE:-$ROOT/data/sglang-cache}"
BUILD="${BUILD:-$ROOT/build}"
QWEN4_BACKEND="${BUILD}/qwen4_exp.py"
QSA_BACKEND="${BUILD}/qwen_sparse_attn_backend.py"
SPEC_UTILS_BACKEND="${BUILD}/spec_utils.py"
PLE_TABLE_BACKEND="${BUILD}/qwen4_exp_ple_table.py"
SNAPSHOT="${HF_CACHE}/hub/models--${MODEL//\//--}/snapshots/${REVISION}"

docker_user() {
  local uid gid
  if [[ -n "${DOCKER_USER:-}" ]]; then
    printf '%s' "${DOCKER_USER}"
    return
  fi
  if [[ -n "${SUDO_UID:-}" ]]; then
    uid="${SUDO_UID}"
    gid="${SUDO_GID:-$SUDO_UID}"
  else
    uid="$(id -u)"
    gid="$(id -g)"
  fi
  if [[ "${uid}" == 0 ]]; then
    echo "refusing to run the container as root (set DOCKER_USER=uid:gid)" >&2
    exit 1
  fi
  printf '%s:%s' "${uid}" "${gid}"
}

extra_gpu_groups() {
  local g gid
  extra_groups=()
  for g in video render; do
    gid="$(getent group "${g}" | cut -d: -f3 || true)"
    if [[ -n "${gid}" ]]; then
      extra_groups+=(--group-add "${gid}")
    fi
  done
}

load_hf_token() {
  if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
  fi
}

require_spark() {
  local arch compute
  arch="$(uname -m)"
  compute="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  [[ "${arch}" == aarch64 ]] || { echo "Expected aarch64, got ${arch}" >&2; exit 1; }
  [[ "${compute}" == 12.1 ]] || { echo "Expected SM 12.1, got ${compute}" >&2; exit 1; }
}
