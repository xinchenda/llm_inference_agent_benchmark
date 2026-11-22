#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-0.6B}"
OUTPUT="${2:-artifacts/p2/nsight/vllm_trace}"
MODE="${3:-graph}"
REQUESTS="${4:-8}"
PREFIX_WORDS="${5:-1500}"
MAX_TOKENS="${6:-16}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "Nsight Systems (nsys) is not installed." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to create a fake GPU profile." >&2
  exit 2
fi

case "${MODE}" in
  graph)
    ENABLE_CUDA_GRAPH=true
    ;;
  eager)
    ENABLE_CUDA_GRAPH=false
    ;;
  *)
    echo "MODE must be 'graph' or 'eager'." >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$(dirname "${OUTPUT}")"

CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/vllm-nsys.XXXXXX.yaml")"
cleanup() {
  rm -f "${CONFIG_FILE}"
}
trap cleanup EXIT

cat >"${CONFIG_FILE}" <<YAML
vllm:
  profile:
    model: ${MODEL}
    max_model_len: 4096
    max_num_seqs: 8
    max_num_batched_tokens: 8192
    gpu_memory_utilization: 0.85
    tensor_parallel_size: 1
    continuous_batching: true
    enable_prefix_caching: true
    enable_cuda_graph: ${ENABLE_CUDA_GRAPH}
    dtype: auto
    kv_cache_dtype: auto
YAML

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
nsys profile \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=cpu \
  --wait=all \
  --output="${OUTPUT}" \
  "${PYTHON:-python}" "${ROOT_DIR}/scripts/run_vllm_smoke.py" \
  --config "${CONFIG_FILE}" \
  --candidate profile \
  --requests "${REQUESTS}" \
  --prefix-words "${PREFIX_WORDS}" \
  --max-tokens "${MAX_TOKENS}" \
  --output "${OUTPUT}.json"
