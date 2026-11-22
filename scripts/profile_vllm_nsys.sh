#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-1.7B}"
OUTPUT="${2:-artifacts/p2/nsight/vllm_trace}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "Nsight Systems (nsys) is not installed." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to create a fake GPU profile." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"
exec nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=cpu \
  --output="${OUTPUT}" \
  vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching
