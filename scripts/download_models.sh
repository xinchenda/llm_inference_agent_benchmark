#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${ROOT}/models"
mkdir -p "${MODELS}"

download() {
  local destination="$1"
  local url="$2"
  if [[ -s "${destination}" ]]; then
    echo "exists: ${destination}"
    return
  fi
  echo "downloading: ${destination}"
  curl --fail --location --retry 3 --output "${destination}.part" "${url}"
  mv "${destination}.part" "${destination}"
}

download "${MODELS}/Qwen3-0.6B-Q8_0.gguf" \
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true"
download "${MODELS}/Qwen3-1.7B-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"
download "${MODELS}/Qwen3-1.7B-Q8_0.gguf" \
  "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf?download=true"

download_hf_tree() {
  local repository="$1"
  local destination="$2"
  shift 2
  mkdir -p "${destination}"
  for relative in "$@"; do
    mkdir -p "$(dirname "${destination}/${relative}")"
    download "${destination}/${relative}" \
      "https://huggingface.co/${repository}/resolve/main/${relative}"
  done
}

download_hf_tree "BAAI/bge-small-en-v1.5" "${MODELS}/bge-small-en-v1.5" \
  model.safetensors config.json config_sentence_transformers.json modules.json \
  sentence_bert_config.json special_tokens_map.json tokenizer.json \
  tokenizer_config.json vocab.txt 1_Pooling/config.json

download_hf_tree \
  "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  "${MODELS}/ms-marco-MiniLM-L-6-v2" \
  model.safetensors config.json special_tokens_map.json tokenizer.json \
  tokenizer_config.json vocab.txt

if [[ ! -f "${MODELS}/models.sha256" ]]; then
  (
    cd "${MODELS}"
    find . -type f \
      ! -name models.sha256 \
      ! -name '*.aria2' \
      ! -name '*.part' \
      ! -name '*.incomplete' \
      -print0 \
      | sort -z \
      | xargs -0 shasum -a 256 > models.sha256
  )
fi

echo "Model files and SHA256 manifest are ready."

