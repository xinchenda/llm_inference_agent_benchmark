#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v python3.12 >/dev/null; then
  echo "python3.12 is required" >&2
  exit 1
fi
if ! command -v llama-server >/dev/null; then
  echo "Install llama-server from a pinned llama.cpp release before continuing." >&2
  exit 1
fi

python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install . --no-deps
./scripts/download_models.sh
.venv/bin/llm-bench --config configs/default.yaml corpus build --refresh
.venv/bin/llm-bench --config configs/default.yaml index build
.venv/bin/llm-bench --config configs/default.yaml serve --host 0.0.0.0 --port 8000

