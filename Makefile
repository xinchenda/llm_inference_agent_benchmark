SHELL := /bin/bash
PYTHON ?= .venv/bin/python
LLM_BENCH ?= ./llm-bench
MODEL ?= qwen3_1_7b_q4
PORT ?= 8000

.PHONY: setup install models corpus index eval perf experiment experiment-report prefix-cache prefix-cache-report serve test lint all clean-results

setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps --config-settings editable_mode=compat
	$(PYTHON) scripts/ensure_editable_path.py

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps --config-settings editable_mode=compat
	$(PYTHON) scripts/ensure_editable_path.py

models:
	./scripts/download_models.sh

corpus:
	$(LLM_BENCH) --config configs/default.yaml corpus build

index:
	$(LLM_BENCH) --config configs/default.yaml index build

eval:
	$(LLM_BENCH) --config configs/default.yaml --set eval.max_questions=100 eval run

perf:
	$(LLM_BENCH) --config configs/default.yaml perf run

experiment:
	$(LLM_BENCH) --config configs/default.yaml experiment run --spec configs/experiments_p0_p1.yaml

experiment-report:
	$(LLM_BENCH) --config configs/default.yaml experiment report \
		--ledger artifacts/experiments/p0_p1_rsi_prefix_v3_20261123/ledger.sqlite

prefix-cache:
	$(LLM_BENCH) --config configs/default.yaml experiment run --spec configs/prefix_cache_ablation.yaml

prefix-cache-report:
	$(LLM_BENCH) --config configs/default.yaml experiment report \
		--ledger artifacts/experiments/prefix_cache_ablation_20261123/ledger.sqlite \
		--output reports/prefix_cache

report:
	$(LLM_BENCH) --config configs/default.yaml report build

serve:
	$(LLM_BENCH) --config configs/default.yaml serve --port $(PORT)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

all: corpus index eval perf report test

clean-results:
	rm -rf artifacts/eval artifacts/perf artifacts/index

