SHELL := /bin/bash
PYTHON ?= .venv/bin/python
LLM_BENCH ?= ./llm-bench
MODEL ?= qwen3_1_7b_q4
PORT ?= 8000

.PHONY: setup install models corpus index eval perf serve test lint all clean-results

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

