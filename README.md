# Local LLM Inference, RAG, and Agent Benchmark

This repository evaluates pretrained open-weight language models for a local
document assistant. It covers inference serving, quantized model comparison,
cited RAG, a controlled tool-using agent, fixed quality questions, latency and
memory measurements, and reproducible reports.

It does **not** train, fine-tune, or generate image models. Image capabilities
are outside the core project.

## Tested environment

- Apple M4, 10 cores, 16 GB unified memory
- macOS 26.6.2
- Python 3.12
- `llama.cpp` 1.0.0 with the Apple Metal backend
- PyTorch 2.15 with MPS

All performance and quality numbers in [reports/validation_report.md](reports/validation_report.md)
come from this machine. The report records the exact model files, evaluation
set, context, generation settings, and command used.

## What is implemented

- Three real pretrained GGUF configurations:
  Qwen3-0.6B Q8_0, Qwen3-1.7B Q4_K_M, and Qwen3-1.7B Q8_0.
- A single local OpenAI-compatible inference client and lifecycle-managed
  `llama-server` process.
- 66 technical documents: 60 pinned public sources with URL, commit, license,
  byte size, and SHA256, plus six project-authored policy/evaluation fixtures.
- Section-aware chunking, BM25, dense retrieval, reciprocal-rank fusion,
  cross-encoder reranking, and document-level evidence deduplication.
- Cited answers with explicit no-answer and conflicting-evidence behavior.
- A bounded Agent with four tools: knowledge search, model comparison,
  performance calculation, and report generation. The default planner asks the
  local LLM for a validated JSON tool plan; a rule planner remains available as
  an interpretable baseline.
- Explicit Agent failure handling: planner schema/route validation, tool
  timeouts, invalid arguments, duplicate calls, step limits, one error-guided
  JSON repair attempt, and complete tool traces.
- A fixed 100-question set covering factual, long-context, unanswerable,
  false-premise, conflict, and tool-call cases.
- RAG ablations: dense vector, hybrid retrieval, and hybrid plus reranking.
- FastAPI endpoints for health, search, cited answering, Agent calls, and
  Prometheus-style counters. Request-level timeouts return 504; failed Agent
  tool execution returns 502 with the partial trace.
- Unit and integration tests for malformed JSON, no evidence, timeout, illegal
  tool arguments, duplicate calls, long documents, and idempotent requests.

## Quick start

```bash
cd llm_inference_agent_benchmark
make setup
brew install llama.cpp
make models
make corpus
make index
make test
```

The setup uses setuptools compatibility editable mode because the project path
contains non-ASCII characters; this keeps `.venv/bin/pytest`, the installed CLI,
and `src/llm_bench` on the same live source tree.

The model download is several GiB and is not stored in Git. Model files are
ignored by `.gitignore`; `models/models.sha256` records their hashes.

## Commands

Rebuild the document corpus from pinned public sources:

```bash
./llm-bench corpus build --refresh
```

Rebuild the retrieval index:

```bash
./llm-bench index build
```

Run all eight quality configurations on the fixed 100 questions:

```bash
./llm-bench eval run
```

Run model/context/concurrency performance tests:

```bash
./llm-bench perf run
```

Start the model and FastAPI service with one command:

```bash
./llm-bench serve --port 8000
```

The repository-local wrapper always imports the live `src/` tree. Global
options go before the subcommand:

```bash
./llm-bench --config configs/default.yaml eval run --set eval.max_questions=10
```

## API

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-1' \
  -d '{"question":"How does vLLM PagedAttention manage KV cache memory?","mode":"hybrid_rerank"}'

curl -X POST http://127.0.0.1:8000/agent \
  -H 'Content-Type: application/json' \
  -d '{"question":"Compare model qwen3_0_6b_q8 with qwen3_1_7b_q4."}'
```

Set `LLM_BENCH_API_TOKEN` to require a bearer token. The local service has an
in-memory 30-requests/minute limiter, request IDs, structured timing logs, and
an idempotency cache for repeated `Idempotency-Key` requests.

## Architecture

```text
Markdown corpus
  -> section-aware chunks
  -> BM25 + dense embeddings
  -> RRF fusion
  -> optional cross-encoder reranking
  -> document-level evidence [E1]...[E5]
  -> local Qwen3 answer with citations

question
  -> LLM JSON planner, validated against tool schemas and allowed routes
     (rule planner is retained as a baseline)
  -> typed tool registry
  -> bounded execution and trace
  -> cited answer or deterministic tool result

llama-server (Metal/CUDA/CPU)
  -> OpenAI-compatible local API
  -> FastAPI /ask and /agent
  -> JSON/CSV results, metrics, figures, validation report
```

The design favors explicit interfaces over a large framework dependency. See
[docs/open_source_research.md](docs/open_source_research.md) for the GitHub
survey and [docs/architecture.md](docs/architecture.md) for failure behavior and
extension points.

## Metrics

The quality report includes:

- Retrieval Recall@5 at document level
- Answer accuracy from fixed required evidence concepts
- Citation precision, support, and invalid-citation rate
- Tool selection accuracy
- Task success rate and failure taxonomy
- p50/p95 end-to-end latency and time-to-first-token
- Twelve manually reviewed samples, kept separate from automated labels

The performance report includes:

- Model file size and peak process RSS
- Cold start
- TTFT p50/p95
- End-to-end latency p50/p95
- Per-request and aggregate output tokens/s
- Prompt-length, concurrency, model-size, and quantization comparisons
- Local incremental cost: USD 0 for the open-source self-hosted stack, with
  hardware purchase and electricity explicitly excluded

## Reproducibility and controls

- Temperature 0, seed 42, fixed maximum output tokens, and a fixed question set
  for quality tests.
- Prompt and output lengths are fixed inside each performance comparison.
- Two warm-up requests precede measured performance batches.
- Corpus sources and model repositories are pinned and hashed.
- The report identifies hardware and software versions and records failed or
  unsupported paths.

## Current limitations

- The measured backend is `llama.cpp` on Apple Metal. A vLLM adapter is a clear
  extension path, not a completed local measurement.
- The Docker/Linux path is supplied but could not be executed on this Mac because
  Docker is not installed.
- The service limiter is process-local. Multi-replica production deployment
  needs a shared limiter and an external identity provider.
- The cross-encoder and answer model are intentionally small. The benchmark
  measures this constrained local configuration rather than claiming frontier
  model quality.

## Deliverables

- [Validation report](reports/validation_report.md)
- [Agent tool failure analysis](reports/agent_failure_analysis.md)
- [Architecture and failure handling](docs/architecture.md)
- [Open-source decision record](docs/open_source_research.md)
- Per-question JSONL and aggregate JSON/CSV under `artifacts/`
- Charts under `reports/figures/`

## Licenses

Project code is MIT. Model weights are Apache-2.0. Public documentation retains
the license of its upstream repository; each file is recorded in
`data/corpus/source_manifest.json`. Local evaluation-policy documents are
project-authored test fixtures.

