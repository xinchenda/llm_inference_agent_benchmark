# Local LLM Inference, RAG, and Agent Benchmark

This repository evaluates pretrained open-weight language models for a local
document assistant. It covers inference serving, quantized model comparison,
cited RAG, a controlled tool-using agent, fixed quality questions, latency and
memory measurements, deterministic experiment optimization, and reproducible
reports.

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
- A quality-gated experiment optimizer with immutable SQLite evidence, resource
  locking, dynamic ports, health checks, process-tree cleanup, controlled
  retry, and explicit failure classification.
- Three fixed serving loads: interactive chat, RAG-shaped long-context reading,
  and concurrent batch generation.
- Equal-budget `default`, `grid`, `random`, `ucb1`, and history-driven `agent`
  strategies. The Agent explores every untested legal candidate, records
  structured Reflection/Memory, and can select only registered configurations.
- Repeated trials, percentile-bootstrap intervals, TTFT, TPOT/ITL, p50/p95/p99,
  output tokens/s, process-tree RSS, fixed-QPS achieved rate, and goodput.
- Pareto analysis, best-so-far curves, similar-history retrieval, and an
  automatically generated optimization report.

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

Run the quality-gated P0/P1 optimization experiment and build its report:

```bash
./llm-bench experiment run --spec configs/experiments_p0_p1.yaml
./llm-bench experiment report \
  --ledger artifacts/experiments/p0_p1_rsi_prefix_v3_20261123/ledger.sqlite

./llm-bench experiment run --spec configs/prefix_cache_ablation.yaml
./llm-bench experiment report \
  --ledger artifacts/experiments/prefix_cache_ablation_20261123/ledger.sqlite \
  --output reports/prefix_cache
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

ExperimentSpec (immutable candidate and load space)
  -> default/grid/random/ucb1/agent policy (one legal candidate)
  -> resource lock, dynamic port, isolated llama-server process
  -> deterministic quality gate
  -> closed-loop or fixed-QPS repeated performance runs
  -> append-only SQLite ledger
  -> Pareto, stability, best-so-far, similar-history report
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

The P0/P1 optimizer additionally reports:

- Fixed-QPS target versus measured request-arrival rate
- TPOT/ITL, p99 latency, output tokens/s, process-tree RSS, and goodput
- Repeat variability and deterministic 95% bootstrap intervals
- Quality-gated Pareto points and equal-budget strategy curves
- Similar historical trials and deterministic Agent policy reasons
- Shared-prefix Cache ablation with request-level llama.cpp `cache_n`
- Structured bottleneck reflections, negative memory, and legal-candidate fallback

### Actual P0/P1 local result

The final audit run contains 36 immutable trial records: one common unoptimized
anchor and seven scored trials for each of five strategies. All 36 passed the
fixed quality gates. The fixed-QPS workload targeted 2.0 requests/s and measured
an arrival rate of 1.981-2.007 requests/s.

| Strategy | Best quality-passing utility | Selected candidate |
| --- | ---: | --- |
| default | 0.63800 | `q06_f16_c4096_p1_cp_off` |
| grid | 1.08436 | `q06_f16_c8192_p4_b128_ub64_cp_on` |
| random | 1.61242 | `q06_q8_q8_c4096_p1_cp_on` |
| ucb1 | 1.36733 | `q06_q8_q8_c4096_p1_cp_on` |
| agent | 1.27079 | `q06_f16_c8192_p4_b128_ub64_cp_on` |

All strategies begin from the same `cp_off` configuration. After two diagnostic
rounds the RSI Agent prunes low-value branches and locks a high-recommendation
candidate instead of traversing every option. It improved its own best-so-far by
140.58% over the unoptimized start, but random and UCB1 found higher utilities in
this run. Structured reflections classified 19 trials with no material
bottleneck, 10 with RSS pressure, and 7 with TTFT pressure.

The separate shared-prefix ablation used the same 0.6B model and identical
batch/context settings for each enabled/disabled pair:

| KV cache | Hit rate on/off | TTFT p50 reduction | TTFT p95 reduction | Goodput change | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| f16/f16 | 0.936 / 0 | 92.32% | 67.15% | +422.02% | 0 |
| q8_0/q8_0 | 0.936 / 0 | 93.25% | 66.35% | +478.31% | 0 |

Both evidence-valid cache configurations had zero missing `cache_n`
observations. These are local shared-prefix results, not universal cache-hit
claims. The independent historical report is retained in
[reports/prefix_cache/experiment_report.md](reports/prefix_cache/experiment_report.md).
Its raw ledger was removed during the user-requested final artifact cleanup and
was not rerun.

## Reproducibility and controls

- Temperature 0, seed 42, fixed maximum output tokens, and a fixed question set
  for quality tests.
- Prompt and output lengths are fixed inside each performance comparison.
- Two warm-up requests precede measured performance batches.
- Corpus sources and model repositories are pinned and hashed.
- The report identifies hardware and software versions and records failed or
  unsupported paths.

## Current limitations

- The main P0/P1 backend is `llama.cpp` on Apple Metal. P2 vLLM measurements were
  executed separately on an NVIDIA RTX 4090D and are never merged into the Mac
  performance table.
- The Docker/Linux path is supplied but could not be executed on this Mac because
  Docker is not installed.
- The service limiter is process-local. Multi-replica production deployment
  needs a shared limiter and an external identity provider.
- The cross-encoder and answer model are intentionally small. The benchmark
  measures this constrained local configuration rather than claiming frontier
  model quality.
- The experimental quality gates are fast deterministic proxies. They verify an
  exact answer, a critical fact, JSON structure, and timeout behavior but do not
  replace the separate 100-question quality evaluation.
- Goodput and p95 comparisons describe this Mac and this fixed local workload.
  They are not GPU-serving or production-SLA results.

## P2 GPU serving

The local Mac still has no NVIDIA GPU or CUDA. P2 was executed on a remote Linux
RTX 4090D with driver 595.71.05, vLLM 0.34.0, PyTorch 2.14+cu130, and
Qwen3-0.6B.

The repository now contains:

- A validated vLLM lifecycle adapter with the required serving flags.
- OpenAI-compatible smoke execution and Prometheus metrics parsing.
- Docker/Compose definitions pinned to vLLM 0.34.0.
- A same-workload comparison gate that refuses cross-hardware ratios.
- Nsight Systems and Triton GPU validation scripts.
- A capability audit and GPU-host runbook.
- Raw-result summaries with source hashes and no remote host identity.

Measured vLLM results:

- 1500-word shared prefix, 8 requests.
- Prefix Cache off: response TTFT mean `0.12329 s`; no cache counter.
- Prefix Cache on: hit rate `0.86522`, response TTFT mean `0.11799 s`.
- Mean TTFT reduction was `4.30%`; p95 histogram TTFT was slightly worse
  (`0.092 s` off, `0.096 s` on). Cache reuse is confirmed, but the latency gain
  is small for this 0.6B workload.
- CUDA Graph: TTFT p95 `0.096 s`, TPOT `0.002425 s`, E2E p95 `0.285 s`.
- Eager: TTFT p95 `0.235 s`, TPOT `0.023692 s`, E2E p95 `0.490 s`.

GPU Triton RMSNorm completed for fp32/fp16/bf16. Triton was `0.0388 ms`,
`0.0227 ms`, and `0.0213 ms`, corresponding to `1.94x`, `2.08x`, and `2.31x`
speedups over PyTorch eager. Maximum absolute errors were `1.91e-6`, `0.0078125`,
and `0.0625`.

SGLang 0.5.10.post1 completed the same Qwen3-0.6B, 8-request, 1500-word
RadixAttention workload. It reported cache hit rate `0.99573`, client TTFT mean
`0.12232 s`, metrics p50 TTFT `0.06429 s`, p95 TTFT `0.355 s`, and E2E p95
`0.355 s`. Relative to vLLM, SGLang had lower p50 TTFT but higher p95/E2E tail
latency. The cache metrics use `sglang:cache_hit_rate`, not vLLM's
hits/queries counters.

The SGLang installation changed PyTorch to 2.9.1 in the shared environment and
may have invalidated vLLM after measurement. Nsight Systems 2025.3.1 is
installed, but no capture has completed. A same-GPU llama.cpp versus vLLM
comparison remains pending.

## Deliverables

- [Validation report](reports/validation_report.md)
- [P0/P1 optimization report](reports/experiments/experiment_report.md)
- [Prefix Cache ablation report](reports/prefix_cache/experiment_report.md)
- [Automated experiment matrix](configs/experiments_p0_p1.yaml)
- [Focused cache matrix](configs/prefix_cache_ablation.yaml)
- [Strategy comparison](reports/experiments/strategy_comparison.csv)
- [Stability intervals](reports/experiments/trial_metrics.csv)
- [Pareto frontier](reports/experiments/pareto_frontier.csv)
- [Best-so-far curves](reports/experiments/best_so_far.csv)
- [Agent tool failure analysis](reports/agent_failure_analysis.md)
- [P2 capability matrix](reports/p2_capability_matrix.md)
- [P2 validation report](reports/p2_validation_report.md)
- [P2 remote result documentation](reports/p2/README.md)
- [Sanitized vLLM result summary](reports/p2/vllm_summary.json)
- [Sanitized SGLang result summary](reports/p2/sglang_summary.json)
- [Derived vLLM comparisons](reports/p2/derived_comparisons.json)
- [GPU Triton RMSNorm result](reports/p2/triton_rmsnorm_gpu.json)
- [P2 GPU runbook](docs/p2_gpu_runbook.md)
- [Architecture and failure handling](docs/architecture.md)
- [Open-source decision record](docs/open_source_research.md)
- Per-question JSONL and aggregate JSON/CSV under `artifacts/`
- Charts under `reports/figures/`

## Licenses

Project code is MIT. Model weights are Apache-2.0. Public documentation retains
the license of its upstream repository; each file is recorded in
`data/corpus/source_manifest.json`. Local evaluation-policy documents are
project-authored test fixtures.

