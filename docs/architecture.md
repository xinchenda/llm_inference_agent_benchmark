# Architecture and failure handling

## Boundaries

The project separates five replaceable concerns:

1. `llm.py`: model process lifecycle and streaming OpenAI-compatible requests.
2. `retrieval.py`: corpus chunking, sparse/dense retrieval, fusion, reranking.
3. `qa.py`: prompt construction, evidence labels, citations, answer recording.
4. `tools.py` and `agent.py`: typed tools, an LLM-generated JSON plan, and a
   bounded executor. The rule planner is retained only as a baseline.
5. `service.py`: HTTP validation, auth, rate limiting, idempotency, request
   tracing, and metrics.

The data flow is one direction: source document -> chunk -> retrieval hit ->
numbered evidence -> answer citation -> evaluation record. Tool calls use the
same evidence store; report generation only writes under the configured
artifact directory.

## Experiment optimization control plane

The P0/P1 optimizer is separate from the document Agent. It optimizes serving
configurations through a deterministic control loop:

```text
ExperimentSpec
  -> one legal candidate selected by default/grid/random/agent policy
  -> exclusive resource lock and unused local port
  -> llama-server start and health check
  -> warmup
  -> deterministic quality gate
  -> repeated closed-loop or fixed-QPS workload
  -> TrialResult with provenance and metrics
  -> append-only SQLite ledger
  -> Pareto, stability, best-so-far, and similar-history report
```

Module boundaries under `src/llm_bench/experiment/`:

| Module | Responsibility |
| --- | --- |
| `types.py` | Frozen specifications, canonical hashing, and validation |
| `ledger.py` | SQLite append-only records, duplicate lookup, and atomic batch insert |
| `runtime.py` | Resource lock, port availability, failure classification, process-tree cleanup |
| `quality.py` | Exact answer, required fact, JSON schema value, and timeout gates |
| `load.py` | Closed-loop and fixed-QPS open-loop scheduling |
| `metrics.py` | Percentiles, TPOT/ITL, goodput, stability, and bootstrap intervals |
| `policies.py` | Four equal-budget candidate-selection strategies |
| `runner.py` | Trial orchestration, retries, artifacts, and ledger persistence |
| `analysis.py` | Pareto, best-so-far, strategy comparison, and similar history |
| `report.py` | CSV, JSON, Markdown, and figure generation |

The document Agent may generate a validated tool plan. The experiment Agent is
more constrained: it selects one ID from a frozen candidate list based on
structured ledger history. Numeric validity, quality judgments, request counts,
SLOs, and candidate legality remain deterministic code.

### Fixed loads and quality ordering

Every trial uses the definitions in `configs/experiments_p0_p1.yaml`:

- `interactive`: short prompt, short output, closed loop.
- `rag_reading`: fixed long evidence context, short output, fixed 2 QPS.
- `batch_generation`: concurrent batch requests with longer output.
- `prefix_cache_rag`: one long document prefix with different final questions.

The quality gate runs before scored performance. A failed exact answer, missing
critical fact, malformed JSON, timeout, or a workload with no successful SLO
response prevents the trial from entering strategy ranking.

`--parallel` divides the server context among slots. Candidate validation
therefore requires enough total context per configured slot; otherwise a long
request can receive HTTP 400 despite a nonzero server context. The failed first
development run exposed this issue and is retained outside the final ledger as
engineering evidence.

Serving candidates also declare:

- `batch_size` and `ubatch_size`, passed to `llama-server`; batch size must be
  divisible by ubatch size.
- `flash_attn=on|off|auto`.
- K/V cache types. The controlled space is `f16` and `q8_0`; any quantized cache
  requires Flash Attention.
- Request-level `cache_prompt=true|false`.

The request client sends the cache preference through the llama.cpp compatible
`cache_prompt` field. A separate mandatory-`false` path remains for callers that
explicitly disable prompt reuse.

### Immutable evidence and failures

`ExperimentLedger` stores canonical specifications and trial results in SQLite.
Database triggers reject updates and deletes. Exact duplicate hashes are
idempotent, and multi-row writes roll back as one transaction.

Every trial record includes git commit, experiment hash, configuration hash,
model hash, runtime and hardware fingerprints, workload IDs, quality result,
performance metrics, failure class, baseline deltas, and local artifact path.

Failure handling:

- Startup, health, port, timeout, OOM, crash, invalid configuration, quality
  degradation, and request failure use stable categories.
- Out-of-memory and invalid configuration do not retry.
- Transient startup, timeout, or crash paths permit at most one controlled retry.
- The complete model process tree is terminated after success or failure.
- Partial request and service logs remain available when cleanup succeeds.

### Statistics and equal-budget strategies

Each strategy receives the same number of scored trials and the same repeated
workloads. A common baseline anchor sits outside the strategy budget and
provides baseline deltas.

- `default`: repeat the declared baseline.
- `grid`: deterministic candidate traversal.
- `random`: fixed-seed candidate selection.
- `ucb1`: baseline UCB that first explores every untested legal candidate.
- `agent`: RSI-style Reflection/Memory loop.

For the RSI Agent, each completed or failed trial produces structured memory:

- TTFT bottleneck.
- Throughput/queueing bottleneck.
- RSS memory pressure.
- Quality-gate or runtime failure.
- Legal candidate recommendations and a negative-experience flag.

Every untested legal candidate receives an exploration opportunity before the
Agent repeats a candidate, unless the trial budget ends earlier. A quality-gate
failure triggers deterministic rollback to the best prior quality-passing legal
candidate. After exploration, UCB scores are adjusted by negative memory, and
the deterministic tie-break remains the registered candidate order.

Per-request metrics include TTFT, TPOT/ITL, end-to-end latency, completion token
count, and SLO status. Per-workload summaries include p50/p95/p99 latency, output
tokens/s, achieved arrival rate, goodput, process-tree RSS, repeat standard
deviation, coefficient of variation, and percentile-bootstrap intervals.

Pareto objectives maximize goodput and minimize p95 latency and peak process
memory. Only quality-passing trials are eligible. Best-so-far curves use the
measured strategy sequence, and similar-history lookup uses ledger values rather
than synthetic records.

### Prefix Cache evidence

The shared-prefix workload repeats one long document and changes only the final
question. This makes prompt-cache reuse auditable rather than inferred from two
unrelated requests.

For every successful request:

- Read `timings.cache_n` from the llama.cpp response.
- Store `cache_n`, request `prompt_tokens`, and `cache_n / prompt_tokens`.
- Mark missing, null, or non-numeric `cache_n` as invalid; never estimate it.
- Aggregate the hit rate as
  `sum(valid cache_n) / sum(valid prompt_tokens)`.
- Preserve error rate and missing-observation counts.

The dedicated `configs/prefix_cache_ablation.yaml` matrix compares f16 and q8_0
KV caches with `cache_prompt` on and off. The report distinguishes request-level
TTFT reduction from actual `cache_n` evidence.

## Agent state machine

The default planner asks Qwen3 for exactly one JSON object describing tool
calls. The plan is parsed and validated before any tool executes:

- Malformed JSON, unknown tools, non-object arguments, and invalid
  `answer_kind` values are rejected.
- A rejected plan receives exactly one repair request containing the invalid
  output and validation error. Both raw attempts, statuses, and the accepted
  plan are retained in the planner trace.
- Tool sequences are constrained by route: model comparison -> `compare_models`;
  calculation -> `calculate_performance`; grounded answer -> `search_knowledge`;
  report -> `search_knowledge` followed by `generate_report`.
- The rule planner (`configs/default.yaml: agent_rule_rerank`) remains an
  independent baseline for comparing planner behavior.

Each call passes through a typed registry. The registry rejects unknown tools,
missing arguments, and extra arguments. `asyncio.wait_for` enforces a timeout.
The executor rejects identical calls after the first one and has independent
step and tool-call limits. Every call records tool name, arguments, elapsed time,
status, summarized result, and error. A `planner` entry records whether the
LLM plan succeeded, failed validation, or timed out.

## Failure policy

| Failure | Behavior | Visible evidence |
|---|---|---|
| No retrieval evidence | Answer explicitly that evidence is absent | Empty hits and no citations in result |
| Tool timeout | Stop that call, mark `timeout`, return partial result | Tool trace |
| Invalid arguments | Reject before function execution | Trace error |
| Duplicate call | Skip repeated signature | `skipped` entry |
| Model JSON damaged | One error-guided repair; fail with trace if still invalid | Planner attempt trace / 502 |
| `/ask` or `/agent` exceeds request budget | Return 504 | Structured error and request log |
| Agent planner or tool execution fails | Return 502 with partial result/trace | Structured error detail |
| HTTP model failure in `/ask` | Return 502 | Structured error and request log |
| Repeated client request | Replay cached response for same `Idempotency-Key` | `x-idempotent-replay: true` |
| Rate limit | Return 429 | Process-local limiter counter |

## Production extension points

- Replace the local HTTP limiter with Redis or the platform API gateway.
- Replace static bearer auth with the company identity provider.
- Add a vLLM/SGLang backend implementing the same `LocalLLM.generate`
  interface on Linux/NVIDIA.
- Store corpus and evaluation revisions in an experiment registry.
- Export logs and Prometheus metrics to the deployment observability stack.
- Add human adjudication UI for the retained badcase queue.
- Replace the local SQLite/file lock with a shared experiment service for
  multi-host scheduling.
- Add GPU memory and utilization counters when a Linux/NVIDIA backend is actually
  available; they are not inferred from the Mac RSS result.

