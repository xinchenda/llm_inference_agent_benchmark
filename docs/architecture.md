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

