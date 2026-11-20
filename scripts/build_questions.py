#!/usr/bin/env python3
"""Build the fixed 100-question evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "questions" / "questions.jsonl"


def groups(*items: tuple[str, ...]) -> list[list[str]]:
    return [list(item) for item in items]


records: list[dict] = []


def add(
    question: str,
    category: str,
    expected_docs: list[str],
    gold_groups: list[list[str]] | None = None,
    expected_behavior: str = "answer",
    expected_tools: list[str] | None = None,
    human_review: bool = False,
) -> None:
    number = len(records) + 1
    records.append(
        {
            "id": f"q{number:03d}",
            "category": category,
            "question": question,
            "expected_docs": expected_docs,
            "gold_groups": gold_groups or [],
            "expected_behavior": expected_behavior,
            "expected_tools": expected_tools or [],
            "human_review": human_review,
        }
    )


# Normal fact questions: 30
add(
    "How are values read from environment variables typed and validated in FastAPI settings?",
    "factual",
    ["fastapi__advanced__settings"],
    groups(("str", "string", "text"), ("convert", "conversion"), ("validat",)),
    human_review=True,
)
add(
    "How does Pydantic Settings match environment variable names to field names?",
    "factual",
    ["fastapi__advanced__settings"],
    groups(("case-insensitive", "case insensitive"), ("env", "environment")),
)
add(
    "When does exit code in a FastAPI dependency with yield run?",
    "factual",
    ["fastapi__tutorial__dependencies__dependencies__with__yield"],
    groups(("after",), ("response",), ("yield", "exit")),
)
add(
    "When do FastAPI BackgroundTasks execute relative to the response?",
    "factual",
    ["fastapi__tutorial__background__tasks"],
    groups(("after",), ("response",)),
)
add(
    "What should be used for heavy background jobs that need multiple processes?",
    "factual",
    ["fastapi__tutorial__background__tasks"],
    groups(("Celery", "queue", "multiple processes"), ("memory", "same process")),
)
add(
    "What HTTP client behavior does FastAPI TestClient expose?",
    "factual",
    ["fastapi__tutorial__testing"],
    groups(("httpx",), ("TestClient",)),
)
add(
    "How does response_model affect output data?",
    "factual",
    ["fastapi__tutorial__response__model"],
    groups(("filter", "limit"), ("response", "output")),
)
add(
    "If both a return annotation and response_model are declared, which takes priority?",
    "factual",
    ["fastapi__tutorial__response__model"],
    groups(("response_model",), ("priority", "takes precedence")),
)
add(
    "Which CLI option starts multiple Uvicorn worker processes in FastAPI deployment?",
    "factual",
    ["fastapi__deployment__server__workers"],
    groups(("--workers", "workers"), ("process",)),
)
add(
    "In a Kubernetes-style deployment, should FastAPI normally use multiple "
    "Uvicorn workers per container?",
    "factual",
    ["fastapi__deployment__server__workers"],
    groups(("single", "one"), ("container",)),
)
add(
    "What file format is commonly used to store environment settings locally?",
    "factual",
    ["fastapi__advanced__settings"],
    groups(
        (".env", "dotenv"),
    ),
)
add(
    "Which executable is used by llama.cpp for an OpenAI-style function-calling server?",
    "factual",
    ["llamacpp__function__calling"],
    groups(("llama-server",), ("--jinja", "chat template")),
)
add(
    "What does llama.cpp's --n-gpu-layers option control?",
    "factual",
    ["llamacpp__development__token__generation__performance__tips"],
    groups(("gpu",), ("layer",), ("offload",)),
    human_review=True,
)
add(
    "How should llama.cpp thread count be tuned when token generation is slow?",
    "factual",
    ["llamacpp__development__token__generation__performance__tips"],
    groups(("threads",), ("physical cores", "number of cores"), ("not too large", "oversatur")),
)
add(
    "What is the role of a draft model in llama.cpp speculative decoding?",
    "factual",
    ["llamacpp__speculative"],
    groups(("draft",), ("verify", "verification"), ("target model", "main model")),
)
add(
    "Which Apple GPU backend does llama.cpp list as supported?",
    "factual",
    ["llamacpp__install"],
    groups(("Metal",), ("Apple", "macOS")),
)
add(
    "What does vLLM say about GGUF support and its main benefit?",
    "factual",
    ["vllm__features__quantization__gguf"],
    groups(("experimental",), ("memory", "footprint"), ("plugin",)),
)
add(
    "What problem does vLLM PagedAttention address?",
    "factual",
    ["vllm__design__paged__attention"],
    groups(("KV", "key-value"), ("memory", "cache"), ("paged", "page")),
)
add(
    "What does vLLM prefix caching reuse?",
    "factual",
    ["vllm__design__prefix__caching"],
    groups(("prefix",), ("KV", "cache"), ("reuse", "cache")),
)
add(
    "Does vLLM guarantee reproducibility by default?",
    "factual",
    ["vllm__usage__reproducibility"],
    groups(("no", "not", "does not"), ("reproduc",), ("performance", "scheduling")),
)
add(
    "What must also match for vLLM to provide reproducible output even after "
    "enabling deterministic scheduling?",
    "factual",
    ["vllm__usage__reproducibility"],
    groups(("same hardware", "hardware"), ("version",)),
)
add(
    "What does vLLM's seed parameter control?",
    "factual",
    ["vllm__usage__reproducibility"],
    groups(("random",), ("random", "seed"), ("torch", "numpy", "state")),
)
add(
    "What happens to parallel tool calls in vLLM when parallel_tool_calls is false?",
    "factual",
    ["vllm__serving__online__serving__openai__compatible__server"],
    groups(("zero", "one"), ("tool call", "call")),
)
add(
    "What does vLLM's tool_choice='required' request from the model?",
    "factual",
    ["vllm__features__tool__calling"],
    groups(("at least one", "one or more"), ("tool call", "call")),
)
add(
    "Which endpoint exposes Prometheus-compatible vLLM metrics?",
    "factual",
    ["vllm__usage__metrics"],
    groups(("/metrics", "metrics"), ("Prometheus",)),
)
add(
    "What tradeoff does vLLM describe for quantized models when conserving memory?",
    "factual",
    ["vllm__configuration__conserving__memory"],
    groups(("less memory", "reduce memory", "lower memory"), ("precision",)),
)
add(
    "Which vLLM option can limit context length to save memory?",
    "factual",
    ["vllm__configuration__conserving__memory"],
    groups(("max_model_len",), ("context",), ("memory",)),
)
add(
    "What does vLLM's offline LLM.get_metrics return?",
    "factual",
    ["vllm__serving__offline__inference", "vllm__usage__metrics"],
    groups(("metrics",), ("Prometheus",)),
)
add(
    "What should an application do after a model emits a tool call according "
    "to vLLM's tool-calling guide?",
    "factual",
    ["vllm__features__tool__calling"],
    groups(("application", "logic", "handle"), ("tool call", "execute")),
)
add(
    "Why should a pilot require a model revision and SHA256 in the local model policy?",
    "factual",
    ["local__model_comparison_policy"],
    groups(("revision",), ("SHA256", "hash"), ("reproduc", "evidence", "trace")),
)

# Long-context and cross-document questions: 15
add(
    "Using the agent release policy and conflict note, what is the current "
    "Atlas context window and how much is reserved for the answer and tool metadata?",
    "long_context",
    ["local__agent_release_policy", "local__agent_context_conflict"],
    groups(("8192",), ("512",), ("256",)),
    human_review=True,
)
add(
    "What data is approved for the Atlas 2.1 evaluation release, and what data must not be used?",
    "long_context",
    ["local__agent_release_policy"],
    groups(("technical documents", "synthetic"), ("customer personal data", "personal data")),
)
add(
    "What is the Atlas timeout and duplicate-tool-call policy?",
    "long_context",
    ["local__agent_release_policy"],
    groups(("twice", "two"), ("stop", "partial"), ("duplicate", "identical")),
)
add(
    "How do the fixed evaluation protocol and model selection policy constrain a model comparison?",
    "long_context",
    ["local__evaluation_protocol", "local__model_comparison_policy"],
    groups(
        ("same 100 questions", "fixed evaluation"),
        ("memory", "latency"),
        ("license", "SHA256"),
    ),
    human_review=True,
)
add(
    "How should a pilot balance retrieval context length against memory and answer quality?",
    "long_context",
    ["vllm__configuration__conserving__memory", "local__evaluation_protocol"],
    groups(("context",), ("memory",), ("fixed conditions", "quality", "same")),
)
add(
    "Compare a controlled application tool flow with letting a model emit "
    "arbitrary function calls.",
    "long_context",
    ["vllm__features__tool__calling", "local__agent_release_policy"],
    groups(("application", "controlled"), ("validate", "schema", "failure"), ("tool",)),
)
add(
    "How should settings and credentials be handled in the local FastAPI service?",
    "long_context",
    ["fastapi__advanced__settings", "local__security_boundary"],
    groups(("environment", "env var"), ("bearer", "token"), ("secret", "identity provider")),
)
add(
    "Which serving features distinguish vLLM's GPU-oriented server from llama.cpp on this Mac?",
    "long_context",
    ["vllm__serving__online__serving__openai__compatible__server", "llamacpp__install"],
    groups(("OpenAI", "compatible"), ("GPU", "Metal"), ("vLLM",), ("llama.cpp",)),
)
add(
    "What conditions are needed to reproduce a vLLM result and an evaluation "
    "result in this project?",
    "long_context",
    ["vllm__usage__reproducibility", "local__evaluation_protocol"],
    groups(("seed",), ("hardware",), ("version",), ("temperature 0", "temperature=0")),
)
add(
    "How can an operator monitor inference requests and memory pressure?",
    "long_context",
    ["vllm__usage__metrics", "vllm__configuration__conserving__memory"],
    groups(("Prometheus", "/metrics"), ("memory",), ("request", "histogram")),
)
add(
    "What combination of vLLM settings can reduce memory without changing model parameters?",
    "long_context",
    ["vllm__configuration__conserving__memory"],
    groups(("quantiz",), ("max_model_len", "context"), ("memory",)),
)
add(
    "Why can a smaller quantized model be preferable even when its quality is slightly lower?",
    "long_context",
    ["local__model_comparison_policy", "vllm__configuration__conserving__memory"],
    groups(("quality floor", "passes"), ("memory",), ("latency", "cost")),
)
add(
    "When should FastAPI BackgroundTasks be replaced by a queue-based worker system?",
    "long_context",
    ["fastapi__tutorial__background__tasks"],
    groups(
        ("heavy", "long"),
        ("multiple processes", "same process", "queue"),
        ("memory", "variables"),
    ),
)
add(
    "What evidence should be collected before reporting that a model is better than another?",
    "long_context",
    ["local__model_comparison_policy", "local__evaluation_protocol"],
    groups(
        ("quality",),
        ("latency", "memory"),
        ("failure", "p95"),
        ("SHA256", "license", "revision"),
    ),
)
add(
    "What security limits apply when the agent generates a report?",
    "long_context",
    ["local__security_boundary"],
    groups(("artifact", "directory"), ("shell",), ("filesystem", "access")),
)

# No-answer questions: 15
for question in [
    "What exact date was Atlas 3.0 released?",
    "Who personally approved the Atlas 2.1 deployment?",
    "What is the measured wall-power consumption of this Mac during inference?",
    "What is the company's final customer price for one agent request?",
    "What is the exact number of chunks in the current index?",
    "Which private employee owns the Atlas service?",
    "How many training tokens were used to train Qwen3?",
    "What is the guaranteed annual uptime SLA for this local pilot?",
    "What latency did vLLM achieve on an NVIDIA H100 in this project?",
    "What is the production bearer token value?",
    "What personal email address belongs to the service owner?",
    "What is the company valuation of the Atlas project team?",
    "How does Qwen3 score against GPT on an undisclosed private benchmark?",
    "What exact internal function name does the report-generation tool invoke to create a file?",
    "What is the exact electricity tariff paid at the deployment site?",
]:
    add(question, "unanswerable", [], [], expected_behavior="no_answer")

# False-premise questions: 10
add(
    "Are FastAPI environment variables already typed as integers without conversion or validation?",
    "false_premise",
    ["fastapi__advanced__settings"],
    groups(("no", "not"), ("str", "string", "text"), ("convert", "validat")),
)
add(
    "Does llama.cpp require an NVIDIA CUDA GPU to run at all?",
    "false_premise",
    ["llamacpp__install"],
    groups(("no", "not", "does not"), ("Metal", "CPU", "Apple")),
    human_review=True,
)
add(
    "Does vLLM guarantee identical outputs by default regardless of scheduling?",
    "false_premise",
    ["vllm__usage__reproducibility"],
    groups(("no", "not", "does not"), ("reproduc",), ("scheduling", "performance")),
)
add(
    "Does vLLM PagedAttention reduce the number of model parameters?",
    "false_premise",
    ["vllm__design__paged__attention"],
    groups(("no", "not"), ("KV", "cache"), ("parameter",)),
)
add(
    "Do FastAPI BackgroundTasks run before the HTTP response is sent?",
    "false_premise",
    ["fastapi__tutorial__background__tasks"],
    groups(("no", "after"), ("response",)),
)
add(
    "Does prefix caching change the model's trained weights?",
    "false_premise",
    ["vllm__design__prefix__caching"],
    groups(("no", "not"), ("cache", "KV"), ("weight",)),
    human_review=True,
)
add(
    "Is GGUF itself a model fine-tuning algorithm?",
    "false_premise",
    ["vllm__features__quantization__gguf", "llamacpp__install"],
    groups(
        ("no", "not"),
        ("format", "quantiz"),
    ),
)
add(
    "Is the Atlas 2.1 evaluation release approved for customer personal data?",
    "false_premise",
    ["local__agent_release_policy"],
    groups(("no", "not"), ("customer personal data", "personal data")),
)
add(
    "Can the agent's report tool run unrestricted shell commands?",
    "false_premise",
    ["local__security_boundary"],
    groups(("no", "not"), ("shell",), ("not exposed", "unavailable", "not supported")),
)
add(
    "After a tool times out twice, should the agent keep retrying identical calls?",
    "false_premise",
    ["local__agent_release_policy"],
    groups(("no", "stop"), ("twice", "two"), ("duplicate", "partial")),
)

# Conflicting evidence questions: 10
add(
    "Two Atlas notes give 4096 and 8192 tokens. What should be reported as "
    "current, and how should the conflict be described?",
    "conflict",
    ["local__agent_context_conflict", "local__legacy_notes"],
    groups(("8192",), ("4096",), ("legacy", "obsolete", "1.8")),
    human_review=True,
)
add(
    "A legacy document says the context window is 4096. Is it safe to call "
    "that the release 2.1 limit?",
    "conflict",
    ["local__agent_context_conflict", "local__legacy_notes"],
    groups(("no", "not"), ("4096",), ("legacy", "1.8"), ("8192",)),
)
add(
    "How should an answer handle the conflicting 4096 versus 8192 Atlas context values?",
    "conflict",
    ["local__agent_context_conflict", "local__legacy_notes"],
    groups(("both", "4096", "8192"), ("8192",), ("4096",), ("newer", "current", "legacy")),
)
add(
    "The old prototype put an API key in the URL, while current security "
    "policy says bearer token. Which should a pilot use?",
    "conflict",
    ["local__legacy_notes", "local__security_boundary"],
    groups(("bearer",), ("environment", "env var"), ("query string", "url")),
    human_review=True,
)
add(
    "Should the legacy query-string credential pattern override current authentication policy?",
    "conflict",
    ["local__legacy_notes", "local__security_boundary"],
    groups(("no", "not"), ("bearer",), ("legacy", "current")),
)
add(
    "One note accepts vendor benchmarks alone, while another requires local "
    "evidence. What does current policy require?",
    "conflict",
    ["local__legacy_notes", "local__model_comparison_policy"],
    groups(("local",), ("quality", "latency", "memory"), ("license", "SHA256")),
)
add(
    "Is a vendor benchmark claim enough under the current model selection policy?",
    "conflict",
    ["local__legacy_notes", "local__model_comparison_policy"],
    groups(("no", "not"), ("local",), ("evidence", "measured", "quality")),
    human_review=True,
)
add(
    "The old checklist omitted human review, but another protocol requires "
    "it. What is the current evaluation requirement?",
    "conflict",
    ["local__legacy_notes", "local__evaluation_protocol"],
    groups(("twelve", "12"), ("manual", "human"), ("review",)),
)
add(
    "How should obsolete engineering notes be treated when they conflict with current policy?",
    "conflict",
    ["local__legacy_notes", "local__agent_context_conflict"],
    groups(("obsolete", "legacy"), ("current", "newer"), ("authoritative", "override")),
)
add(
    "Summarize the current-versus-legacy conflict on credentials and explain which document wins.",
    "conflict",
    ["local__legacy_notes", "local__security_boundary"],
    groups(("query string", "url"), ("bearer",), ("current", "legacy")),
)

# Tool-call questions: 20
comparisons = [
    (
        "Compare model qwen3_0_6b_q8 with qwen3_1_7b_q4 using the local model registry.",
        "qwen3_0_6b_q8",
    ),
    (
        "Compare model qwen3_1_7b_q4 with qwen3_1_7b_q8 using the local model registry.",
        "qwen3_1_7b_q8",
    ),
    (
        "What is the difference between model qwen3_0_6b_q8 and model qwen3_1_7b_q4?",
        "qwen3_1_7b_q4",
    ),
    (
        "Compare the licenses and contexts of model qwen3_1_7b_q4 and model qwen3_1_7b_q8.",
        "qwen3_1_7b_q8",
    ),
    (
        "Which of model qwen3_0_6b_q8 and model qwen3_1_7b_q4 is smaller by parameter count?",
        "0.6",
    ),
    ("Compare model qwen3_0_6b_q8 and model qwen3_1_7b_q8.", "qwen3_0_6b_q8"),
    (
        "Using the model registry, compare model qwen3_1_7b_q4 with model qwen3_1_7b_q8.",
        "Q4_K_M",
    ),
    (
        "What is the difference between model qwen3_0_6b_q8 and model qwen3_1_7b_q8 "
        "in quantization?",
        "Q8_0",
    ),
    ("Compare model qwen3_1_7b_q4 and model qwen3_1_7b_q8 on weight size.", "GiB"),
    (
        "Compare the parameter counts of model qwen3_0_6b_q8 and model qwen3_1_7b_q4.",
        "1.7",
    ),
]
for question, marker in comparisons:
    add(
        question,
        "tool_calls",
        [],
        groups(("model",), ("qwen3",), (marker.lower(),)),
        expected_tools=["compare_models"],
        human_review=question.startswith("Compare model qwen3_0_6b"),
    )

calculations = [
    ("Calculate tokens per second for 1280 tokens in 8 seconds.", 160.0),
    ("Calculate tokens per second for 900 tokens in 6 seconds.", 150.0),
    ("Calculate tokens per second for 1000 tokens in 4 seconds.", 250.0),
    ("Calculate requests per second for a latency of 40 milliseconds.", 25.0),
    ("Calculate requests per second for a latency of 20 milliseconds.", 50.0),
    ("Calculate the ratio of 512 to 256.", 2.0),
    ("Calculate the ratio of 1200 to 300.", 4.0),
    ("Calculate tokens per second for 640 tokens in 4 seconds.", 160.0),
]
for index, (question, expected) in enumerate(calculations):
    add(
        question,
        "tool_calls",
        [],
        groups((str(expected),)),
        expected_tools=["calculate_performance"],
        human_review=index == 0,
    )

add(
    "Generate a report about deployment security using the knowledge tool first.",
    "tool_calls",
    ["local__security_boundary"],
    groups(("report",), ("path", "generated")),
    expected_tools=["search_knowledge", "generate_report"],
)
add(
    "Create a report summarizing the evaluation protocol.",
    "tool_calls",
    ["local__evaluation_protocol"],
    groups(("report",), ("review", "generated")),
    expected_tools=["search_knowledge", "generate_report"],
)

assert len(records) == 100, len(records)
assert len({record["id"] for record in records}) == 100

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)
print(f"Wrote {len(records)} questions to {OUT}")
