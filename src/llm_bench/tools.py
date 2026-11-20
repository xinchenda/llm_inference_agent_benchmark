from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_bench.config import PROJECT_ROOT
from llm_bench.json_utils import OutputFormatError, parse_json_object
from llm_bench.llm import LocalLLM
from llm_bench.retrieval import Hit, KnowledgeIndex


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_args: tuple[str, ...]
    function: Callable[..., Any]


@dataclass
class ToolCall:
    step: int
    tool: str
    arguments: dict[str, Any]
    status: str
    latency_s: float
    result: Any = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = self.result
        if isinstance(result, list):
            result = result[:3]
        return {
            "step": self.step,
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "latency_s": self.latency_s,
            "result": result,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec], timeout_s: float = 15) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.timeout_s = timeout_s

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        spec = self.tools[name]
        missing = [key for key in spec.required_args if key not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments for {name}: {missing}")
        unknown = sorted(set(arguments) - set(spec.required_args))
        if unknown:
            raise ValueError(f"Unknown arguments for {name}: {unknown}")
        return await asyncio.wait_for(
            asyncio.to_thread(spec.function, **arguments),
            timeout=self.timeout_s,
        )


class AgentTools:
    def __init__(
        self,
        index: KnowledgeIndex,
        models: dict[str, Any],
        retrieval_config: dict[str, Any],
        artifact_dir: Path,
    ) -> None:
        self.index = index
        self.models = models
        self.retrieval_config = retrieval_config
        self.artifact_dir = artifact_dir
        self._last_hits: list[Hit] = []

    def search_knowledge(self, query: str) -> list[dict[str, Any]]:
        from llm_bench.qa import retrieve

        hits = retrieve(self.index, query, "hybrid_rerank", self.retrieval_config)
        self._last_hits = hits
        return [
            {
                "document_id": hit.document_id,
                "title": hit.title,
                "section": hit.section,
                "text": hit.text[:1200],
                "source_url": hit.source_url,
                "license": hit.license,
            }
            for hit in hits
        ]

    def compare_models(self, model_ids: str) -> dict[str, Any]:
        requested = [item.strip() for item in model_ids.split(",") if item.strip()]
        if not requested:
            raise ValueError("At least one model id is required")
        selected: dict[str, Any] = {}
        for model_id in requested:
            if model_id not in self.models:
                raise ValueError(f"Unknown model id: {model_id}")
            item = dict(self.models[model_id])
            path = PROJECT_ROOT / item["file"]
            item["weight_bytes"] = path.stat().st_size if path.exists() else None
            selected[model_id] = item
        return selected

    def calculate_performance(
        self,
        operation: str,
        numerator: float,
        denominator: float,
        unit: str = "",
    ) -> dict[str, Any]:
        if denominator == 0:
            raise ValueError("denominator must be non-zero")
        if numerator < 0 or denominator < 0:
            raise ValueError("arguments must be non-negative")
        if operation == "tokens_per_second":
            value = numerator / denominator
        elif operation == "requests_per_second":
            value = 1000.0 / denominator
        elif operation == "ratio":
            value = numerator / denominator
        elif operation == "percent_reduction":
            value = (numerator - denominator) / numerator * 100.0
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        return {
            "operation": operation,
            "value": value,
            "rounded": round(value, 4),
            "unit": unit,
            "formula": f"{operation}({numerator}, {denominator})",
        }

    def generate_report(self, topic: str) -> dict[str, Any]:
        safe_topic = "".join(
            character if character.isalnum() else "-" for character in topic.lower()
        ).strip("-")[:60]
        digest = hashlib.sha1(topic.encode()).hexdigest()[:8]
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"agent-report-{safe_topic or 'topic'}-{digest}.md"
        sources = []
        for hit in self._last_hits[:5]:
            sources.append(
                f"- {hit.title} / {hit.section} (`{hit.document_id}`) "
                f"{hit.source_url or ''}".rstrip()
            )
        content = (
            f"# Agent report: {topic}\n\n"
            "## Summary\n\n"
            "This report was generated from the controlled local knowledge tool. "
            "Open questions and source limitations remain visible below.\n\n"
            "## Evidence\n\n"
            + ("\n".join(sources) if sources else "(No retrieved evidence.)")
            + "\n\n## Review status\n\n"
            "Automated generation; not yet approved for external distribution.\n"
        )
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "bytes": len(content.encode()), "sources": len(sources)}


def build_tool_registry(toolkit: AgentTools, timeout_s: float = 15) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="search_knowledge",
                description="Search the technical knowledge base",
                required_args=("query",),
                function=toolkit.search_knowledge,
            ),
            ToolSpec(
                name="compare_models",
                description="Compare exact model registry entries",
                required_args=("model_ids",),
                function=toolkit.compare_models,
            ),
            ToolSpec(
                name="calculate_performance",
                description="Calculate a transparent performance metric",
                required_args=("operation", "numerator", "denominator", "unit"),
                function=toolkit.calculate_performance,
            ),
            ToolSpec(
                name="generate_report",
                description="Write a Markdown report from prior evidence",
                required_args=("topic",),
                function=toolkit.generate_report,
            ),
        ],
        timeout_s=timeout_s,
    )


@dataclass
class AgentPlan:
    calls: list[tuple[str, dict[str, Any]]]
    answer_kind: str
    route_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentPlanningError(ValueError):
    """Raised when the LLM planner does not return a valid tool plan."""

    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


PLANNER_SYSTEM = """You are the planner for a bounded technical-document Agent.
Return exactly one JSON object and no prose. The schema is:
{"tool_calls":[{"tool":"tool_name","arguments":{...}}],"answer_kind":"grounded"}
Allowed answer_kind values: grounded, model_comparison, calculation, report.
Allowed tools and arguments:
- search_knowledge: {"query": string}
- compare_models: {"model_ids": "comma-separated ids"}
- calculate_performance: {"operation": string, "numerator": number,
  "denominator": number, "unit": string}
- generate_report: {"topic": string}; first call search_knowledge.
Use the minimum number of calls. Never invent tool names or arguments.
Examples:
Question: Compare model qwen3_0_6b_q8 with qwen3_1_7b_q4.
Output: {"tool_calls":[{"tool":"compare_models","arguments":
{"model_ids":"qwen3_0_6b_q8,qwen3_1_7b_q4"}}],"answer_kind":"model_comparison"}
Question: Calculate tokens per second for 1280 tokens in 8 seconds.
Output: {"tool_calls":[{"tool":"calculate_performance","arguments":
{"operation":"tokens_per_second","numerator":1280,"denominator":8,
"unit":"tokens/s"}}],"answer_kind":"calculation"}
Question: What does the timeout policy say?
Output: {"tool_calls":[{"tool":"search_knowledge","arguments":
{"query":"timeout policy"}}],"answer_kind":"grounded"}"""
PLANNER_ROUTES = {
    "grounded": ["search_knowledge"],
    "model_comparison": ["compare_models"],
    "calculation": ["calculate_performance"],
    "report": ["search_knowledge", "generate_report"],
}


def remove_articles(text: str) -> str:
    return " ".join(word for word in text.split() if word.lower() not in {"a", "an", "the"})


def plan_agent(question: str) -> AgentPlan:
    """Rule-based planner retained as an interpretable baseline."""
    lowered = question.lower()
    if "report" in lowered and any(
        marker in lowered for marker in ("generate", "create", "write", "draft")
    ):
        return AgentPlan(
            calls=[
                ("search_knowledge", {"query": question}),
                ("generate_report", {"topic": question}),
            ],
            answer_kind="report",
            route_reason="report_generation_pattern",
        )
    if (
        any(marker in lowered for marker in ("compare", "difference between"))
        and "model" in lowered
    ):
        candidates = []
        for model_id in (
            "qwen3_0_6b_q8",
            "qwen3_1_7b_q4",
            "qwen3_1_7b_q8",
        ):
            if model_id in lowered:
                candidates.append(model_id)
        return AgentPlan(
            calls=[("compare_models", {"model_ids": ",".join(candidates)})],
            answer_kind="model_comparison",
            route_reason="model_comparison_pattern",
            metadata={"model_ids": candidates},
        )
    if any(marker in lowered for marker in ("calculate", "compute")) and any(
        marker in lowered for marker in ("tokens per second", "requests per second", "ratio")
    ):
        numbers = [float(value) for value in re.findall(r"\b\d+(?:\.\d+)?\b", question)]
        if len(numbers) >= 2:
            if "requests per second" in lowered:
                operation = "requests_per_second"
                unit = "requests/s"
            elif "ratio" in lowered:
                operation = "ratio"
                unit = "ratio"
            else:
                operation = "tokens_per_second"
                unit = "tokens/s"
            return AgentPlan(
                calls=[
                    (
                        "calculate_performance",
                        {
                            "operation": operation,
                            "numerator": numbers[0],
                            "denominator": numbers[1],
                            "unit": unit,
                        },
                    )
                ],
                answer_kind="calculation",
                route_reason="calculation_pattern",
            )
    return AgentPlan(
        calls=[("search_knowledge", {"query": question})],
        answer_kind="grounded",
        route_reason="default_knowledge_route",
    )


def _planner_user_prompt(question: str) -> str:
    return f"""Question:
{question}

Return the JSON plan now."""


def validate_llm_plan(payload: dict[str, Any], max_tool_calls: int = 4) -> AgentPlan:
    allowed_tools = {
        "search_knowledge",
        "compare_models",
        "calculate_performance",
        "generate_report",
    }
    allowed_kinds = {"grounded", "model_comparison", "calculation", "report"}
    tool_calls = payload.get("tool_calls")
    answer_kind = payload.get("answer_kind")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise AgentPlanningError("tool_calls must be a non-empty list")
    if len(tool_calls) > max_tool_calls:
        raise AgentPlanningError(
            f"planner returned {len(tool_calls)} calls; max is {max_tool_calls}"
        )
    if answer_kind not in allowed_kinds:
        raise AgentPlanningError(f"invalid answer_kind: {answer_kind!r}")
    calls: list[tuple[str, dict[str, Any]]] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            raise AgentPlanningError(f"tool_calls[{index}] must be an object")
        name = call.get("tool")
        arguments = call.get("arguments")
        if name not in allowed_tools:
            raise AgentPlanningError(f"unknown planner tool: {name!r}")
        if not isinstance(arguments, dict):
            raise AgentPlanningError(f"arguments for {name} must be an object")
        calls.append((str(name), arguments))
    route_tools = [name for name, _ in calls]
    required_route = PLANNER_ROUTES[str(answer_kind)]
    if route_tools != required_route:
        raise AgentPlanningError(
            f"answer_kind {answer_kind!r} requires tools {required_route}, got {route_tools}"
        )
    return AgentPlan(
        calls=calls,
        answer_kind=str(answer_kind),
        route_reason="llm_json",
        metadata={"raw_plan": payload},
    )


def llm_plan_agent(
    llm: LocalLLM,
    question: str,
    max_tool_calls: int = 4,
    max_attempts: int = 2,
) -> AgentPlan:
    """Obtain a validated JSON tool plan, with one error-guided repair attempt."""
    attempts: list[dict[str, Any]] = []
    prompt = _planner_user_prompt(question)
    last_error: AgentPlanningError | None = None
    for attempt_number in range(1, max_attempts + 1):
        response = llm.generate(
            PLANNER_SYSTEM,
            prompt,
            max_tokens=260,
            disable_cache=True,
        )
        try:
            payload = parse_json_object(response.text)
            plan = validate_llm_plan(payload, max_tool_calls=max_tool_calls)
        except (OutputFormatError, AgentPlanningError) as exc:
            last_error = AgentPlanningError(str(exc), attempts=attempts)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "invalid",
                    "error": str(exc),
                    "raw_output": response.text[:800],
                    "latency_s": response.latency_s,
                }
            )
            if attempt_number < max_attempts:
                prompt = f"""Original question:
{question}

Your previous plan was invalid:
{response.text}

Validation error:
{exc}

Return one corrected JSON object using the required schema. No prose."""
            continue
        attempts.append(
            {
                "attempt": attempt_number,
                "status": "valid",
                "error": None,
                "raw_output": response.text[:800],
                "latency_s": response.latency_s,
            }
        )
        plan.metadata["planner_attempts"] = attempts
        plan.metadata["planner_repaired"] = attempt_number > 1
        return plan
    raise AgentPlanningError(
        str(last_error) if last_error else "planner produced no output",
        attempts=attempts,
    )


def deterministic_answer(kind: str, result: Any) -> str:
    if kind == "calculation":
        return f"Result: {result['rounded']} {result['unit']}. Formula: {result['formula']}."
    if kind == "model_comparison":
        lines = ["Model comparison from the local registry:"]
        for model_id, item in result.items():
            size = item.get("weight_bytes")
            size_text = f"{size / (1024**3):.2f} GiB" if size else "size unavailable"
            lines.append(
                f"- `{model_id}`: {item['display_name']}, {item['quantization']}, "
                f"{size_text}, license {item['license']}, context {item['context_length']}."
            )
        lines.append("Choose the smallest model that passes the task quality floor.")
        return "\n".join(lines)
    if kind == "report":
        return (
            f"Report generated at `{result['path']}` with {result['sources']} cited "
            "sources. It is marked for review and is not an approved external document."
        )
    return str(result)


def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]
