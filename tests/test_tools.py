from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from llm_bench.tools import ToolRegistry, ToolSpec


@pytest.mark.asyncio
async def test_tool_timeout_is_reported() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="slow",
                description="slow tool",
                required_args=("value",),
                function=lambda value: time.sleep(0.1) or value,
            )
        ],
        timeout_s=0.01,
    )
    with pytest.raises(TimeoutError):
        await registry.call("slow", {"value": 1})


@pytest.mark.asyncio
async def test_illegal_tool_arguments_are_rejected() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="echo",
                description="echo",
                required_args=("value",),
                function=lambda value: value,
            )
        ]
    )
    with pytest.raises(ValueError, match="Missing required"):
        await registry.call("echo", {})
    with pytest.raises(ValueError, match="Unknown arguments"):
        await registry.call("echo", {"value": 1, "extra": 2})


@pytest.mark.asyncio
async def test_duplicate_agent_calls_are_rejected(toy_index, fake_llm) -> None:
    from llm_bench.agent import ControlledAgent
    from llm_bench.tools import plan_agent

    agent = ControlledAgent(
        llm=fake_llm,
        index=toy_index,
        models={},
        retrieval_config={
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        artifact_dir=Path("/tmp/agent-reports"),
        model_id="fake",
        planner_mode="rule",
    )
    plan = plan_agent("What is the timeout policy?")
    plan.calls = [plan.calls[0], plan.calls[0]]
    original_plan = __import__("llm_bench.agent", fromlist=["plan_agent"]).plan_agent
    try:
        __import__("llm_bench.agent", fromlist=["plan_agent"]).plan_agent = lambda _: plan
        result = await agent.run("What is the timeout policy?")
    finally:
        __import__("llm_bench.agent", fromlist=["plan_agent"]).plan_agent = original_plan
    statuses = [item["status"] for item in result.tool_trace if item["tool"] != "planner"]
    assert statuses == ["ok", "skipped"]


@pytest.mark.asyncio
async def test_no_retrieval_result_is_explicit(toy_index, fake_llm) -> None:
    from llm_bench.llm import LLMResponse
    from llm_bench.qa import grounded_from_tool

    class RefusalLLM:
        def generate(self, *_, **__):
            return LLMResponse(
                text="I cannot determine this from the provided evidence.",
                latency_s=0.01,
                ttft_s=0.005,
                prompt_tokens=5,
                completion_tokens=8,
                timings={},
            )

    result = await asyncio.to_thread(
        grounded_from_tool,
        RefusalLLM(),
        "Unknown question",
        "fake",
        [],
    )
    assert "cannot determine" in result.answer
    assert result.retrieved == []


def test_llm_planner_uses_json_tool_call(toy_index) -> None:
    from llm_bench.llm import LLMResponse
    from llm_bench.tools import llm_plan_agent

    class PlannerLLM:
        def generate(self, *_, **__):
            return LLMResponse(
                text=(
                    '{"tool_calls":[{"tool":"search_knowledge",'
                    '"arguments":{"query":"timeout policy"}}],'
                    '"answer_kind":"grounded"}'
                ),
                latency_s=0.02,
                ttft_s=0.01,
                prompt_tokens=30,
                completion_tokens=20,
                timings={},
            )

    plan = llm_plan_agent(PlannerLLM(), "What is the timeout policy?")
    assert plan.route_reason == "llm_json"
    assert plan.calls == [("search_knowledge", {"query": "timeout policy"})]
    assert plan.answer_kind == "grounded"


def test_llm_planner_rejects_malformed_json() -> None:
    from llm_bench.llm import LLMResponse
    from llm_bench.tools import AgentPlanningError, llm_plan_agent

    class BrokenPlanner:
        def generate(self, *_, **__):
            return LLMResponse(
                text='{"tool_calls": [',
                latency_s=0.01,
                ttft_s=0.005,
                prompt_tokens=10,
                completion_tokens=4,
                timings={},
            )

    with pytest.raises(AgentPlanningError, match="Invalid JSON"):
        llm_plan_agent(BrokenPlanner(), "question")


def test_llm_planner_repairs_one_invalid_json_output() -> None:
    from llm_bench.llm import LLMResponse
    from llm_bench.tools import llm_plan_agent

    class RepairPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *_, **__):
            self.calls += 1
            text = (
                '{"tool_calls": ['
                if self.calls == 1
                else (
                    '{"tool_calls":[{"tool":"search_knowledge",'
                    '"arguments":{"query":"timeout policy"}}],'
                    '"answer_kind":"grounded"}'
                )
            )
            return LLMResponse(
                text=text,
                latency_s=0.01,
                ttft_s=0.005,
                prompt_tokens=10,
                completion_tokens=8,
                timings={},
            )

    planner = RepairPlanner()
    plan = llm_plan_agent(planner, "What is the timeout policy?")
    assert planner.calls == 2
    assert plan.metadata["planner_repaired"] is True
    assert [item["status"] for item in plan.metadata["planner_attempts"]] == [
        "invalid",
        "valid",
    ]


def test_llm_planner_rejects_inconsistent_route() -> None:
    from llm_bench.tools import AgentPlanningError, validate_llm_plan

    with pytest.raises(AgentPlanningError, match="requires tools"):
        validate_llm_plan(
            {
                "tool_calls": [
                    {
                        "tool": "search_knowledge",
                        "arguments": {"query": "model comparison"},
                    }
                ],
                "answer_kind": "model_comparison",
            }
        )


def test_compare_models_resolves_project_relative_paths(toy_index, tmp_path, monkeypatch) -> None:
    from llm_bench.tools import AgentTools

    monkeypatch.setattr("llm_bench.tools.PROJECT_ROOT", tmp_path)
    model_path = tmp_path / "models" / "test.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"weights")
    toolkit = AgentTools(
        index=toy_index,
        models={
            "model-a": {
                "file": "models/test.gguf",
                "display_name": "Test",
                "quantization": "Q4",
                "context_length": 1024,
                "license": "MIT",
            }
        },
        retrieval_config={
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        artifact_dir=tmp_path / "artifacts",
    )
    result = toolkit.compare_models("model-a")
    assert result["model-a"]["weight_bytes"] == 7
