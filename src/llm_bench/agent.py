from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from llm_bench.llm import LocalLLM
from llm_bench.qa import AnswerResult, grounded_from_tool
from llm_bench.retrieval import Hit, KnowledgeIndex
from llm_bench.tools import (
    AgentTools,
    ToolCall,
    build_tool_registry,
    deterministic_answer,
    llm_plan_agent,
    plan_agent,
)


class ControlledAgent:
    def __init__(
        self,
        llm: LocalLLM,
        index: KnowledgeIndex,
        models: dict[str, Any],
        retrieval_config: dict[str, Any],
        artifact_dir: Path,
        model_id: str,
        max_steps: int = 6,
        max_tool_calls: int = 4,
        tool_timeout_s: float = 15,
        planner_mode: str = "llm",
        planner_timeout_s: float | None = None,
    ) -> None:
        self.llm = llm
        self.index = index
        self.model_id = model_id
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.planner_mode = planner_mode
        self.planner_timeout_s = planner_timeout_s or tool_timeout_s
        self.toolkit = AgentTools(
            index=index,
            models=models,
            retrieval_config=retrieval_config,
            artifact_dir=artifact_dir,
        )
        self.registry = build_tool_registry(self.toolkit, tool_timeout_s)

    async def run(self, question: str) -> AnswerResult:
        trace: list[dict[str, Any]] = []
        planner_started = time.perf_counter()
        try:
            if self.planner_mode == "rule":
                plan = plan_agent(question)
                planner_status = "ok"
                planner_error = None
            elif self.planner_mode == "llm":
                plan = await asyncio.wait_for(
                    asyncio.to_thread(
                        llm_plan_agent,
                        self.llm,
                        question,
                        self.max_tool_calls,
                    ),
                    timeout=self.planner_timeout_s,
                )
                planner_status = "ok"
                planner_error = None
            else:
                raise ValueError(f"Unknown planner mode: {self.planner_mode}")
        except Exception as exc:
            trace.append(
                {
                    "step": 0,
                    "tool": "planner",
                    "arguments": {"mode": self.planner_mode},
                    "status": "timeout" if isinstance(exc, TimeoutError) else "error",
                    "latency_s": time.perf_counter() - planner_started,
                    "result": {
                        "planner_attempts": getattr(exc, "attempts", []),
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return AnswerResult(
                answer="",
                mode="agent",
                model_id=self.model_id,
                tool_trace=trace,
                error=f"Agent planning failed: {type(exc).__name__}: {exc}",
            )
        trace.append(
            {
                "step": 0,
                "tool": "planner",
                "arguments": {"mode": self.planner_mode},
                "status": planner_status,
                "latency_s": time.perf_counter() - planner_started,
                "result": {
                    "calls": plan.calls,
                    "answer_kind": plan.answer_kind,
                    "route_reason": plan.route_reason,
                    "planner_attempts": plan.metadata.get("planner_attempts", []),
                    "planner_repaired": plan.metadata.get("planner_repaired", False),
                },
                "error": planner_error,
            }
        )
        observations: list[Any] = []
        seen_calls: set[tuple[str, str]] = set()
        for step, (tool_name, arguments) in enumerate(plan.calls, 1):
            if step > self.max_steps or step > self.max_tool_calls:
                trace.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "arguments": arguments,
                        "status": "skipped",
                        "latency_s": 0.0,
                        "error": "Agent step limit reached",
                    }
                )
                break
            signature = (tool_name, repr(sorted(arguments.items())))
            if signature in seen_calls:
                trace.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "arguments": arguments,
                        "status": "skipped",
                        "latency_s": 0.0,
                        "error": "Duplicate tool call rejected",
                    }
                )
                continue
            seen_calls.add(signature)
            call = ToolCall(
                step=step,
                tool=tool_name,
                arguments=arguments,
                status="running",
                latency_s=0.0,
            )
            started = time.perf_counter()
            try:
                call.result = await self.registry.call(tool_name, arguments)
                call.status = "ok"
                observations.append(call.result)
            except TimeoutError:
                call.status = "timeout"
                call.error = f"Tool exceeded {self.registry.timeout_s:g}s timeout"
            except Exception as exc:
                call.status = "error"
                call.error = f"{type(exc).__name__}: {exc}"
            finally:
                call.latency_s = time.perf_counter() - started
                trace.append(call.as_dict())

        if plan.answer_kind == "grounded":
            hits: list[Hit] = list(self.toolkit._last_hits)
            result = await asyncio.to_thread(
                grounded_from_tool,
                self.llm,
                question,
                self.model_id,
                hits,
            )
            tool_error = next(
                (item["error"] for item in trace if item["status"] in {"error", "timeout"}),
                None,
            )
            if tool_error and not result.error:
                result.error = f"Tool failure: {tool_error}"
            result.tool_trace = trace
            return result

        successful = [observation for observation in observations if observation is not None]
        answer = (
            deterministic_answer(plan.answer_kind, successful[-1])
            if successful
            else "The requested tool did not complete successfully."
        )
        citations = []
        if plan.answer_kind == "grounded":
            citations = []
        return AnswerResult(
            answer=answer,
            mode="agent",
            model_id=self.model_id,
            citations=citations,
            retrieved=[hit.as_dict() for hit in self.toolkit._last_hits],
            tool_trace=trace,
            error=next(
                (item["error"] for item in trace if item["status"] in {"error", "timeout"}),
                None,
            ),
        )
