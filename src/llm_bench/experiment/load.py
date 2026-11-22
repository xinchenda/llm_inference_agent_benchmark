from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from llm_bench.experiment.runtime import classify_failure
from llm_bench.experiment.types import WorkloadSpec
from llm_bench.llm import LocalLLM


def cache_observation(
    timings: dict[str, Any] | None,
    prompt_tokens: int | None,
) -> tuple[int | None, int | None, float | None]:
    """Return measured cache_n, prompt tokens, and hit rate or explicit None values."""
    raw_cache_n = (timings or {}).get("cache_n")
    cache_n = (
        int(raw_cache_n)
        if isinstance(raw_cache_n, (int, float))
        and not isinstance(raw_cache_n, bool)
        and raw_cache_n >= 0
        else None
    )
    observed_prompt_tokens = (
        int(prompt_tokens)
        if isinstance(prompt_tokens, int) and prompt_tokens > 0
        else None
    )
    hit_rate = (
        cache_n / observed_prompt_tokens
        if cache_n is not None
        and observed_prompt_tokens is not None
        and observed_prompt_tokens > 0
        else None
    )
    return cache_n, observed_prompt_tokens, hit_rate


@dataclass(frozen=True)
class RequestRecord:
    index: int
    scheduled_s: float
    started_s: float
    latency_s: float
    ttft_s: float | None
    tpot_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    success: bool
    slo_met: bool
    error_type: str | None
    error: str | None
    text: str
    cache_n: int | None = None
    cache_prompt_tokens: int | None = None
    cache_hit_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scheduled_s": self.scheduled_s,
            "started_s": self.started_s,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "success": self.success,
            "slo_met": self.slo_met,
            "error_type": self.error_type,
            "error": self.error,
            "text": self.text,
            "cache_n": self.cache_n,
            "cache_prompt_tokens": self.cache_prompt_tokens,
            "cache_hit_rate": self.cache_hit_rate,
        }


def make_prompt(workload: WorkloadSpec, request_index: int) -> str:
    technical = (
        "The inference service processes technical documents and reports measured "
        "latency, memory, citation quality, and tool status. "
    )
    repeats = max(1, workload.prompt_words // len(technical.split()))
    filler = (technical * repeats).strip()
    if workload.id == "rag_reading":
        return (
            "Reference evidence:\n"
            "Atlas limit is 42. The Atlas service uses a bounded request queue.\n"
            f"{filler}\n\n"
            "Use only the reference fact. Reply briefly and do not repeat the context."
        )
    if workload.id == "batch_generation":
        return (
            f"Request {request_index}. {filler}\n"
            "Write a concise technical status paragraph under the requested token limit."
        )
    if workload.id == "prefix_cache_rag":
        shared_document = (
            "Atlas serving reference. The Atlas limit is 42 requests. "
            "The queue is bounded to preserve latency. Authentication uses a "
            "bearer token from the environment; secrets are never stored in source "
            "control. Metrics include TTFT, total latency, memory, and goodput. "
            f"{filler}"
        )
        questions = (
            "What is the Atlas limit?",
            "Why is the Atlas queue bounded?",
            "How is authentication supplied?",
            "Which serving metrics are recorded?",
            "What should not be stored in source control?",
            "Which workload uses a fixed request rate?",
        )
        question = questions[request_index % len(questions)]
        return f"{shared_document}\n\nQuestion: {question}\nAnswer briefly."
    return f"Request {request_index}. {filler}\nReply briefly."


async def _request(
    llm: LocalLLM,
    workload: WorkloadSpec,
    index: int,
    scheduled_s: float,
    semaphore: asyncio.Semaphore,
) -> RequestRecord:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm.generate,
                    "Answer directly. /no_think",
                    make_prompt(workload, index),
                    max_tokens=workload.output_tokens,
                    seed=42 + index,
                    disable_cache=False,
                ),
                timeout=max(llm.timeout_s, workload.latency_slo_s * 2),
            )
            completion_tokens = response.completion_tokens
            prompt_tokens = response.prompt_tokens
            cache_n, observed_prompt_tokens, cache_hit_rate = cache_observation(
                response.timings,
                prompt_tokens,
            )
            tpot_s = None
            if response.ttft_s is not None and completion_tokens and completion_tokens > 1:
                tpot_s = max(response.latency_s - response.ttft_s, 0.0) / (
                    completion_tokens - 1
                )
            slo_met = response.latency_s <= workload.latency_slo_s
            return RequestRecord(
                index=index,
                scheduled_s=scheduled_s,
                started_s=started,
                latency_s=response.latency_s,
                ttft_s=response.ttft_s,
                tpot_s=tpot_s,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                success=True,
                slo_met=slo_met,
                error_type=None if slo_met else "slo_violation",
                error=None if slo_met else f"latency exceeded {workload.latency_slo_s:g}s",
                text=response.text,
                cache_n=cache_n,
                cache_prompt_tokens=observed_prompt_tokens,
                cache_hit_rate=cache_hit_rate,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            return RequestRecord(
                index=index,
                scheduled_s=scheduled_s,
                started_s=started,
                latency_s=latency,
                ttft_s=None,
                tpot_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                success=False,
                slo_met=False,
                error_type=classify_failure(exc),
                error=f"{type(exc).__name__}: {exc}",
                text="",
            )


async def run_workload(
    llm: LocalLLM,
    workload: WorkloadSpec,
) -> tuple[list[RequestRecord], float]:
    semaphore = asyncio.Semaphore(workload.concurrency)
    run_started = time.perf_counter()
    if workload.load_mode == "closed":
        tasks = [
            asyncio.create_task(
                _request(llm, workload, index, time.perf_counter() - run_started, semaphore)
            )
            for index in range(workload.requests)
        ]
    elif workload.load_mode == "open_qps":
        assert workload.target_qps is not None

        async def scheduled_request(index: int) -> RequestRecord:
            target_offset = index / workload.target_qps
            delay = run_started + target_offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            return await _request(llm, workload, index, target_offset, semaphore)

        tasks = [
            asyncio.create_task(scheduled_request(index))
            for index in range(workload.requests)
        ]
    else:
        raise ValueError(f"Unsupported load mode: {workload.load_mode}")
    records = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - run_started
    return records, wall_s
