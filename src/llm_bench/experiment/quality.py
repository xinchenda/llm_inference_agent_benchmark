from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from llm_bench.experiment.runtime import classify_failure
from llm_bench.experiment.types import QualitySpec, WorkloadSpec
from llm_bench.json_utils import OutputFormatError, parse_json_object
from llm_bench.llm import LocalLLM


def _normalized_exact(value: str) -> str:
    return re.sub(r"^[\s\"'`]+|[\s\"'`.]+$", "", value).strip().lower()


def _contains_json(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def evaluate_quality_output(spec: QualitySpec, text: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if spec.expected_exact is not None:
        expected = _normalized_exact(spec.expected_exact)
        actual = _normalized_exact(text)
        if actual != expected:
            failures.append(f"exact mismatch: expected {spec.expected_exact!r}, got {text!r}")
    for expected in spec.expected_contains:
        if expected.lower() not in text.lower():
            failures.append(f"missing required fact: {expected!r}")
    if spec.expected_json:
        try:
            payload = parse_json_object(text)
        except OutputFormatError as exc:
            failures.append(f"invalid JSON: {exc}")
        else:
            if not _contains_json(payload, spec.expected_json):
                failures.append(
                    f"JSON mismatch: expected subset {spec.expected_json!r}, got {payload!r}"
                )
    return not failures, failures


async def run_quality_gate(
    llm: LocalLLM,
    workloads: tuple[WorkloadSpec, ...],
    timeout_s: float,
) -> tuple[bool, dict[str, Any]]:
    results: dict[str, Any] = {}
    all_passed = True
    for workload in workloads:
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm.generate,
                    "Answer directly. /no_think",
                    workload.quality.prompt,
                    max_tokens=64,
                    seed=7,
                    disable_cache=True,
                ),
                timeout=timeout_s,
            )
            passed, failures = evaluate_quality_output(workload.quality, response.text)
            results[workload.id] = {
                "passed": passed,
                "failures": failures,
                "latency_s": response.latency_s,
                "ttft_s": response.ttft_s,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "raw_output": response.text,
                "error_type": None,
                "error": None,
            }
        except Exception as exc:
            passed = False
            results[workload.id] = {
                "passed": False,
                "failures": [f"quality request failed: {type(exc).__name__}: {exc}"],
                "latency_s": time.perf_counter() - started,
                "ttft_s": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "raw_output": "",
                "error_type": classify_failure(exc),
                "error": f"{type(exc).__name__}: {exc}",
            }
        all_passed = all_passed and passed
    return all_passed, results
