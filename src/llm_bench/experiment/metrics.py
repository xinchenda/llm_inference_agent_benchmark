from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from llm_bench.experiment.load import RequestRecord
from llm_bench.experiment.types import WorkloadSpec


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def bootstrap_interval(
    values: list[float],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return {"low": float(array[0]), "high": float(array[0])}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
    }


def summarize_requests(
    workload: WorkloadSpec,
    records: list[RequestRecord],
    wall_s: float,
    peak_rss_mb: float,
) -> dict[str, Any]:
    latencies = [record.latency_s for record in records if record.success]
    ttfts = [
        record.ttft_s
        for record in records
        if record.success and record.ttft_s is not None
    ]
    tpots = [
        record.tpot_s
        for record in records
        if record.success and record.tpot_s is not None
    ]
    completion_tokens = [
        record.completion_tokens
        for record in records
        if record.success and record.completion_tokens is not None
    ]
    good = sum(record.success and record.slo_met for record in records)
    successful = sum(record.success for record in records)
    cache_records = [
        record
        for record in records
        if record.success
        and record.cache_n is not None
        and record.cache_prompt_tokens is not None
        and record.cache_prompt_tokens > 0
    ]
    cache_n_total = sum(record.cache_n or 0 for record in cache_records)
    cache_prompt_tokens_total = sum(
        record.cache_prompt_tokens or 0 for record in cache_records
    )
    invalid_cache_observations = sum(
        record.success
        and (
            record.cache_n is None
            or record.cache_prompt_tokens is None
            or record.cache_prompt_tokens <= 0
        )
        for record in records
    )
    failures = Counter(
        record.error_type for record in records if record.error_type is not None
    )
    if workload.load_mode == "open_qps" and len(records) > 1:
        starts = sorted(record.started_s for record in records)
        start_span = max(starts[-1] - starts[0], 1e-9)
        achieved_request_rate = (len(starts) - 1) / start_span
    else:
        achieved_request_rate = len(records) / wall_s if wall_s > 0 else None
    return {
        "workload_id": workload.id,
        "schedule": workload.load_mode,
        "target_qps": workload.target_qps,
        "achieved_request_rate": achieved_request_rate,
        "requests": len(records),
        "successful_requests": successful,
        "good_requests": good,
        "error_rate": 1.0 - successful / len(records) if records else 0.0,
        "wall_s": wall_s,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "tpot_itl_mean_s": float(np.mean(tpots)) if tpots else None,
        "output_tokens_per_second": (
            sum(completion_tokens) / wall_s if wall_s > 0 and completion_tokens else None
        ),
        "goodput_requests_per_second": good / wall_s if wall_s > 0 else 0.0,
        "peak_process_tree_rss_mb": peak_rss_mb,
        "cache_n_total": cache_n_total,
        "cache_prompt_tokens_total": cache_prompt_tokens_total,
        "cache_hit_rate": (
            cache_n_total / cache_prompt_tokens_total
            if cache_prompt_tokens_total > 0
            else None
        ),
        "cache_observations": len(cache_records),
        "invalid_cache_observations": invalid_cache_observations,
        "cache_evidence_valid": bool(
            successful > 0 and len(cache_records) == successful
        ),
        "latency_slo_s": workload.latency_slo_s,
        "failure_counts": dict(failures),
    }


def aggregate_workload_repeats(
    workload: WorkloadSpec,
    repeats: list[dict[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_itl_mean_s",
        "output_tokens_per_second",
        "goodput_requests_per_second",
        "peak_process_tree_rss_mb",
        "achieved_request_rate",
        "cache_hit_rate",
        "error_rate",
    )
    stability: dict[str, Any] = {}
    for offset, metric in enumerate(metrics):
        values = [
            float(repeat[metric])
            for repeat in repeats
            if repeat.get(metric) is not None
        ]
        mean = float(np.mean(values)) if values else None
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        stability[metric] = {
            "values": values,
            "mean": mean,
            "std": std,
            "coefficient_of_variation": (
                std / abs(mean) if std is not None and mean not in (None, 0.0) else None
            ),
            "ci95": bootstrap_interval(
                values,
                resamples=resamples,
                confidence=confidence,
                seed=seed + offset,
            ),
        }
    return {
        "workload_id": workload.id,
        "repeats": repeats,
        "cache_n_total": sum(int(repeat.get("cache_n_total") or 0) for repeat in repeats),
        "cache_prompt_tokens_total": sum(
            int(repeat.get("cache_prompt_tokens_total") or 0) for repeat in repeats
        ),
        "cache_hit_rate_pooled": (
            sum(int(repeat.get("cache_n_total") or 0) for repeat in repeats)
            / sum(int(repeat.get("cache_prompt_tokens_total") or 0) for repeat in repeats)
            if sum(int(repeat.get("cache_prompt_tokens_total") or 0) for repeat in repeats) > 0
            else None
        ),
        "cache_evidence_valid": all(
            bool(repeat.get("cache_evidence_valid")) for repeat in repeats
        ),
        "invalid_cache_observations": sum(
            int(repeat.get("invalid_cache_observations") or 0) for repeat in repeats
        ),
        "stability": stability,
    }


def _better(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline:
        return {}
    latency = candidate.get("latency_p95_s")
    base_latency = baseline.get("latency_p95_s")
    goodput = candidate.get("goodput_requests_per_second")
    base_goodput = baseline.get("goodput_requests_per_second")
    rss = candidate.get("peak_process_tree_rss_mb")
    base_rss = baseline.get("peak_process_tree_rss_mb")
    return {
        "latency_p95_s_delta": (
            float(latency) - float(base_latency)
            if latency is not None and base_latency is not None
            else None
        ),
        "latency_p95_s_change_ratio": (
            float(latency) / float(base_latency) - 1.0
            if latency is not None and base_latency not in (None, 0.0)
            else None
        ),
        "goodput_change_ratio": (
            float(goodput) / float(base_goodput) - 1.0
            if goodput is not None and base_goodput not in (None, 0.0)
            else None
        ),
        "peak_rss_mb_delta": (
            float(rss) - float(base_rss)
            if rss is not None and base_rss is not None
            else None
        ),
    }


def baseline_deltas(
    candidate_performance: dict[str, Any],
    baseline_performance: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline_performance:
        return {"status": "unavailable"}
    output: dict[str, Any] = {}
    for workload_id, candidate in candidate_performance.items():
        baseline = baseline_performance.get(workload_id)
        if not baseline:
            continue
        merged_candidate = {
            **candidate,
            **{
                metric: candidate["stability"][metric]["mean"]
                for metric in candidate.get("stability", {})
            },
        }
        merged_baseline = {
            **baseline,
            **{
                metric: baseline["stability"][metric]["mean"]
                for metric in baseline.get("stability", {})
            },
        }
        output[workload_id] = _better(merged_candidate, merged_baseline)
    return output
