from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


@dataclass(frozen=True)
class PrometheusSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus_text(text: str) -> list[PrometheusSample]:
    samples: list[PrometheusSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_PATTERN.match(line)
        if not match:
            continue
        labels: dict[str, str] = {}
        for item in (match.group("labels") or "").split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            labels[key.strip()] = value.strip().strip('"')
        value = float(match.group("value"))
        if math.isnan(value):
            continue
        samples.append(
            PrometheusSample(
                name=match.group("name"),
                labels=labels,
                value=value,
            )
        )
    return samples


def _total(samples: list[PrometheusSample], names: set[str]) -> float | None:
    values = [sample.value for sample in samples if sample.name in names]
    return sum(values) if values else None


def _gauge(samples: list[PrometheusSample], names: set[str]) -> float | None:
    values = [sample.value for sample in samples if sample.name in names]
    return values[-1] if values else None


def _bucket_values(
    samples: list[PrometheusSample],
    metric: str,
) -> list[tuple[float, float]]:
    buckets: dict[float, float] = {}
    for sample in samples:
        if sample.name != f"{metric}_bucket":
            continue
        boundary = sample.labels.get("le")
        if boundary is None:
            continue
        limit = math.inf if boundary == "+Inf" else float(boundary)
        buckets[limit] = buckets.get(limit, 0.0) + sample.value
    return sorted(buckets.items())


def histogram_quantile(
    samples: list[PrometheusSample],
    metric: str,
    quantile: float,
) -> float | None:
    buckets = _bucket_values(samples, metric)
    if not buckets or not buckets[-1][1]:
        return None
    target = quantile * buckets[-1][1]
    previous_limit = 0.0
    previous_count = 0.0
    for limit, cumulative in buckets:
        if cumulative >= target:
            if math.isinf(limit):
                return previous_limit
            if cumulative == previous_count:
                return limit
            fraction = (target - previous_count) / (cumulative - previous_count)
            return previous_limit + fraction * (limit - previous_limit)
        previous_limit = limit
        previous_count = cumulative
    return None


def histogram_mean(samples: list[PrometheusSample], metric: str) -> float | None:
    total = _total(samples, {f"{metric}_sum"})
    count = _total(samples, {f"{metric}_count"})
    return total / count if total is not None and count not in (None, 0) else None


def summarize_serving_metrics(text: str) -> dict[str, Any]:
    samples = parse_prometheus_text(text)
    prefix_queries = _total(
        samples,
        {"vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"},
    )
    prefix_hits = _total(
        samples,
        {"vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"},
    )
    ttft_metric = "vllm:time_to_first_token_seconds"
    tpot_metric = "vllm:request_time_per_output_token_seconds"
    e2e_metric = "vllm:e2e_request_latency_seconds"
    return {
        "samples": len(samples),
        "raw_metric_names": sorted({sample.name for sample in samples}),
        "num_requests_running": _gauge(
            samples,
            {"vllm:num_requests_running"},
        ),
        "num_requests_waiting": _gauge(
            samples,
            {"vllm:num_requests_waiting"},
        ),
        "kv_cache_usage_perc": _gauge(
            samples,
            {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"},
        ),
        "prefix_cache_queries": prefix_queries,
        "prefix_cache_hits": prefix_hits,
        "prefix_cache_hit_rate": (
            prefix_hits / prefix_queries
            if prefix_hits is not None and prefix_queries not in (None, 0)
            else None
        ),
        "prompt_tokens_total": _total(samples, {"vllm:prompt_tokens_total"}),
        "generation_tokens_total": _total(
            samples,
            {"vllm:generation_tokens_total"},
        ),
        "request_success_total": _total(
            samples,
            {"vllm:request_success_total"},
        ),
        "ttft_histogram_bucket_count": _gauge(
            samples,
            {"vllm:time_to_first_token_seconds_count"},
        ),
        "ttft_p50_s": histogram_quantile(samples, ttft_metric, 0.50),
        "ttft_p95_s": histogram_quantile(samples, ttft_metric, 0.95),
        "tpot_mean_s": histogram_mean(samples, tpot_metric),
        "e2e_latency_p95_s": histogram_quantile(samples, e2e_metric, 0.95),
        "e2e_latency_histogram_bucket_count": _gauge(
            samples,
            {"vllm:e2e_request_latency_seconds_count"},
        ),
    }


def summarize_sglang_serving_metrics(text: str) -> dict[str, Any]:
    samples = parse_prometheus_text(text)
    ttft_metric = "sglang:time_to_first_token_seconds"
    itl_metric = "sglang:inter_token_latency_seconds"
    e2e_metric = "sglang:e2e_request_latency_seconds"
    return {
        "samples": len(samples),
        "raw_metric_names": sorted({sample.name for sample in samples}),
        "num_running_reqs": _gauge(samples, {"sglang:num_running_reqs"}),
        "num_queue_reqs": _gauge(samples, {"sglang:num_queue_reqs"}),
        "token_usage": _gauge(samples, {"sglang:token_usage"}),
        "cache_hit_rate": _gauge(samples, {"sglang:cache_hit_rate"}),
        "prompt_tokens_total": _total(samples, {"sglang:prompt_tokens_total"}),
        "generation_tokens_total": _total(
            samples,
            {"sglang:generation_tokens_total"},
        ),
        "ttft_p50_s": histogram_quantile(samples, ttft_metric, 0.50),
        "ttft_p95_s": histogram_quantile(samples, ttft_metric, 0.95),
        "itl_mean_s": histogram_mean(samples, itl_metric),
        "e2e_latency_p95_s": histogram_quantile(samples, e2e_metric, 0.95),
    }
