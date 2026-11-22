from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from llm_bench.experiment.types import ExperimentSpec, TrialResult


def _metric(performance: dict[str, Any], workload_id: str, metric: str) -> float | None:
    workload = performance.get(workload_id, {})
    value = workload.get("stability", {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


def _best_historical_candidate(history: list[dict[str, Any]]) -> str | None:
    from llm_bench.experiment.policies import trial_utility

    scored = [
        (trial_utility(trial), trial["candidate"]["id"])
        for trial in history
        if trial_utility(trial) is not None
    ]
    return max(scored)[1] if scored else None


def reflect_trial(
    spec: ExperimentSpec,
    result: TrialResult,
    history: list[dict[str, Any]],
    baseline_performance: dict[str, Any] | None,
) -> dict[str, Any]:
    categories: list[str] = []
    evidence: dict[str, Any] = {}
    recommendations: list[str] = []
    negative = False

    if result.status == "quality_gate_failed":
        categories.append("quality_gate_failure")
        negative = True
        failed = {
            workload_id: payload.get("failures", [])
            for workload_id, payload in result.quality.items()
            if isinstance(payload, dict) and not payload.get("passed", True)
        }
        evidence["quality_failures"] = failed
        recommendations.extend(
            candidate.id
            for candidate in spec.candidates
            if candidate.id != result.candidate.id
        )
    elif result.status == "failed":
        categories.append("request_or_runtime_failure")
        negative = True
        evidence["failure_type"] = result.failure_type
        evidence["failure_detail"] = result.failure_detail
    elif result.performance:
        workload_evidence: dict[str, Any] = {}
        for workload in spec.workloads:
            ttft = _metric(result.performance, workload.id, "ttft_p95_s")
            latency = _metric(result.performance, workload.id, "latency_p95_s")
            goodput = _metric(
                result.performance,
                workload.id,
                "goodput_requests_per_second",
            )
            rss = _metric(result.performance, workload.id, "peak_process_tree_rss_mb")
            workload_evidence[workload.id] = {
                "ttft_p95_s": ttft,
                "latency_p95_s": latency,
                "goodput_requests_per_second": goodput,
                "peak_process_tree_rss_mb": rss,
            }
            if ttft is not None and ttft >= workload.latency_slo_s * 0.4:
                categories.append("ttft_bottleneck")
            if (
                latency is not None
                and goodput is not None
                and latency >= workload.latency_slo_s * 0.65
            ):
                categories.append("throughput_queueing")
        max_rss = max(
            (
                value["peak_process_tree_rss_mb"]
                for value in workload_evidence.values()
                if value["peak_process_tree_rss_mb"] is not None
            ),
            default=None,
        )
        baseline_rss = None
        if baseline_performance:
            baseline_rss = max(
                (
                    _metric(
                        baseline_performance,
                        workload.id,
                        "peak_process_tree_rss_mb",
                    )
                    or 0.0
                    for workload in spec.workloads
                ),
                default=0.0,
            )
        if max_rss is not None and (
            max_rss >= 1800 or (baseline_rss and max_rss >= baseline_rss * 1.25)
        ):
            categories.append("rss_memory_pressure")
        evidence["workloads"] = workload_evidence
        evidence["baseline_max_rss_mb"] = baseline_rss

        if "rss_memory_pressure" in categories:
            recommendations.extend(
                candidate.id
                for candidate in spec.candidates
                if "0_6b" in candidate.model_id or candidate.cache_type_v == "q8_0"
            )
        if "ttft_bottleneck" in categories or "throughput_queueing" in categories:
            recommendations.extend(
                candidate.id
                for candidate in spec.candidates
                if candidate.parallel >= result.candidate.parallel
                or candidate.batch_size >= result.candidate.batch_size
            )

    categories = list(dict.fromkeys(categories)) or ["no_material_bottleneck"]
    recommendations = list(dict.fromkeys(recommendations))
    legal_recommendations = [
        candidate.id
        for candidate in spec.candidates
        if candidate.id in recommendations and candidate.id != result.candidate.id
    ]
    best_candidate = _best_historical_candidate(history)
    if not legal_recommendations and best_candidate:
        legal_recommendations = [best_candidate]
    summary = (
        f"{result.candidate.id}: {', '.join(categories)}; "
        f"recommended legal candidates={legal_recommendations or ['none']}"
    )
    return {
        "reflection_id": f"reflection-{result.trial_id}",
        "trial_id": result.trial_id,
        "spec_id": result.spec_id,
        "strategy": result.strategy,
        "candidate_id": result.candidate.id,
        "bottleneck_categories": categories,
        "positive_experience": (
            result.status == "complete"
            and result.quality_passed
            and not negative
        ),
        "negative_experience": negative,
        "evidence": evidence,
        "recommended_candidate_ids": legal_recommendations,
        "memory_summary": summary,
        "created_at": datetime.now(UTC).isoformat(),
    }
