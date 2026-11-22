from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_bench.experiment.types import CandidateConfig, ExperimentSpec


@dataclass(frozen=True)
class Selection:
    candidate: CandidateConfig
    reason: str
    ucb_scores: dict[str, float]
    skipped: dict[str, str] | None = None


def _mean_metric(trial: dict[str, Any], metric: str) -> float | None:
    values = []
    for workload in trial.get("performance", {}).values():
        value = workload.get("stability", {}).get(metric, {}).get("mean")
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def trial_utility(trial: dict[str, Any]) -> float | None:
    if trial["status"] != "complete" or not trial.get("quality_passed"):
        return None
    goodput = _mean_metric(trial, "goodput_requests_per_second")
    latency = _mean_metric(trial, "latency_p95_s")
    rss = _mean_metric(trial, "peak_process_tree_rss_mb")
    if goodput is None or latency is None:
        return None
    memory_penalty = (rss or 0.0) / 4096.0
    return goodput / (1.0 + latency) - 0.1 * memory_penalty


def _ucb(history: list[dict[str, Any]], candidate_id: str, constant: float = 1.0) -> float:
    values = []
    for trial in history:
        if trial["candidate"]["id"] != candidate_id:
            continue
        utility = trial_utility(trial)
        if utility is not None:
            values.append(utility)
    if not values:
        return -math.inf
    total = max(
        1,
        sum(
            1
            for trial in history
            if trial.get("quality_passed") and trial["status"] == "complete"
        ),
    )
    mean = float(np.mean(values))
    return mean + constant * math.sqrt(math.log(total + 1) / len(values))


def _untested_prior(candidate: CandidateConfig) -> tuple[int, int, str]:
    model_penalty = 2 if "1_7b" in candidate.model_id else 1
    return (model_penalty, -candidate.parallel, candidate.id)


def _history_best(history: list[dict[str, Any]]) -> CandidateConfig | None:
    scored = [
        (trial_utility(trial), trial["candidate"])
        for trial in history
        if trial_utility(trial) is not None
    ]
    if not scored:
        return None
    candidate = max(scored, key=lambda item: (item[0], item[1]["id"]))[1]
    return CandidateConfig(**candidate)


def select_candidate(
    spec: ExperimentSpec,
    strategy: str,
    sequence_no: int,
    history: list[dict[str, Any]],
    memory: list[dict[str, Any]] | None = None,
) -> Selection:
    candidates = list(spec.candidates)
    if not history:
        return Selection(
            spec.default_candidate,
            "shared_unoptimized_start",
            {},
            {},
        )
    if strategy == "default":
        return Selection(
            spec.default_candidate,
            "declared_default_baseline",
            {},
            {},
        )
    if strategy == "grid":
        ordered = [spec.default_candidate] + [
            candidate
            for candidate in candidates
            if candidate.id != spec.default_candidate.id
        ]
        candidate = ordered[sequence_no % len(ordered)]
        return Selection(
            candidate,
            "deterministic_grid_order",
            {},
            {},
        )
    if strategy == "random":
        generator = random.Random(spec.seed + sequence_no)
        candidate = generator.choice(candidates)
        return Selection(
            candidate,
            "seeded_uniform_random",
            {},
            {},
        )
    if strategy not in {"ucb1", "agent"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    negative_ids = {
        item["candidate_id"]
        for item in memory or []
        if item.get("negative_experience")
    }
    if strategy == "agent" and (
        history[-1].get("status") != "complete" or not history[-1].get("quality_passed")
    ):
        fallback = _history_best(history[:-1])
        if fallback is not None:
            return Selection(
                fallback,
                "quality_failure_rollback_to_best_prior_legal_candidate",
                {},
                {},
            )

    tested = {trial["candidate"]["id"] for trial in history}
    untested = [candidate for candidate in candidates if candidate.id not in tested]
    if strategy == "ucb1" and untested:
        candidate = min(untested, key=_untested_prior)
        return Selection(
            candidate,
            "ucb1_exploration_of_untested_legal_candidate",
            {},
            {},
        )

    if strategy == "agent":
        legal_untested = [
            item for item in untested if item.id not in negative_ids
        ]
        if len(history) < 2 and legal_untested:
            candidate = min(legal_untested, key=_untested_prior)
            skipped = {
                candidate_id: "negative_experience"
                for candidate_id in negative_ids
            }
            return Selection(
                candidate,
                "early_reflection_window_explore_untested_legal_candidate",
                {},
                skipped,
            )
        recommendation_counts: dict[str, int] = {}
        categories: set[str] = set()
        for item in memory or []:
            categories.update(item.get("bottleneck_categories", []))
            for candidate_id in item.get("recommended_candidate_ids", []):
                recommendation_counts[candidate_id] = (
                    recommendation_counts.get(candidate_id, 0) + 1
                )
        skipped = {
            candidate_id: "negative_experience"
            for candidate_id in negative_ids
        }
        legal = [candidate for candidate in candidates if candidate.id not in negative_ids]
        if not legal:
            legal = list(candidates)
            skipped = {}

        def agent_prior(candidate: CandidateConfig) -> tuple[float, str]:
            score = 0.0
            reasons = []
            if candidate.cache_prompt:
                score += 0.35
                reasons.append("cache_prompt_on")
            if candidate.cache_type_v == "q8_0":
                score += 0.2
                reasons.append("q8_kv_cache")
            if "0_6b" in candidate.model_id:
                score += 0.2
                reasons.append("lower_memory_model")
            if candidate.parallel > 1:
                score += 0.1
                reasons.append("parallel_batch_candidate")
            if "rss_memory_pressure" in categories and candidate.cache_type_v == "q8_0":
                score += 0.3
                reasons.append("diagnosis_prefers_q8")
            if "ttft_bottleneck" in categories and candidate.cache_prompt:
                score += 0.3
                reasons.append("diagnosis_prefers_prefix_cache")
            if "throughput_queueing" in categories and candidate.parallel > 1:
                score += 0.3
                reasons.append("diagnosis_prefers_parallel")
            score += 0.08 * recommendation_counts.get(candidate.id, 0)
            if recommendation_counts.get(candidate.id):
                reasons.append("positive_memory_recommendation")
            historical = _ucb(history, candidate.id)
            if historical != -math.inf:
                score += 0.5 * historical
                reasons.append("historical_ucb")
            return score, "+".join(reasons) or "legal_exploration"

        ranked = sorted(
            legal,
            key=lambda candidate: (
                -agent_prior(candidate)[0],
                candidates.index(candidate),
            ),
        )
        candidate = ranked[0]
        score, detail = agent_prior(candidate)
        if skipped:
            skipped["_reason"] = "reflection_negative_memory_pruning"
        return Selection(
            candidate,
            f"agent_reflection_pruning_selected score={score:.3f} detail={detail}",
            {candidate.id: score},
            skipped,
        )

    scores = {
        candidate.id: _ucb(history, candidate.id)
        for candidate in candidates
        if any(trial["candidate"]["id"] == candidate.id for trial in history)
    }
    selected = max(
        (candidate for candidate in candidates if candidate.id in scores),
        key=lambda candidate: (scores[candidate.id], -candidates.index(candidate)),
    )
    return Selection(
        selected,
        f"ucb1_selected_with_score={scores[selected.id]:.6f}",
        scores,
        {},
    )
