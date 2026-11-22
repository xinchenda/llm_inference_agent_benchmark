from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from llm_bench.experiment.policies import trial_utility

METRIC_KEYS = (
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "tpot_itl_mean_s",
    "output_tokens_per_second",
    "goodput_requests_per_second",
    "achieved_request_rate",
    "peak_process_tree_rss_mb",
)


def quality_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["status"] == "complete" and row.get("quality_passed")
    ]


def flatten_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for trial in rows:
        for workload_id, workload in trial.get("performance", {}).items():
            stability = workload.get("stability", {})
            row = {
                "trial_id": trial["trial_id"],
                "strategy": trial["strategy"],
                "sequence_no": trial["sequence_no"],
                "candidate_id": trial["candidate"]["id"],
                "quality_passed": trial["quality_passed"],
                "status": trial["status"],
                "workload_id": workload_id,
            }
            for metric in METRIC_KEYS:
                summary = stability.get(metric, {})
                row[f"{metric}_mean"] = summary.get("mean")
                row[f"{metric}_cv"] = summary.get("coefficient_of_variation")
                interval = summary.get("ci95", {})
                row[f"{metric}_ci_low"] = interval.get("low")
                row[f"{metric}_ci_high"] = interval.get("high")
            output.append(row)
    return pd.DataFrame(output)


def strategy_comparison(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output = []
    for strategy in ("default", "grid", "random", "ucb1", "agent"):
        selected = [
            row
            for row in rows
            if row["strategy"] == strategy and row["status"] == "complete"
        ]
        passed = [row for row in selected if row["quality_passed"]]
        scored = [
            (trial_utility(row), row)
            for row in passed
            if trial_utility(row) is not None
        ]
        scored.sort(key=lambda item: (-item[0], item[1]["sequence_no"]))
        best_utility, best = scored[0] if scored else (None, None)
        output.append(
            {
                "strategy": strategy,
                "trials": len(selected),
                "quality_passed_trials": len(passed),
                "best_utility": best_utility,
                "best_candidate": best["candidate"]["id"] if best else None,
                "best_trial_id": best["trial_id"] if best else None,
            }
        )
    return pd.DataFrame(output)


def best_so_far_curve(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output = []
    for strategy in ("default", "grid", "random", "ucb1", "agent"):
        best = None
        sequence = 0
        for row in sorted(
            (item for item in rows if item["strategy"] == strategy),
            key=lambda item: item["sequence_no"],
        ):
            sequence += 1
            utility = trial_utility(row)
            if utility is not None and (best is None or utility > best["utility"]):
                best = {
                    "utility": utility,
                    "candidate": row["candidate"]["id"],
                    "trial": row["trial_id"],
                }
            output.append(
                {
                    "strategy": strategy,
                    "sequence_no": sequence,
                    "trial_id": row["trial_id"],
                    "candidate_id": row["candidate"]["id"],
                    "trial_utility": utility,
                    "best_utility": best["utility"] if best else None,
                    "best_candidate": best["candidate"] if best else None,
                    "best_trial_id": best["trial"] if best else None,
                }
            )
    return pd.DataFrame(output)


def _trial_objectives(trial: dict[str, Any]) -> tuple[float, float, float] | None:
    goodputs = []
    latencies = []
    memories = []
    for workload in trial.get("performance", {}).values():
        stability = workload.get("stability", {})
        goodput = stability.get("goodput_requests_per_second", {}).get("mean")
        latency = stability.get("latency_p95_s", {}).get("mean")
        memory = stability.get("peak_process_tree_rss_mb", {}).get("mean")
        if goodput is not None:
            goodputs.append(float(goodput))
        if latency is not None:
            latencies.append(float(latency))
        if memory is not None:
            memories.append(float(memory))
    if not goodputs or not latencies or not memories:
        return None
    return float(sum(goodputs) / len(goodputs)), float(max(latencies)), float(max(memories))


def pareto_frontier(rows: list[dict[str, Any]]) -> pd.DataFrame:
    points = []
    for trial in quality_trials(rows):
        objectives = _trial_objectives(trial)
        if objectives is None:
            continue
        points.append(
            {
                "trial_id": trial["trial_id"],
                "strategy": trial["strategy"],
                "candidate_id": trial["candidate"]["id"],
                "goodput": objectives[0],
                "p95_latency_s": objectives[1],
                "peak_rss_mb": objectives[2],
            }
        )
    frontier = []
    for candidate in points:
        dominated = any(
            other["goodput"] >= candidate["goodput"]
            and other["p95_latency_s"] <= candidate["p95_latency_s"]
            and other["peak_rss_mb"] <= candidate["peak_rss_mb"]
            and (
                other["goodput"] > candidate["goodput"]
                or other["p95_latency_s"] < candidate["p95_latency_s"]
                or other["peak_rss_mb"] < candidate["peak_rss_mb"]
            )
            for other in points
            if other["trial_id"] != candidate["trial_id"]
        )
        if not dominated:
            frontier.append(candidate)
    return pd.DataFrame(frontier).sort_values(
        ["goodput", "p95_latency_s"],
        ascending=[False, True],
    )


def similar_history(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    target_objectives = _trial_objectives(target)
    if target_objectives is None:
        return []
    target_goodput, target_latency, target_memory = target_objectives
    candidates = []
    for row in quality_trials(rows):
        if row["trial_id"] == target["trial_id"]:
            continue
        objectives = _trial_objectives(row)
        if objectives is None:
            continue
        goodput, latency, memory = objectives
        family_match = row["candidate"]["model_id"] == target["candidate"]["model_id"]
        distance = (
            abs(goodput - target_goodput) / max(abs(target_goodput), 1e-9)
            + abs(latency - target_latency) / max(abs(target_latency), 1e-9)
            + abs(memory - target_memory) / max(abs(target_memory), 1e-9)
            + (0.0 if family_match else 0.25)
        )
        candidates.append(
            {
                "distance": distance,
                "family_match": family_match,
                "trial_id": row["trial_id"],
                "candidate_id": row["candidate"]["id"],
                "strategy": row["strategy"],
                "goodput": goodput,
                "p95_latency_s": latency,
                "peak_rss_mb": memory,
                "artifact_path": row["artifact_path"],
            }
        )
    return sorted(
        candidates,
        key=lambda item: (item["distance"], item["trial_id"]),
    )[:limit]


def cache_ablation(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output = []
    for trial in quality_trials(rows):
        workload = trial.get("performance", {}).get("prefix_cache_rag")
        if not workload:
            continue
        stability = workload.get("stability", {})

        def value(
            metric: str,
            stability: dict[str, Any] = stability,
        ) -> float | None:
            return stability.get(metric, {}).get("mean")

        candidate = trial["candidate"]
        repeats = workload.get("repeats", [])
        valid_cache_repeats = [
            repeat for repeat in repeats if repeat.get("cache_evidence_valid")
        ]
        cache_n_total = sum(int(repeat.get("cache_n_total") or 0) for repeat in repeats)
        cache_prompt_tokens_total = sum(
            int(repeat.get("cache_prompt_tokens_total") or 0) for repeat in repeats
        )
        output.append(
            {
                "trial_id": trial["trial_id"],
                "strategy": trial["strategy"],
                "candidate_id": candidate["id"],
                "model_id": candidate["model_id"],
                "cache_prompt": candidate["cache_prompt"],
                "cache_type_k": candidate["cache_type_k"],
                "cache_type_v": candidate["cache_type_v"],
                "context_size": candidate["context_size"],
                "parallel": candidate["parallel"],
                "batch_size": candidate["batch_size"],
                "ubatch_size": candidate["ubatch_size"],
                "flash_attn": candidate["flash_attn"],
                "cache_hit_rate": (
                    cache_n_total / cache_prompt_tokens_total
                    if cache_prompt_tokens_total > 0
                    else None
                ),
                "cache_evidence_valid": bool(repeats)
                and len(valid_cache_repeats) == len(repeats),
                "invalid_cache_observations": sum(
                    int(repeat.get("invalid_cache_observations") or 0)
                    for repeat in repeats
                ),
                "ttft_p50_s": value("ttft_p50_s"),
                "ttft_p95_s": value("ttft_p95_s"),
                "goodput_requests_per_second": value(
                    "goodput_requests_per_second"
                ),
                "peak_process_tree_rss_mb": value("peak_process_tree_rss_mb"),
                "error_rate": value("error_rate"),
            }
        )
    return pd.DataFrame(output)


def cache_ablation_pairs(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    group_columns = [
        "model_id",
        "cache_type_k",
        "cache_type_v",
        "context_size",
        "parallel",
        "batch_size",
        "ubatch_size",
        "flash_attn",
        "cache_prompt",
    ]
    metrics = [
        "cache_hit_rate",
        "ttft_p50_s",
        "ttft_p95_s",
        "goodput_requests_per_second",
        "peak_process_tree_rss_mb",
        "error_rate",
    ]
    grouped = trials.groupby(group_columns, dropna=False)[metrics].mean().reset_index()
    evidence = (
        trials.groupby(group_columns, dropna=False)
        .agg(
            cache_evidence_valid=("cache_evidence_valid", "all"),
            invalid_cache_observations=("invalid_cache_observations", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.merge(evidence, on=group_columns, how="left")
    pairs = []
    for _key, group in grouped.groupby(
        [column for column in group_columns if column != "cache_prompt"],
        dropna=False,
    ):
        enabled = group[group["cache_prompt"] == True]  # noqa: E712
        disabled = group[group["cache_prompt"] == False]  # noqa: E712
        if enabled.empty or disabled.empty:
            continue
        on = enabled.iloc[0]
        off = disabled.iloc[0]
        pairs.append(
            {
                "model_id": on["model_id"],
                "kv_cache": f"{on['cache_type_k']}/{on['cache_type_v']}",
                "context_size": on["context_size"],
                "parallel": on["parallel"],
                "batch_size": on["batch_size"],
                "ubatch_size": on["ubatch_size"],
                "flash_attn": on["flash_attn"],
                "cache_hit_rate_enabled": on["cache_hit_rate"],
                "cache_hit_rate_disabled": off["cache_hit_rate"],
                "cache_evidence_valid_enabled": bool(on["cache_evidence_valid"]),
                "cache_evidence_valid_disabled": bool(off["cache_evidence_valid"]),
                "invalid_cache_observations_enabled": int(
                    on["invalid_cache_observations"]
                ),
                "invalid_cache_observations_disabled": int(
                    off["invalid_cache_observations"]
                ),
                "ttft_p50_enabled_s": on["ttft_p50_s"],
                "ttft_p50_disabled_s": off["ttft_p50_s"],
                "ttft_p50_reduction_pct": (
                    (off["ttft_p50_s"] - on["ttft_p50_s"])
                    / off["ttft_p50_s"]
                    * 100.0
                    if off["ttft_p50_s"] not in (None, 0)
                    and on["ttft_p50_s"] is not None
                    else None
                ),
                "ttft_p95_enabled_s": on["ttft_p95_s"],
                "ttft_p95_disabled_s": off["ttft_p95_s"],
                "ttft_p95_reduction_pct": (
                    (off["ttft_p95_s"] - on["ttft_p95_s"])
                    / off["ttft_p95_s"]
                    * 100.0
                    if off["ttft_p95_s"] not in (None, 0)
                    and on["ttft_p95_s"] is not None
                    else None
                ),
                "goodput_change_pct": (
                    (on["goodput_requests_per_second"] - off["goodput_requests_per_second"])
                    / off["goodput_requests_per_second"]
                    * 100.0
                    if off["goodput_requests_per_second"] not in (None, 0)
                    and on["goodput_requests_per_second"] is not None
                    else None
                ),
                "peak_rss_change_pct": (
                    (on["peak_process_tree_rss_mb"] - off["peak_process_tree_rss_mb"])
                    / off["peak_process_tree_rss_mb"]
                    * 100.0
                    if off["peak_process_tree_rss_mb"] not in (None, 0)
                    and on["peak_process_tree_rss_mb"] is not None
                    else None
                ),
                "error_rate_enabled": on["error_rate"],
                "error_rate_disabled": off["error_rate"],
            }
        )
    return pd.DataFrame(pairs)


def export_analysis(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = flatten_metrics(rows)
    comparison = strategy_comparison(rows)
    best = best_so_far_curve(rows)
    pareto = pareto_frontier(rows)
    cache_trials = cache_ablation(rows)
    cache_pairs = cache_ablation_pairs(cache_trials)
    metrics.to_csv(output_dir / "trial_metrics.csv", index=False)
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    best.to_csv(output_dir / "best_so_far.csv", index=False)
    pareto.to_csv(output_dir / "pareto_frontier.csv", index=False)
    cache_trials.to_csv(output_dir / "cache_ablation_trials.csv", index=False)
    cache_pairs.to_csv(output_dir / "cache_ablation_pairs.csv", index=False)
    quality = quality_trials(rows)
    similar = {
        trial["trial_id"]: similar_history(rows, trial)
        for trial in quality[-4:]
    }
    (output_dir / "similar_history.json").write_text(
        json.dumps(similar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "comparison": comparison,
        "best_so_far": best,
        "pareto": pareto,
        "cache_ablation": cache_trials,
        "cache_pairs": cache_pairs,
        "similar_history": similar,
    }
