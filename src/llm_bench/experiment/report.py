from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_bench.config import PROJECT_ROOT
from llm_bench.experiment.analysis import export_analysis
from llm_bench.experiment.ledger import ExperimentLedger


def _markdown_table(frame: Any) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_so_far_figure(best: Any, destination: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    for strategy in ("default", "grid", "random", "ucb1", "agent"):
        subset = best[best["strategy"] == strategy]
        if subset.empty:
            continue
        plt.plot(
            subset["sequence_no"],
            subset["best_utility"],
            marker="o",
            label=strategy,
        )
    plt.xlabel("Trial sequence within equal budget")
    plt.ylabel("Best quality-passing utility")
    plt.title("Best-so-far by strategy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=160)
    plt.close()


def _pareto_figure(pareto: Any, destination: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    if not pareto.empty:
        sizes = pareto["peak_rss_mb"].to_numpy(dtype=float)
        scale = 2400.0 / max(float(sizes.max()), 1.0)
        plt.scatter(
            pareto["p95_latency_s"],
            pareto["goodput"],
            s=sizes * scale,
            alpha=0.7,
        )
        for _, row in pareto.iterrows():
            plt.annotate(
                row["candidate_id"],
                (row["p95_latency_s"], row["goodput"]),
                fontsize=7,
            )
    plt.xlabel("Mean p95 latency (s), lower is better")
    plt.ylabel("Mean goodput (requests/s), higher is better")
    plt.title("Quality-passing Pareto frontier")
    plt.grid(alpha=0.25)
    plt.margins(x=0.16, y=0.12)
    plt.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=160)
    plt.close()


def _cache_ablation_figure(pairs: Any, destination: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    if not pairs.empty:
        labels = pairs["kv_cache"].astype(str).tolist()
        positions = list(range(len(labels)))
        width = 0.36
        plt.bar(
            [position - width / 2 for position in positions],
            pairs["ttft_p95_disabled_s"],
            width,
            label="cache_prompt=False",
        )
        plt.bar(
            [position + width / 2 for position in positions],
            pairs["ttft_p95_enabled_s"],
            width,
            label="cache_prompt=True",
        )
        plt.xticks(positions, labels)
    plt.xlabel("KV cache type")
    plt.ylabel("Shared-prefix TTFT p95 (s)")
    plt.title("Prefix Cache ablation with real cache_n evidence")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=160)
    plt.close()


def build_experiment_report(
    ledger_path: Path,
    output_dir: Path,
) -> Path:
    ledger = ExperimentLedger(ledger_path)
    rows = ledger.all_trials()
    if not rows:
        raise ValueError("Experiment ledger contains no trials")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger.export_jsonl(output_dir / "trials.jsonl")
    reflections = ledger.all_reflections()
    (output_dir / "reflections.json").write_text(
        json.dumps(reflections, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_summary = ledger_path.parent / "run_summary.json"
    if run_summary.exists():
        shutil.copyfile(run_summary, output_dir / "run_summary.json")
    analysis = export_analysis(rows, output_dir)
    figures = output_dir / "figures"
    _best_so_far_figure(analysis["best_so_far"], figures / "best_so_far.png")
    _pareto_figure(analysis["pareto"], figures / "pareto.png")
    _cache_ablation_figure(analysis["cache_pairs"], figures / "cache_ablation.png")

    comparison = analysis["comparison"].fillna("")
    pareto = analysis["pareto"].fillna("")
    cache_pairs = analysis["cache_pairs"].fillna("")
    cache_columns = [
        "kv_cache",
        "cache_hit_rate_enabled",
        "cache_hit_rate_disabled",
        "cache_evidence_valid_enabled",
        "cache_evidence_valid_disabled",
        "invalid_cache_observations_enabled",
        "invalid_cache_observations_disabled",
        "ttft_p95_enabled_s",
        "ttft_p95_disabled_s",
        "ttft_p95_reduction_pct",
        "goodput_change_pct",
        "peak_rss_change_pct",
        "error_rate_enabled",
        "error_rate_disabled",
    ]
    cache_table = cache_pairs[cache_columns] if not cache_pairs.empty else cache_pairs
    metrics = analysis["metrics"]
    agent_best = analysis["best_so_far"][
        (analysis["best_so_far"]["strategy"] == "agent")
        & analysis["best_so_far"]["best_utility"].notna()
    ]
    if agent_best.empty:
        agent_observation = "No Agent strategy was run in this experiment."
    else:
        agent_start = float(agent_best.iloc[0]["best_utility"])
        agent_end = float(agent_best.iloc[-1]["best_utility"])
        agent_change = (agent_end / agent_start - 1.0) * 100.0 if agent_start else None
        agent_observation = (
            f"The agent policy improved its own best-so-far utility by "
            f"{agent_change:.2f}% over its first trial. This does not imply that "
            "the agent beat every strategy: the measured equal-budget winner "
            "remains visible in the table."
        )
    rag_rates = metrics[
        (metrics["workload_id"] == "rag_reading")
        & metrics["achieved_request_rate_mean"].notna()
    ]["achieved_request_rate_mean"]
    rate_text = (
        f"{rag_rates.min():.3f}-{rag_rates.max():.3f} requests/s"
        if not rag_rates.empty
        else "unavailable"
    )
    failures = [row for row in rows if row["status"] != "complete"]
    bottleneck_counts: dict[str, int] = {}
    for reflection in reflections:
        for category in reflection["bottleneck_categories"]:
            bottleneck_counts[category] = bottleneck_counts.get(category, 0) + 1
    negative_reflections = sum(
        reflection.get("negative_experience", False) for reflection in reflections
    )
    positive_reflections = sum(
        reflection.get("positive_experience", False) for reflection in reflections
    )
    environment = rows[0]["runtime_fingerprint"]
    hardware = rows[0]["hardware_fingerprint"]
    (output_dir / "environment.json").write_text(
        json.dumps({"hardware": hardware, "runtime": environment}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# Experiment optimization report

This report is derived from the append-only SQLite ledger at
`{ledger_path.relative_to(PROJECT_ROOT)}`. Every value below comes from a
quality-gated local trial.

## Scope

The measured backend is `llama.cpp` on this Mac. The report does not claim
vLLM, SGLang, CUDA, NPU, distributed serving, or production SLA results.

Hardware: `{json.dumps(hardware, ensure_ascii=False)}`

Runtime: `{json.dumps(environment, ensure_ascii=False)}`

## Equal-budget strategy comparison

{_markdown_table(comparison)}

All strategies receive the same scored trial budget. The common baseline anchor
is outside the strategy budget and is used only for deterministic deltas.

The fixed-QPS `rag_reading` workload targeted 2.0 requests/s. Actual scheduled
arrival rates ranged from {rate_text}, so the open-loop scheduler is reported as
measured rather than assumed.

{agent_observation}

## Best-so-far

![Best-so-far](figures/best_so_far.png)

## Quality-passing Pareto frontier

{_markdown_table(pareto)}

![Pareto](figures/pareto.png)

## Prefix Cache ablation

The shared-prefix workload repeats one long document and changes only the final
question. `cache_n` and prompt-token counts are raw llama.cpp timing fields; a
missing or null field is marked invalid and excluded from the numerator. The hit
rate is `sum(valid cache_n) / sum(valid prompt_tokens)`.

{_markdown_table(cache_table)}

![Prefix Cache ablation](figures/cache_ablation.png)

TTFT reductions are request-level observations. They are reported separately
from `cache_n`, so a timing improvement is not presented as proof of cache reuse
unless the measured cache evidence is valid.

## Run integrity

- Trial records: {len(rows)}
- Quality-passing complete trials: {len(rows) - len(failures)}
- Failed or quality-rejected trials: {len(failures)}
- Structured reflections: {len(reflections)}
- Positive experiences: {positive_reflections}
- Negative memories: {negative_reflections}
- Bottleneck categories: `{json.dumps(bottleneck_counts, sort_keys=True)}`
- Similar-history evidence: `similar_history.json`
- Full per-trial metrics and intervals: `trial_metrics.csv`
- Reflection memory export: `reflections.json`
- Ledger export: `artifacts/experiments/<experiment_id>/export/trials.jsonl`

## Limitations

- Fixed quality probes verify exact answer, a critical fact, JSON structure, and
  timeout behavior. They do not replace the separate 100-question RAG/Agent
  benchmark.
- Peak RSS covers the model service process tree. It is not total system memory
  or GPU memory.
- Goodput uses the workload SLO and successful local responses.
- The RAG-shaped workload holds retrieved context fixed to isolate serving cost.

## Reproduction

```bash
./llm-bench --config configs/default.yaml experiment run \\
  --spec configs/experiments_p0_p1.yaml
./llm-bench --config configs/default.yaml experiment report \\
  --ledger {ledger_path.relative_to(PROJECT_ROOT)}
```
"""
    destination = output_dir / "experiment_report.md"
    destination.write_text(report, encoding="utf-8")
    return destination
