from __future__ import annotations

# The backend must be selected before pyplot is imported.
# ruff: noqa: E402, I001

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import psutil

from llm_bench.config import PROJECT_ROOT, load_models, resolve_path


CONFIG_LABELS = {
    "direct_qwen3_0_6b_q8": "Direct 0.6B Q8",
    "direct_qwen3_1_7b_q4": "Direct 1.7B Q4",
    "direct_qwen3_1_7b_q8": "Direct 1.7B Q8",
    "rag_vector": "Vector RAG",
    "rag_hybrid": "Hybrid RAG",
    "rag_hybrid_rerank": "Hybrid + rerank",
    "agent_hybrid_rerank": "Controlled Agent",
    "agent_rule_rerank": "Rule-planner baseline",
}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def _public_text(text: str) -> str:
    return text.replace(str(PROJECT_ROOT), ".")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    version = subprocess.run(
        ["llama-server", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "llama_server": (version.stdout or version.stderr).strip(),
    }


def _quality_chart(summaries: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in summaries:
        label = CONFIG_LABELS.get(row["configuration"], row["configuration"])
        x = row.get("latency_p95_s") or 0
        y = row.get("answer_accuracy") or 0
        axis.scatter(x, y, s=55)
        axis.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("p95 end-to-end latency (s)")
    axis.set_ylabel("answer accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.set_title("Quality versus latency on 100 fixed questions")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _ablation_chart(summaries: list[dict[str, Any]], path: Path) -> None:
    names = [
        "rag_vector",
        "rag_hybrid",
        "rag_hybrid_rerank",
        "agent_hybrid_rerank",
        "agent_rule_rerank",
    ]
    selected = [row for name in names for row in summaries if row["configuration"] == name]
    labels = [CONFIG_LABELS[row["configuration"]] for row in selected]
    accuracy = [(row.get("answer_accuracy") or 0) * 100 for row in selected]
    recall = [(row.get("recall_at_5") or 0) * 100 for row in selected]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, accuracy, width, label="answer accuracy")
    axis.bar(x + width / 2, recall, width, label="Recall@5")
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylabel("percent")
    axis.set_ylim(0, 105)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_title("RAG and Agent ablation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _performance_charts(rows: list[dict[str, Any]], directory: Path) -> None:
    context_rows = [row for row in rows if row["run_kind"] == "context"]
    figure, axis = plt.subplots(figsize=(9, 5))
    for model_id in sorted({row["model_id"] for row in context_rows}):
        selected = sorted(
            [row for row in context_rows if row["model_id"] == model_id],
            key=lambda row: row["mean_prompt_tokens"] or 0,
        )
        axis.plot(
            [row["mean_prompt_tokens"] for row in selected],
            [row["latency_p95_s"] for row in selected],
            marker="o",
            label=model_id,
        )
    axis.set_xlabel("measured prompt tokens")
    axis.set_ylabel("p95 latency (s)")
    axis.set_title("Context length and quantization")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / "performance_context.png", dpi=180)
    plt.close(figure)

    concurrency_rows = [row for row in rows if row["run_kind"] == "concurrency"]
    if concurrency_rows:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model_id in sorted({row["model_id"] for row in concurrency_rows}):
            selected = sorted(
                [row for row in concurrency_rows if row["model_id"] == model_id],
                key=lambda row: row["concurrency"],
            )
            axis.plot(
                [row["concurrency"] for row in selected],
                [row["aggregate_output_tokens_per_second"] for row in selected],
                marker="o",
                label=model_id,
            )
        axis.set_xlabel("concurrent requests")
        axis.set_ylabel("aggregate output tokens/s")
        axis.set_title("Concurrency scaling at a 256-word prompt")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "performance_concurrency.png", dpi=180)
        plt.close(figure)


def _write_quality_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "configuration",
        "model_id",
        "answer_accuracy",
        "task_success_rate",
        "recall_at_5",
        "citation_precision",
        "citation_support_rate",
        "faithfulness_overlap",
        "negation_reversal_rate",
        "tool_selection_accuracy",
        "latency_p50_s",
        "latency_p95_s",
        "ttft_p50_s",
        "server_cold_start_s",
        "server_rss_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})


def _collect_badcases(eval_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*/results.jsonl")):
        configuration = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("failure_type"):
                rows.append({"configuration": configuration, **record})
    priority = {
        "hallucinated_when_unanswerable": 0,
        "wrong_tool": 1,
        "retrieval_miss": 2,
        "citation_missing": 3,
        "answer_incorrect": 4,
        "runtime_or_tool_error": 5,
    }
    rows.sort(key=lambda row: priority.get(row["failure_type"], 99))
    return rows[:limit]


def _manual_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not path.exists():
        return {"samples": 0, "status": "missing"}, []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        "samples": len(rows),
        "correct": sum(row.get("review_label") == "correct" for row in rows),
        "partially_correct": sum(row.get("review_label") == "partially_correct" for row in rows),
        "incorrect": sum(row.get("review_label") == "incorrect" for row in rows),
    }
    return result, rows


def _config_row(row: dict[str, Any]) -> str:
    label = CONFIG_LABELS.get(row["configuration"], row["configuration"])
    return (
        f"| {label} | {_pct(row.get('answer_accuracy'))} | "
        f"{_pct(row.get('task_success_rate'))} | {_pct(row.get('recall_at_5'))} | "
        f"{_pct(row.get('citation_support_rate'))} | "
        f"{_pct(row.get('faithfulness_overlap'))} | "
        f"{_pct(row.get('negation_reversal_rate'))} | "
        f"{_pct(row.get('tool_selection_accuracy'))} | "
        f"{_seconds(row.get('latency_p95_s'))} |"
    )


def _key_observations(summaries: list[dict[str, Any]]) -> list[str]:
    lookup = {row["configuration"]: row for row in summaries}
    observations: list[str] = []
    vector = lookup.get("rag_vector")
    hybrid = lookup.get("rag_hybrid_rerank")
    if vector and hybrid:
        delta = (hybrid.get("answer_accuracy") or 0) - (vector.get("answer_accuracy") or 0)
        observations.append(
            "Hybrid retrieval plus reranking changed answer accuracy by "
            f"{delta * 100:+.1f} percentage points versus dense retrieval."
        )
    rag = lookup.get("rag_hybrid_rerank")
    agent = lookup.get("agent_hybrid_rerank")
    if rag and agent:
        observations.append(
            "The controlled Agent achieved tool-selection accuracy of "
            f"{_pct(agent.get('tool_selection_accuracy'))}; hybrid RAG citation "
            f"support was {_pct(rag.get('citation_support_rate'))}."
        )
    q4 = lookup.get("direct_qwen3_1_7b_q4")
    q8 = lookup.get("direct_qwen3_1_7b_q8")
    if q4 and q8:
        observations.append(
            "Direct 1.7B Q8 minus Q4 answer-accuracy difference: "
            f"{((q8.get('answer_accuracy') or 0) - (q4.get('answer_accuracy') or 0)) * 100:+.1f} "
            "percentage points. This is descriptive on 100 questions, not a universal "
            "quantization-quality claim."
        )
    return observations


def build_reports(config: dict[str, Any]) -> Path:
    artifacts = resolve_path(config, "paths.artifacts")
    eval_dir = artifacts / "eval"
    perf_dir = artifacts / "perf"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_json(eval_dir / "all_summaries.json")
    performance = _load_json(perf_dir / "results.json")
    environment = _environment()
    (reports_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_quality_csv(summaries, reports_dir / "quality_summary.csv")
    _quality_chart(summaries, figures_dir / "quality_vs_latency.png")
    _ablation_chart(summaries, figures_dir / "rag_ablation.png")
    _performance_charts(performance, figures_dir)
    badcases = _collect_badcases(eval_dir)
    (eval_dir / "badcases.json").write_text(
        json.dumps(badcases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manual_summary, manual_rows = _manual_summary(eval_dir / "human_review.csv")

    model_registry = load_models()
    models = model_registry["models"]
    model_rows = []
    for model_id, model in models.items():
        path = PROJECT_ROOT / model["file"]
        model_rows.append(
            {
                **model,
                "model_id": model_id,
                "local_bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
        )
    for model_id, model in model_registry.get("embedding_models", {}).items():
        path = PROJECT_ROOT / model["file"]
        model_rows.append(
            {
                **model,
                "model_id": model_id,
                "local_bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
        )
    (reports_dir / "model_manifest.json").write_text(
        json.dumps(model_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = resolve_path(config, "paths.manifest")
    corpus = _load_json(manifest_path)
    local_documents = list(resolve_path(config, "paths.corpus").glob("local__*.md"))
    total_documents = len(corpus) + len(local_documents)
    corpus_failures = [
        item
        for item in corpus
        if _sha256(resolve_path(config, "paths.corpus") / item["file"]) != item["sha256"]
    ]
    observations = _key_observations(summaries)
    quality_table = "\n".join(_config_row(row) for row in summaries)
    quality_header = (
        "| Configuration | Answer accuracy | Task success | Recall@5 | Citation support | "
        "Faithfulness overlap | Negation reversal | Tool selection | p95 latency |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    perf_context = [
        row
        for row in performance
        if row["run_kind"] == "context" and row["requested_prompt_words"] == 256
    ]
    perf_context_table = "\n".join(
        "| {model} | {model_mb:.0f} | {prompt:.0f} | {ttft} | {p95} | "
        "{tps:.1f} | {rss:.0f} |".format(
            model=row["model_id"],
            model_mb=row["model_file_mb"],
            prompt=row["mean_prompt_tokens"] or 0,
            ttft=_seconds(row["ttft_p95_s"]),
            p95=_seconds(row["latency_p95_s"]),
            tps=row["mean_request_tokens_per_second"],
            rss=row["peak_process_rss_mb"],
        )
        for row in perf_context
    )
    performance_header = (
        "| Model | File MiB | Prompt tokens | TTFT p95 | Latency p95 | "
        "Output tokens/s | Peak process RSS MiB |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    concurrency_rows = [row for row in performance if row["run_kind"] == "concurrency"]
    concurrency_table = "\n".join(
        f"| {row['model_id']} | {row['concurrency']} | "
        f"{row['peak_process_rss_mb']:.0f} | {_seconds(row['latency_p95_s'])} | "
        f"{row['aggregate_output_tokens_per_second']:.1f} |"
        for row in concurrency_rows
    )
    badcase_preview = "\n\n".join(
        f"### {row['configuration']} / {row['question_id']} — {row['failure_type']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Expected documents: {', '.join(row['expected_docs']) or 'none'}\n\n"
        f"Answer: {_public_text(row['answer'][:700]) or '(empty)'}\n\n"
        f"Retrieved: {', '.join(row['retrieved_document_ids']) or 'none'}"
        for row in badcases[:8]
    )
    manual_table = "\n".join(
        f"| {row['question_id']} | {row.get('review_label', '')} | "
        f"{row.get('factual_support', '')} | {row.get('citation_ok', '')} | "
        f"{row.get('badcase_note', '')} |"
        for row in manual_rows
    )

    report = f"""# Validation report

Generated: {datetime.now(UTC).isoformat()}

## Scope

Pretrained open-weight model inference, cited RAG, controlled Agent tools,
quality evaluation, and local performance/cost measurement. No model training,
fine-tuning, image model, ISP, or computer-vision quality work is included.

## Hardware and software

- {environment["platform"]}
- {environment["logical_cores"]} logical cores, {environment["memory_gb"]:.1f} GB memory
- `{environment["llama_server"]}`
- Python 3.12, `sentence-transformers`, PyTorch MPS

## Data and model provenance

- {total_documents} documents: {len(corpus)} pinned public files plus
  {len(local_documents)} project-authored fixtures; hash failures:
  {len(corpus_failures)}
- Three Qwen3 GGUF model configurations, Apache-2.0
- Exact local model sizes and SHA256 values: `reports/model_manifest.json`
- Document source URLs, commits, licenses, and hashes:
  `data/corpus/source_manifest.json`

## Quality on 100 fixed questions

{quality_header}
{quality_table}

![Quality versus latency](figures/quality_vs_latency.png)

![RAG ablation](figures/rag_ablation.png)

### Observations

{chr(10).join(f"- {item}" for item in observations)}

## Model/quantization performance at a 256-word prompt

{performance_header}
{perf_context_table}

![Context scaling](figures/performance_context.png)

![Concurrency scaling](figures/performance_concurrency.png)

| Model | Concurrent requests | Peak RSS MiB | p95 latency | Aggregate output tokens/s |
|---|---:|---:|---:|---:|
{concurrency_table}

All performance rows discard two warm-ups and fix output generation at
{config["perf"]["output_tokens"]} tokens. Peak RSS is the measured
`llama-server` process plus children and excludes the separate Python/FastAPI
process. The local incremental service cost is USD 0 for open-source software;
hardware purchase, electricity, operations labor, and cloud alternatives are
not included.

## Manual review

Automated answer scoring is a required-concept proxy, not proof of semantic
correctness. `artifacts/eval/human_review.csv` is the retained manual label
source.

Summary: {manual_summary}

| Question | Label | Factual support | Citation OK | Badcase note |
|---|---|---|---|---|
{manual_table}

## Representative badcases

{badcase_preview or "No badcase was produced by the selected configuration."}

## Reproduce

```bash
./llm-bench corpus build --refresh
./llm-bench index build
./llm-bench eval run
./llm-bench perf run
./llm-bench report build
```

## Unsupported paths

- vLLM and SGLang were not executed on this Apple M4; their intended use is a
  Linux/NVIDIA serving path.
- CUDA and NPU adaptation were not executed.
- The Linux aarch64 image was built and its CLI/environment entrypoint was
  executed successfully. Model weights remain mounted at runtime rather than
  baked into the image.
"""
    report_path = reports_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
