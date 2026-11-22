from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import uvicorn

from llm_bench.config import (
    PROJECT_ROOT,
    apply_overrides,
    load_config,
    load_models,
    resolve_path,
)
from llm_bench.evaluation import recompute_text_metrics, run_evaluation
from llm_bench.experiment.report import build_experiment_report
from llm_bench.experiment.runner import run_experiment_from_file
from llm_bench.llm import LlamaServer, LocalLLM, choose_device
from llm_bench.performance import run_performance
from llm_bench.reporting import build_reports
from llm_bench.retrieval import KnowledgeIndex, build_index
from llm_bench.service import create_app


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return apply_overrides(load_config(args.config), getattr(args, "set", None))


def command_corpus(args: argparse.Namespace) -> None:
    config = _config(args)
    corpus_dir = resolve_path(config, "paths.corpus")
    manifest = resolve_path(config, "paths.manifest")
    script = PROJECT_ROOT / "scripts" / "fetch_corpus.py"
    if args.refresh:
        subprocess.run([sys.executable, str(script)], check=True)
    documents = list(corpus_dir.glob("*.md"))
    if len(documents) < 50:
        raise SystemExit("Corpus has fewer than 50 documents; run with --refresh")
    print(
        json.dumps(
            {
                "documents": len(documents),
                "corpus_dir": str(corpus_dir),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


def command_build_index(args: argparse.Namespace) -> None:
    config = _config(args)
    retrieval = config["retrieval"]
    index = build_index(
        corpus_dir=resolve_path(config, "paths.corpus"),
        index_dir=resolve_path(config, "paths.index"),
        manifest_path=resolve_path(config, "paths.manifest"),
        chunk_size=int(retrieval["chunk_size"]),
        chunk_overlap=int(retrieval["chunk_overlap"]),
        embedding_model=str(retrieval["embedding_model"]),
        reranker_model=str(retrieval["reranker_model"]),
        device=choose_device(str(retrieval["device"])),
    )
    print(
        json.dumps(
            {
                "documents": len({chunk.document_id for chunk in index.chunks}),
                "chunks": len(index.chunks),
                "index": str(resolve_path(config, "paths.index")),
            },
            indent=2,
        )
    )


def command_eval(args: argparse.Namespace) -> None:
    summaries = asyncio.run(run_evaluation(_config(args)))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def command_perf(args: argparse.Namespace) -> None:
    rows = asyncio.run(run_performance(_config(args)))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def command_experiment_run(args: argparse.Namespace) -> None:
    config = _config(args)
    summary = asyncio.run(
        run_experiment_from_file(
            Path(args.spec),
            config,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_experiment_report(args: argparse.Namespace) -> None:
    config = _config(args)
    ledger = Path(args.ledger)
    if not ledger.is_absolute():
        ledger = PROJECT_ROOT / ledger
    output_dir = (
        Path(args.output)
        if args.output
        else resolve_path(config, "paths.artifacts").parent / "reports" / "experiments"
    )
    print(build_experiment_report(ledger, output_dir))


def command_recompute_metrics(args: argparse.Namespace) -> None:
    config = _config(args)
    index = KnowledgeIndex.load(
        resolve_path(config, "paths.index"),
        device=choose_device(str(config["retrieval"]["device"])),
    )
    summaries = recompute_text_metrics(
        resolve_path(config, "paths.artifacts") / "eval",
        index,
        resolve_path(config, "paths.questions"),
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def command_serve(args: argparse.Namespace) -> None:
    config = _config(args)
    model_id = args.model or str(config["llm"]["active_model"])
    registry = load_models()["models"]
    if model_id not in registry:
        raise SystemExit(f"Unknown model: {model_id}")
    model = registry[model_id]
    device = choose_device(str(config["retrieval"]["device"]))
    index = KnowledgeIndex.load(resolve_path(config, "paths.index"), device=device)
    server = LlamaServer(
        model_path=PROJECT_ROOT / model["file"],
        port=8080,
        context_size=int(config["llm"]["context_size"]),
        parallel=2,
        log_path=resolve_path(config, "paths.artifacts") / "runtime" / "llama-server.log",
    )
    server.start()
    llm = LocalLLM(
        base_url=server.base_url,
        model_name=model_id,
        api_key=str(config["llm"]["api_key"]),
        timeout_s=float(config["llm"]["request_timeout_s"]),
        temperature=float(config["llm"]["temperature"]),
        top_p=float(config["llm"]["top_p"]),
        max_tokens=int(config["llm"]["max_tokens"]),
    )
    app = create_app(config, llm, index, default_model_id=model_id)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        server.stop()


def command_models(args: argparse.Namespace) -> None:
    registry = load_models()["models"]
    output = []
    for model in registry.values():
        path = PROJECT_ROOT / model["file"]
        output.append(
            {
                **model,
                "exists": path.exists(),
                "local_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def command_environment(args: argparse.Namespace) -> None:
    import psutil

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "llama_server": (lambda result: (result.stdout or result.stderr).strip())(
            subprocess.run(
                ["llama-server", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-bench")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[])
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("corpus", help="Validate or rebuild the document corpus")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_sub.add_parser("build")
    corpus_build.add_argument("--refresh", action="store_true")
    corpus_build.set_defaults(func=command_corpus)

    questions = subparsers.add_parser("questions", help="Question-set utilities")
    questions_sub = questions.add_subparsers(dest="questions_command", required=True)
    questions_build = questions_sub.add_parser("build")

    def build_questions(_: argparse.Namespace) -> None:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_questions.py")],
            check=True,
        )

    questions_build.set_defaults(func=build_questions)

    index = subparsers.add_parser("index", help="Build the retrieval index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_build = index_sub.add_parser("build")
    index_build.set_defaults(func=command_build_index)

    evaluation = subparsers.add_parser("eval", help="Run quality evaluations")
    evaluation_sub = evaluation.add_subparsers(dest="eval_command", required=True)
    evaluation_run = evaluation_sub.add_parser("run")
    evaluation_run.set_defaults(func=command_eval)

    performance = subparsers.add_parser("perf", help="Run performance benchmarks")
    performance_sub = performance.add_subparsers(dest="perf_command", required=True)
    performance_run = performance_sub.add_parser("run")
    performance_run.set_defaults(func=command_perf)

    experiment = subparsers.add_parser("experiment", help="Run optimization experiments")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_run = experiment_sub.add_parser("run")
    experiment_run.add_argument("--spec", default="configs/experiments_p0_p1.yaml")
    experiment_run.set_defaults(func=command_experiment_run)
    experiment_report = experiment_sub.add_parser("report")
    experiment_report.add_argument(
        "--ledger",
        default="artifacts/experiments/p0_p1_local_20261122/ledger.sqlite",
    )
    experiment_report.add_argument("--output")
    experiment_report.set_defaults(func=command_experiment_report)

    metrics = subparsers.add_parser("metrics", help="Recompute text-only metrics")
    metrics_sub = metrics.add_subparsers(dest="metrics_command", required=True)
    metrics_recompute = metrics_sub.add_parser("recompute")
    metrics_recompute.set_defaults(func=command_recompute_metrics)

    report = subparsers.add_parser("report", help="Build charts and handoff reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)

    def command_report(params: argparse.Namespace) -> None:
        print(build_reports(_config(params)))

    report_build = report_sub.add_parser("build")
    report_build.set_defaults(func=command_report)

    serve = subparsers.add_parser("serve", help="Start FastAPI and llama.cpp")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model")
    serve.set_defaults(func=command_serve)

    models = subparsers.add_parser("models", help="Inspect the model registry")
    models.set_defaults(func=command_models)
    environment = subparsers.add_parser("environment")
    environment.set_defaults(func=command_environment)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
