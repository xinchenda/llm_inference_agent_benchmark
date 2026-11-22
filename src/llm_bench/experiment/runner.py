from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from llm_bench.config import PROJECT_ROOT, load_models, resolve_path
from llm_bench.experiment.ledger import ExperimentLedger
from llm_bench.experiment.load import RequestRecord, run_workload
from llm_bench.experiment.metrics import (
    aggregate_workload_repeats,
    baseline_deltas,
    summarize_requests,
)
from llm_bench.experiment.policies import select_candidate
from llm_bench.experiment.quality import run_quality_gate
from llm_bench.experiment.reflection import reflect_trial
from llm_bench.experiment.runtime import (
    ResourceLock,
    allocate_port,
    classify_failure,
    retryable_failure,
    terminate_process_tree,
)
from llm_bench.experiment.types import (
    CandidateConfig,
    ExperimentSpec,
    TrialResult,
    canonical_json,
    load_experiment_spec,
    stable_hash,
)
from llm_bench.llm import LlamaServer, LocalLLM
from llm_bench.performance import PeakRSSMonitor


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() or "unknown"


def _llama_version() -> str:
    process = subprocess.run(
        ["llama-server", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (process.stdout or process.stderr).strip()


def _model_hashes() -> dict[str, str]:
    path = PROJECT_ROOT / "models" / "models.sha256"
    if not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if digest and name:
            hashes[Path(name).name] = digest
    return hashes


def _fingerprints() -> tuple[dict[str, Any], dict[str, Any]]:
    memory = psutil.virtual_memory()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "llama_server": _llama_version(),
    }
    hardware = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gb": round(memory.total / (1024**3), 3),
    }
    return runtime, hardware


def _provenance(
    spec: ExperimentSpec,
    candidate: CandidateConfig,
    model_path: Path,
) -> dict[str, str]:
    runtime, hardware = _fingerprints()
    hashes = _model_hashes()
    return {
        "git_commit": _git_commit(),
        "spec_hash": spec.spec_hash,
        "config_hash": stable_hash(
            {
                "candidate": candidate.as_dict(),
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            }
        ),
        "model_hash": hashes.get(model_path.name, "unavailable"),
        "runtime_fingerprint": canonical_json(runtime),
        "hardware_fingerprint": canonical_json(hardware),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_requests(path: Path, records: list[RequestRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


async def _warmup(llm: LocalLLM, requests: int) -> None:
    for index in range(requests):
        await asyncio.to_thread(
            llm.generate,
            "Reply briefly. /no_think",
            f"Warmup request {index}.",
            max_tokens=8,
            seed=index,
            disable_cache=True,
        )


async def _quality_and_performance(
    spec: ExperimentSpec,
    candidate: CandidateConfig,
    server: LlamaServer,
    llm: LocalLLM,
    trial_dir: Path,
) -> tuple[str, str | None, str | None, bool, dict[str, Any], dict[str, Any]]:
    await _warmup(llm, spec.warmup_requests)
    quality_passed, quality = await run_quality_gate(
        llm,
        spec.workloads,
        spec.request_timeout_s,
    )
    _write_json(trial_dir / "quality.json", quality)
    if not quality_passed:
        return (
            "quality_gate_failed",
            "quality_gate_failed",
            "One or more deterministic quality probes failed",
            False,
            quality,
            {},
        )

    performance: dict[str, Any] = {}
    for workload in spec.workloads:
        repeats = []
        for repeat in range(spec.repeats_per_trial):
            with PeakRSSMonitor(server) as monitor:
                records, wall_s = await run_workload(llm, workload)
            repeat_metrics = summarize_requests(
                workload,
                records,
                wall_s,
                monitor.peak_mb,
            )
            repeats.append(repeat_metrics)
            _write_requests(
                trial_dir / f"requests-{workload.id}-r{repeat}.jsonl",
                records,
            )
        performance[workload.id] = aggregate_workload_repeats(
            workload,
            repeats,
            resamples=spec.bootstrap_resamples,
            confidence=spec.confidence_level,
            seed=spec.seed + len(performance),
        )
        if not any(repeat["good_requests"] for repeat in repeats):
            _write_json(trial_dir / "metrics.json", performance)
            return (
                "failed",
                "request_failure",
                f"No quality-valid SLO response completed for workload {workload.id}",
                True,
                quality,
                performance,
            )
    _write_json(trial_dir / "metrics.json", performance)
    return "complete", None, None, True, quality, performance


async def execute_candidate_trial(
    spec: ExperimentSpec,
    candidate: CandidateConfig,
    strategy: str,
    sequence_no: int,
    baseline_performance: dict[str, Any] | None,
    output_root: Path,
) -> TrialResult:
    registry = load_models()["models"]
    if candidate.model_id not in registry:
        raise ValueError(f"Unknown model in registry: {candidate.model_id}")
    model = registry[candidate.model_id]
    model_path = PROJECT_ROOT / model["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    experiment_hash = spec.experiment_hash(strategy, sequence_no, 0, candidate.id)
    trial_id = f"trial-{experiment_hash[:20]}"
    trial_dir = output_root / "runs" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(spec, candidate, model_path)

    attempts = 0
    final: TrialResult | None = None
    while attempts <= spec.max_retries:
        attempts += 1
        port = allocate_port()
        server = LlamaServer(
            model_path=model_path,
            port=port,
            context_size=candidate.context_size,
            parallel=candidate.parallel,
            batch_size=candidate.batch_size,
            ubatch_size=candidate.ubatch_size,
            flash_attn=candidate.flash_attn,
            cache_type_k=candidate.cache_type_k,
            cache_type_v=candidate.cache_type_v,
            log_path=trial_dir / f"service-{attempts}.log",
        )
        try:
            server.start(timeout_s=int(spec.service_start_timeout_s))
            base_url = server.base_url
            llm = LocalLLM(
                base_url=base_url,
                model_name=candidate.model_id,
                timeout_s=spec.request_timeout_s,
                temperature=0.0,
                top_p=0.8,
                max_tokens=128,
                cache_prompt=candidate.cache_prompt,
            )
            (
                status,
                failure_type,
                failure_detail,
                quality_passed,
                quality,
                performance,
            ) = await _quality_and_performance(spec, candidate, server, llm, trial_dir)
            result = TrialResult(
                trial_id=trial_id,
                experiment_hash=experiment_hash,
                spec_id=spec.experiment_id,
                strategy=strategy,
                sequence_no=sequence_no,
                repeat_index=0,
                candidate=candidate,
                status=status,
                failure_type=failure_type,
                failure_detail=failure_detail,
                quality_passed=quality_passed,
                quality=quality,
                performance=performance,
                baseline_delta=baseline_deltas(performance, baseline_performance),
                provenance=provenance,
                artifact_path=str(trial_dir.relative_to(PROJECT_ROOT)),
                created_at=datetime.now(UTC).isoformat(),
            )
            final = result
            if status == "complete" or status == "quality_gate_failed":
                break
        except Exception as exc:
            failure_type = classify_failure(exc)
            detail = f"{type(exc).__name__}: {exc}"
            result = TrialResult(
                trial_id=trial_id,
                experiment_hash=experiment_hash,
                spec_id=spec.experiment_id,
                strategy=strategy,
                sequence_no=sequence_no,
                repeat_index=0,
                candidate=candidate,
                status="failed",
                failure_type=failure_type,
                failure_detail=detail,
                quality_passed=False,
                quality={},
                performance={},
                baseline_delta={"status": "unavailable"},
                provenance=provenance,
                artifact_path=str(trial_dir.relative_to(PROJECT_ROOT)),
                created_at=datetime.now(UTC).isoformat(),
            )
            final = result
            if attempts > spec.max_retries or not retryable_failure(failure_type):
                break
            await asyncio.sleep(0.5 * attempts)
        finally:
            terminate_process_tree(server.process)
            if server._log_handle is not None:
                server._log_handle.close()
                server._log_handle = None

    if final is None:
        raise RuntimeError("Trial produced no result")
    _write_json(trial_dir / "trial.json", final.as_dict())
    return final


async def run_experiment(spec: ExperimentSpec, config: dict[str, Any]) -> dict[str, Any]:
    registry = load_models()["models"]
    limits = {
        model_id: int(model.get("context_length", 32768))
        for model_id, model in registry.items()
    }
    spec.validate(limits)
    base_dir = resolve_path(config, "paths.artifacts") / "experiments" / spec.experiment_id
    base_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(base_dir / "ledger.sqlite")
    created_at = datetime.now(UTC).isoformat()
    ledger.register_spec(spec, created_at)

    summary: dict[str, Any] = {
        "spec_id": spec.experiment_id,
        "spec_hash": spec.spec_hash,
        "strategy_trials": {},
        "duplicates": 0,
        "started_at": created_at,
    }

    with ResourceLock(base_dir / "experiment.lock"):
        baseline_candidate = spec.default_candidate
        baseline_hash = spec.experiment_hash("baseline_anchor", 0, 0, baseline_candidate.id)
        existing = ledger.find_by_hash(baseline_hash)
        if existing:
            baseline_performance = existing["performance"]
        else:
            baseline = await execute_candidate_trial(
                spec,
                baseline_candidate,
                "baseline_anchor",
                0,
                None,
                base_dir,
            )
            ledger.append_trial(baseline)
            baseline_performance = baseline.performance if baseline.quality_passed else None
            baseline_reflection = reflect_trial(
                spec,
                baseline,
                [],
                baseline_performance,
            )
            ledger.append_reflection(baseline_reflection)
            _write_json(
                PROJECT_ROOT / baseline.artifact_path / "reflection.json",
                baseline_reflection,
            )

        if baseline_performance is None:
            raise RuntimeError("Baseline anchor failed its quality gate")

        completed_strategies = {
            strategy: ledger.strategy_history(spec.experiment_id, strategy)
            for strategy in spec.strategies
        }
        if all(
            len(history) >= spec.trials_per_strategy
            for history in completed_strategies.values()
        ):
            summary["status"] = "already_complete"
            export_path = base_dir / "export" / "trials.jsonl"
            summary["exported_trials"] = ledger.export_jsonl(
                export_path,
                spec.experiment_id,
            )
            summary["finished_at"] = datetime.now(UTC).isoformat()
            summary["ledger"] = str((base_dir / "ledger.sqlite").relative_to(PROJECT_ROOT))
            _write_json(base_dir / "run_summary.json", summary)
            return summary

        for strategy in spec.strategies:
            history = ledger.strategy_history(spec.experiment_id, strategy)
            memory = ledger.strategy_memory(spec.experiment_id, strategy)
            completed = 0
            sequence_no = len(history)
            while completed < spec.trials_per_strategy:
                selection = select_candidate(
                    spec,
                    strategy,
                    sequence_no,
                    history,
                    memory,
                )
                experiment_hash = spec.experiment_hash(
                    strategy,
                    sequence_no,
                    0,
                    selection.candidate.id,
                )
                existing = ledger.find_by_hash(experiment_hash)
                if existing:
                    summary["duplicates"] += 1
                    history = ledger.strategy_history(spec.experiment_id, strategy)
                    sequence_no += 1
                    completed += 1
                    continue
                result = await execute_candidate_trial(
                    spec,
                    selection.candidate,
                    strategy,
                    sequence_no,
                    baseline_performance,
                    base_dir,
                )
                result = TrialResult(
                    **{
                        **result.__dict__,
                        "quality": {
                            **result.quality,
                            "_policy": {
                                "reason": selection.reason,
                                "ucb_scores": selection.ucb_scores,
                                "skipped": selection.skipped or {},
                            },
                        },
                    }
                )
                ledger.append_trial(result)
                history = ledger.strategy_history(spec.experiment_id, strategy)
                reflection = reflect_trial(
                    spec,
                    result,
                    history,
                    baseline_performance,
                )
                ledger.append_reflection(reflection)
                _write_json(
                    PROJECT_ROOT / result.artifact_path / "reflection.json",
                    reflection,
                )
                memory = ledger.strategy_memory(spec.experiment_id, strategy)
                sequence_no += 1
                completed += 1
            summary["strategy_trials"][strategy] = completed

    export_path = base_dir / "export" / "trials.jsonl"
    summary["exported_trials"] = ledger.export_jsonl(export_path, spec.experiment_id)
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["ledger"] = str((base_dir / "ledger.sqlite").relative_to(PROJECT_ROOT))
    _write_json(base_dir / "run_summary.json", summary)
    return summary


async def run_experiment_from_file(
    spec_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return await run_experiment(load_experiment_spec(spec_path), config)
