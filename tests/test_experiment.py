from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from llm_bench.experiment import runner as runner_module
from llm_bench.experiment.analysis import cache_ablation_pairs
from llm_bench.experiment.ledger import ExperimentLedger
from llm_bench.experiment.load import RequestRecord, cache_observation, run_workload
from llm_bench.experiment.metrics import (
    aggregate_workload_repeats,
    bootstrap_interval,
    summarize_requests,
)
from llm_bench.experiment.policies import select_candidate
from llm_bench.experiment.quality import evaluate_quality_output, run_quality_gate
from llm_bench.experiment.reflection import reflect_trial
from llm_bench.experiment.runtime import (
    ResourceLock,
    allocate_port,
    classify_failure,
    port_is_free,
    terminate_process_tree,
)
from llm_bench.experiment.types import (
    CandidateConfig,
    ExperimentSpec,
    QualitySpec,
    TrialResult,
    WorkloadSpec,
    canonical_json,
    load_experiment_spec,
)
from llm_bench.llm import LlamaServer, LLMResponse, LocalLLM


def make_spec(*, repeats: int = 2) -> ExperimentSpec:
    quality = QualitySpec(prompt="Reply OK", expected_contains=("ok",))
    workloads = (
        WorkloadSpec("interactive", 8, 4, 2, "closed", 1, 2.0, quality),
        WorkloadSpec(
            "rag_reading",
            16,
            4,
            2,
            "open_qps",
            2,
            5.0,
            QualitySpec(prompt="Limit is 42?", expected_contains=("42",)),
            target_qps=4.0,
        ),
        WorkloadSpec(
            "batch_generation",
            12,
            8,
            2,
            "closed",
            2,
            10.0,
            QualitySpec(
                prompt='Return {"status":"ok"}',
                expected_json={"status": "ok"},
            ),
        ),
        WorkloadSpec(
            "prefix_cache_rag",
            12,
            4,
            2,
            "closed",
            1,
            10.0,
            QualitySpec(prompt="Limit is 42?", expected_contains=("42",)),
        ),
    )
    return ExperimentSpec(
        schema_version=1,
        experiment_id="test-experiment",
        seed=42,
        baseline_candidate="a",
        candidates=(
            CandidateConfig("a", "model-a", 2048, 1),
            CandidateConfig("b", "model-b", 2048, 2),
        ),
        workloads=workloads,
        strategies=("default", "grid", "random", "ucb1", "agent"),
        trials_per_strategy=3,
        repeats_per_trial=repeats,
        warmup_requests=0,
        max_retries=1,
        service_start_timeout_s=5,
        request_timeout_s=5,
        bootstrap_resamples=100,
        confidence_level=0.95,
    )


def make_trial(spec: ExperimentSpec, candidate: CandidateConfig, suffix: str) -> TrialResult:
    performance = {
        workload.id: {
            "workload_id": workload.id,
            "stability": {
                "goodput_requests_per_second": {"mean": 4.0},
                "latency_p95_s": {"mean": 1.0},
                "peak_process_tree_rss_mb": {"mean": 100.0},
            },
        }
        for workload in spec.workloads
    }
    return TrialResult(
        trial_id=f"trial-{suffix}",
        experiment_hash=spec.experiment_hash("default", 0, 0, candidate.id) + suffix,
        spec_id=spec.experiment_id,
        strategy="default",
        sequence_no=0,
        repeat_index=0,
        candidate=candidate,
        status="complete",
        failure_type=None,
        failure_detail=None,
        quality_passed=True,
        quality={},
        performance=performance,
        baseline_delta={},
        provenance={
            "git_commit": "abc",
            "spec_hash": spec.spec_hash,
            "config_hash": "cfg",
            "model_hash": "model",
            "runtime_fingerprint": canonical_json({"python": "3.12"}),
            "hardware_fingerprint": canonical_json({"memory_gb": 16}),
        },
        artifact_path="artifacts/trial",
        created_at="2026-11-22T00:00:00+00:00",
    )


def test_config_validation_rejects_missing_fixed_loads() -> None:
    spec = make_spec()
    invalid = ExperimentSpec(
        **{
            **spec.__dict__,
            "workloads": spec.workloads[:2],
        }
    )
    with pytest.raises(ValueError, match="workloads must be exactly"):
        invalid.validate()


def test_candidate_rejects_invalid_batch_and_cache_configuration() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CandidateConfig(
            "bad-batch",
            "model-a",
            4096,
            2,
            batch_size=512,
            ubatch_size=96,
        ).validate()
    with pytest.raises(ValueError, match="quantized KV cache"):
        CandidateConfig(
            "bad-cache",
            "model-a",
            4096,
            2,
            flash_attn="off",
            cache_type_k="f16",
            cache_type_v="q8_0",
        ).validate()


def test_llama_server_passes_optimization_flags(tmp_path: Path) -> None:
    server = LlamaServer(
        model_path=tmp_path / "model.gguf",
        port=18080,
        context_size=8192,
        parallel=4,
        batch_size=512,
        ubatch_size=128,
        flash_attn="on",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    command = server.command()
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--ubatch-size") + 1] == "128"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert command[command.index("--parallel") + 1] == "4"


def test_cache_prompt_request_field_is_explicitly_transmitted() -> None:
    enabled = LocalLLM("http://127.0.0.1:1/v1", "fake", cache_prompt=True)
    disabled = LocalLLM("http://127.0.0.1:1/v1", "fake", cache_prompt=False)
    kwargs = {
        "system_prompt": "system",
        "user_prompt": "user",
        "max_tokens": 8,
        "seed": 42,
        "disable_cache": False,
        "cache_prompt": None,
    }
    assert enabled.build_payload(**kwargs)["cache_prompt"] is True
    assert disabled.build_payload(**kwargs)["cache_prompt"] is False


def test_cache_n_requires_explicit_valid_timing_value() -> None:
    assert cache_observation({"cache_n": 80}, 120) == (80, 120, 2 / 3)
    assert cache_observation({}, 120) == (None, 120, None)
    assert cache_observation({"cache_n": None}, 120) == (None, 120, None)
    assert cache_observation({"cache_n": 0}, 0) == (0, None, None)


def test_cache_metrics_mark_missing_values_invalid_and_handle_zero_denominator() -> None:
    workload = next(
        item for item in make_spec().workloads if item.id == "prefix_cache_rag"
    )
    records = [
        RequestRecord(
            index=0,
            scheduled_s=0,
            started_s=0,
            latency_s=0.1,
            ttft_s=0.05,
            tpot_s=0.01,
            prompt_tokens=100,
            completion_tokens=4,
            success=True,
            slo_met=True,
            error_type=None,
            error=None,
            text="ok",
            cache_n=60,
            cache_prompt_tokens=100,
            cache_hit_rate=0.6,
        ),
        RequestRecord(
            index=1,
            scheduled_s=0,
            started_s=0,
            latency_s=0.1,
            ttft_s=0.05,
            tpot_s=0.01,
            prompt_tokens=0,
            completion_tokens=4,
            success=True,
            slo_met=True,
            error_type=None,
            error=None,
            text="ok",
            cache_n=0,
            cache_prompt_tokens=None,
            cache_hit_rate=None,
        ),
    ]
    summary = summarize_requests(workload, records, wall_s=0.2, peak_rss_mb=100)
    assert summary["cache_hit_rate"] == 0.6
    assert summary["cache_evidence_valid"] is False
    assert summary["invalid_cache_observations"] == 1


def test_cache_ablation_pairs_compute_ttft_and_hit_rate_changes() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "model_id": "model-a",
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "context_size": 4096,
                "parallel": 1,
                "batch_size": 512,
                "ubatch_size": 128,
                "flash_attn": "auto",
                "cache_prompt": True,
                "cache_hit_rate": 0.8,
                "cache_evidence_valid": True,
                "invalid_cache_observations": 0,
                "ttft_p50_s": 0.1,
                "ttft_p95_s": 0.2,
                "goodput_requests_per_second": 5.0,
                "peak_process_tree_rss_mb": 1000.0,
                "error_rate": 0.0,
            },
            {
                "model_id": "model-a",
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "context_size": 4096,
                "parallel": 1,
                "batch_size": 512,
                "ubatch_size": 128,
                "flash_attn": "auto",
                "cache_prompt": False,
                "cache_hit_rate": 0.0,
                "cache_evidence_valid": True,
                "invalid_cache_observations": 0,
                "ttft_p50_s": 0.5,
                "ttft_p95_s": 1.0,
                "goodput_requests_per_second": 4.0,
                "peak_process_tree_rss_mb": 1000.0,
                "error_rate": 0.0,
            },
        ]
    )
    pairs = cache_ablation_pairs(rows)
    assert len(pairs) == 1
    assert pairs.iloc[0]["ttft_p95_reduction_pct"] == pytest.approx(80.0)
    assert pairs.iloc[0]["goodput_change_pct"] == pytest.approx(25.0)


def test_ledger_is_append_only_and_duplicate_is_idempotent(tmp_path: Path) -> None:
    spec = make_spec()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite")
    assert ledger.register_spec(spec, "2026-11-22T00:00:00+00:00")
    assert not ledger.register_spec(spec, "2026-11-22T00:00:00+00:00")
    trial = make_trial(spec, spec.default_candidate, "one")
    assert ledger.append_trial(trial)
    assert not ledger.append_trial(trial)
    assert len(ledger.all_trials()) == 1
    with sqlite3.connect(tmp_path / "ledger.sqlite") as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE trials SET status = 'changed'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM trials")


def test_ledger_batch_rolls_back_on_duplicate(tmp_path: Path) -> None:
    spec = make_spec()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite")
    ledger.register_spec(spec, "2026-11-22T00:00:00+00:00")
    first = make_trial(spec, spec.default_candidate, "one")
    second = make_trial(spec, spec.candidates[1], "two")
    ledger.append_trial(first)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append_many([second, first])
    assert [row["trial_id"] for row in ledger.all_trials()] == [first.trial_id]


def test_ledger_persists_structured_reflection_memory(tmp_path: Path) -> None:
    spec = make_spec()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite")
    ledger.register_spec(spec, "2026-11-22T00:00:00+00:00")
    trial = make_trial(spec, spec.default_candidate, "memory")
    trial = TrialResult(
        **{
            **trial.__dict__,
            "strategy": "agent",
            "status": "quality_gate_failed",
            "quality_passed": False,
            "quality": {"interactive": {"passed": False, "failures": ["bad"]}},
            "performance": {},
        }
    )
    ledger.append_trial(trial)
    reflection = reflect_trial(spec, trial, [], None)
    assert ledger.append_reflection(reflection)
    assert not ledger.append_reflection(reflection)
    memory = ledger.strategy_memory(spec.experiment_id, "agent")
    assert memory[0]["negative_experience"] is True
    assert memory[0]["bottleneck_categories"] == ["quality_gate_failure"]


def test_port_occupancy_and_allocator_failure() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        assert not port_is_free(port)
        with pytest.raises(OSError, match="No free port"):
            allocate_port(port, port)


def test_failure_classification_covers_startup_timeout_oom_crash_and_config() -> None:
    assert classify_failure(RuntimeError("llama-server exited during startup")) == (
        "service_start_failure"
    )
    assert classify_failure(TimeoutError("health deadline")) == "timeout"
    assert classify_failure(RuntimeError("out of memory while loading weights")) == "oom"
    assert classify_failure(RuntimeError("connection reset by peer")) == "process_crash"
    assert classify_failure(ValueError("invalid context_size")) == "invalid_config"


@pytest.mark.asyncio
async def test_service_start_failure_is_recorded_after_one_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = make_spec()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model-a.gguf").write_bytes(b"model")
    attempts = 0

    class FailingServer:
        process = None
        _log_handle = None

        def __init__(self, **_: object) -> None:
            pass

        def start(self, **_: object) -> None:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("llama-server exited during startup")

    monkeypatch.setattr(runner_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner_module,
        "load_models",
        lambda: {"models": {"model-a": {"file": "models/model-a.gguf"}}},
    )
    monkeypatch.setattr(runner_module, "LlamaServer", FailingServer)
    monkeypatch.setattr(runner_module, "allocate_port", lambda: 18080)
    result = await runner_module.execute_candidate_trial(
        spec,
        spec.default_candidate,
        "default",
        0,
        None,
        tmp_path / "experiment",
    )
    assert attempts == 2
    assert result.status == "failed"
    assert result.failure_type == "service_start_failure"


@pytest.mark.asyncio
async def test_no_good_performance_request_fails_trial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = make_spec()

    class FakeServer:
        rss_mb = 100.0

    class FakeLLM:
        timeout_s = 1

        def generate(self, *_: object, **__: object) -> LLMResponse:
            return LLMResponse("ok", 0.01, 0.005, 1, 1, {})

    failed = RequestRecord(
        index=0,
        scheduled_s=0,
        started_s=0,
        latency_s=0.1,
        ttft_s=None,
        tpot_s=None,
        prompt_tokens=None,
        completion_tokens=None,
        success=False,
        slo_met=False,
        error_type="timeout",
        error="timeout",
        text="",
    )

    async def passing_quality(*_: object, **__: object):
        return True, {}

    async def failed_load(*_: object, **__: object):
        return [failed], 0.1

    monkeypatch.setattr(runner_module, "run_quality_gate", passing_quality)
    monkeypatch.setattr(runner_module, "run_workload", failed_load)
    status, failure, _, quality_passed, _, performance = (
        await runner_module._quality_and_performance(
            spec,
            spec.default_candidate,
            FakeServer(),
            FakeLLM(),
            tmp_path,
        )
    )
    assert status == "failed"
    assert failure == "request_failure"
    assert quality_passed is True
    assert performance


def test_resource_lock_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "resource.lock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,time; "
                f"h=open({str(path)!r},'w'); "
                "fcntl.flock(h,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(5)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(TimeoutError):
            with ResourceLock(path, timeout_s=0.15):
                pass
    finally:
        terminate_process_tree(process)


def test_process_tree_cleanup_kills_children() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(p.pid,flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        terminate_process_tree(process)
        time.sleep(0.1)
        assert process.poll() is not None
        assert not psutil.pid_exists(child_pid)
    finally:
        terminate_process_tree(process)


def test_quality_checks_exact_fact_json_and_degradation() -> None:
    assert evaluate_quality_output(
        QualitySpec("x", expected_exact="READY"),
        " READY. ",
    )[0]
    assert evaluate_quality_output(
        QualitySpec("x", expected_contains=("42",)),
        "The limit is 42.",
    )[0]
    assert evaluate_quality_output(
        QualitySpec("x", expected_json={"status": "ok"}),
        '{"status":"ok"}',
    )[0]
    passed, reasons = evaluate_quality_output(
        QualitySpec("x", expected_exact="READY"),
        "NOT READY",
    )
    assert not passed
    assert reasons


def test_reflection_records_quality_failure_as_negative_memory() -> None:
    spec = make_spec()
    failed = make_trial(spec, spec.default_candidate, "quality-failure")
    failed = TrialResult(
        **{
            **failed.__dict__,
            "strategy": "agent",
            "status": "quality_gate_failed",
            "failure_type": "quality_gate_failed",
            "quality_passed": False,
            "quality": {
                "interactive": {
                    "passed": False,
                    "failures": ["expected READY"],
                }
            },
            "performance": {},
        }
    )
    insight = reflect_trial(spec, failed, [], None)
    assert insight["negative_experience"] is True
    assert insight["positive_experience"] is False
    assert "quality_gate_failure" in insight["bottleneck_categories"]
    assert insight["recommended_candidate_ids"]


def test_rsi_reflection_reports_ttft_queueing_and_rss() -> None:
    spec = make_spec()
    trial = make_trial(spec, spec.default_candidate, "reflection")
    performance = trial.performance
    for workload in performance.values():
        workload["stability"]["ttft_p95_s"] = {"mean": 4.0}
        workload["stability"]["latency_p95_s"] = {"mean": 8.0}
        workload["stability"]["peak_process_tree_rss_mb"] = {"mean": 2000.0}
    trial = TrialResult(**{**trial.__dict__, "performance": performance})
    insight = reflect_trial(spec, trial, [], None)
    assert "ttft_bottleneck" in insight["bottleneck_categories"]
    assert "throughput_queueing" in insight["bottleneck_categories"]
    assert "rss_memory_pressure" in insight["bottleneck_categories"]


@pytest.mark.asyncio
async def test_quality_gate_returns_structured_timeout_failure() -> None:
    class SlowLLM:
        def generate(self, *_: object, **__: object) -> LLMResponse:
            time.sleep(0.1)
            return LLMResponse("", 0.1, 0.01, 1, 1, {})

    spec = make_spec()
    passed, results = await run_quality_gate(
        SlowLLM(),
        spec.workloads,
        timeout_s=0.01,
    )
    assert not passed
    assert all(item["error_type"] == "timeout" for item in results.values())


@pytest.mark.asyncio
async def test_open_qps_load_records_target_and_achieved_rate() -> None:
    class FastLLM:
        timeout_s = 1

        def generate(self, *_: object, **__: object) -> LLMResponse:
            return LLMResponse("ok", 0.02, 0.01, 4, 2, {})

    workload = WorkloadSpec(
        "rag_reading",
        4,
        2,
        4,
        "open_qps",
        2,
        1.0,
        QualitySpec("fact 42", expected_contains=("42",)),
        target_qps=10.0,
    )
    records, wall_s = await run_workload(FastLLM(), workload)
    metrics = summarize_requests(workload, records, wall_s, 100.0)
    assert metrics["target_qps"] == 10.0
    assert 5.0 <= metrics["achieved_request_rate"] <= 15.0
    assert metrics["goodput_requests_per_second"] > 0


def test_repeat_stability_has_bootstrap_interval() -> None:
    values = [1.0, 1.2, 0.9, 1.1]
    interval = bootstrap_interval(values, resamples=200)
    assert interval["low"] is not None
    assert interval["low"] <= interval["high"]
    workload = make_spec().workloads[0]
    records = [
        RequestRecord(
            index=i,
            scheduled_s=0,
            started_s=0,
            latency_s=0.1,
            ttft_s=0.02,
            tpot_s=0.01,
            prompt_tokens=4,
            completion_tokens=8,
            success=True,
            slo_met=True,
            error_type=None,
            error=None,
            text="ok",
        )
        for i in range(2)
    ]
    repeat = summarize_requests(workload, records, wall_s=0.2, peak_rss_mb=100)
    aggregate = aggregate_workload_repeats(
        workload,
        [repeat, {**repeat, "goodput_requests_per_second": 11.0}],
        resamples=100,
        confidence=0.95,
        seed=42,
    )
    assert aggregate["stability"]["goodput_requests_per_second"]["ci95"]["low"] is not None


def test_equal_budget_policies_only_select_legal_candidates() -> None:
    spec = make_spec()
    legal = {candidate.id for candidate in spec.candidates}
    for strategy in spec.strategies:
        selections = [
            select_candidate(spec, strategy, index, [])
            for index in range(spec.trials_per_strategy)
        ]
        assert selections[0].candidate.id == spec.baseline_candidate
        assert all(selection.candidate.id in legal for selection in selections)


def test_agent_policy_uses_structured_history() -> None:
    spec = make_spec()
    first = make_trial(spec, spec.default_candidate, "one")
    first = TrialResult(**{**first.__dict__, "strategy": "agent", "sequence_no": 0})
    selection = select_candidate(spec, "agent", 1, [first.as_dict()])
    assert selection.candidate.id == "b"
    assert "reflection_window" in selection.reason


def test_agent_policy_rolls_back_after_quality_failure() -> None:
    spec = make_spec()
    passed = make_trial(spec, spec.default_candidate, "pass")
    passed = TrialResult(
        **{
            **passed.__dict__,
            "strategy": "agent",
            "sequence_no": 0,
        }
    )
    failed = make_trial(spec, spec.candidates[1], "failed")
    failed = TrialResult(
        **{
            **failed.__dict__,
            "strategy": "agent",
            "sequence_no": 1,
            "status": "quality_gate_failed",
            "quality_passed": False,
        }
    )
    selection = select_candidate(
        spec,
        "agent",
        2,
        [passed.as_dict(), failed.as_dict()],
        [
            {
                "candidate_id": "b",
                "negative_experience": True,
                "recommended_candidate_ids": ["a"],
            }
        ],
    )
    assert selection.candidate.id == "a"
    assert "rollback" in selection.reason


def test_agent_prunes_negative_memory_and_records_skip_reason() -> None:
    spec = make_spec()
    first = make_trial(spec, spec.default_candidate, "pass")
    first = TrialResult(
        **{
            **first.__dict__,
            "strategy": "agent",
            "sequence_no": 0,
        }
    )
    selection = select_candidate(
        spec,
        "agent",
        1,
        [first.as_dict()],
        [
            {
                "candidate_id": "b",
                "negative_experience": True,
                "positive_experience": False,
                "bottleneck_categories": ["quality_gate_failure"],
                "recommended_candidate_ids": ["a"],
            }
        ],
    )
    assert selection.candidate.id == "a"
    assert selection.skipped
    assert selection.skipped["b"] == "negative_experience"


def test_ucb1_baseline_explores_untested_candidate_before_repeating() -> None:
    spec = make_spec()
    first = make_trial(spec, spec.default_candidate, "ucb")
    first = TrialResult(
        **{
            **first.__dict__,
            "strategy": "ucb1",
            "sequence_no": 0,
        }
    )
    selection = select_candidate(spec, "ucb1", 1, [first.as_dict()])
    assert selection.candidate.id == "b"
    assert "untested" in selection.reason


def test_sample_config_is_valid() -> None:
    spec = load_experiment_spec(Path("configs/experiments_p0_p1.yaml"))
    spec.validate(
        {
            "qwen3_0_6b_q8": 32768,
            "qwen3_1_7b_q4": 32768,
            "qwen3_1_7b_q8": 32768,
        }
    )
    assert spec.trials_per_strategy == 7
    assert spec.strategies == ("default", "grid", "random", "ucb1", "agent")
    assert spec.baseline_candidate == "q06_f16_c4096_p1_cp_off"
    assert {workload.id for workload in spec.workloads} == {
        "interactive",
        "rag_reading",
        "batch_generation",
        "prefix_cache_rag",
    }
    by_id = {candidate.id: candidate for candidate in spec.candidates}
    assert by_id["q06_f16_c4096_p1_cp_on"].cache_prompt is True
    assert by_id["q06_f16_c4096_p1_cp_off"].cache_prompt is False
    assert by_id["q06_q8_q8_c4096_p1_cp_on"].cache_type_k == "q8_0"
    assert by_id["q06_q8_q8_c4096_p1_cp_on"].flash_attn == "on"


def test_prefix_cache_matrix_covers_four_legal_combinations() -> None:
    spec = load_experiment_spec(Path("configs/prefix_cache_ablation.yaml"))
    spec.validate({"qwen3_0_6b_q8": 32768})
    combinations = {
        (candidate.cache_type_k, candidate.cache_type_v, candidate.cache_prompt)
        for candidate in spec.candidates
    }
    assert combinations == {
        ("f16", "f16", True),
        ("f16", "f16", False),
        ("q8_0", "q8_0", True),
        ("q8_0", "q8_0", False),
    }
    assert all(
        candidate.flash_attn == "on"
        for candidate in spec.candidates
        if candidate.cache_type_k == "q8_0" or candidate.cache_type_v == "q8_0"
    )


def test_process_cleanup_does_not_leave_trial_processes() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    terminate_process_tree(process)
    assert process.poll() is not None
    assert os.getpid() != process.pid
