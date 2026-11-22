from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

STRATEGIES = ("default", "grid", "random", "ucb1", "agent")
LOAD_MODES = ("closed", "open_qps")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateConfig:
    id: str
    model_id: str
    context_size: int
    parallel: int
    batch_size: int = 512
    ubatch_size: int = 128
    flash_attn: str = "auto"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    cache_prompt: bool = True

    def validate(self, model_limit: int | None = None) -> None:
        if not self.id.strip():
            raise ValueError("candidate id must not be empty")
        if not self.model_id.strip():
            raise ValueError(f"{self.id}: model_id must not be empty")
        if self.context_size < 256:
            raise ValueError(f"{self.id}: context_size must be >= 256")
        if self.parallel < 1 or self.parallel > 32:
            raise ValueError(f"{self.id}: parallel must be between 1 and 32")
        if self.batch_size < 1 or self.ubatch_size < 1:
            raise ValueError(f"{self.id}: batch and ubatch sizes must be positive")
        if self.ubatch_size > self.batch_size:
            raise ValueError(f"{self.id}: ubatch_size must be <= batch_size")
        if self.batch_size % self.ubatch_size != 0:
            raise ValueError(f"{self.id}: batch_size must be divisible by ubatch_size")
        if self.ubatch_size > self.context_size:
            raise ValueError(f"{self.id}: ubatch_size must be <= context_size")
        if self.flash_attn not in {"on", "off", "auto"}:
            raise ValueError(f"{self.id}: flash_attn must be on, off, or auto")
        allowed_cache = {"f16", "q8_0"}
        if self.cache_type_k not in allowed_cache or self.cache_type_v not in allowed_cache:
            raise ValueError(f"{self.id}: cache types must be f16 or q8_0")
        if "q8_0" in {self.cache_type_k, self.cache_type_v} and self.flash_attn != "on":
            raise ValueError(f"{self.id}: quantized KV cache requires flash_attn=on")
        if not isinstance(self.cache_prompt, bool):
            raise ValueError(f"{self.id}: cache_prompt must be boolean")
        if model_limit is not None and self.context_size > model_limit:
            raise ValueError(
                f"{self.id}: context_size {self.context_size} exceeds model limit {model_limit}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualitySpec:
    prompt: str
    expected_exact: str | None = None
    expected_contains: tuple[str, ...] = ()
    expected_json: dict[str, Any] = field(default_factory=dict)
    critical_fact: str | None = None

    def validate(self, workload_id: str) -> None:
        if not self.prompt.strip():
            raise ValueError(f"{workload_id}: quality prompt must not be empty")
        if not any((self.expected_exact, self.expected_contains, self.expected_json)):
            raise ValueError(f"{workload_id}: quality prompt has no deterministic check")

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "expected_exact": self.expected_exact,
            "expected_contains": list(self.expected_contains),
            "expected_json": self.expected_json,
            "critical_fact": self.critical_fact,
        }


@dataclass(frozen=True)
class WorkloadSpec:
    id: str
    prompt_words: int
    output_tokens: int
    requests: int
    load_mode: str
    concurrency: int
    latency_slo_s: float
    quality: QualitySpec
    target_qps: float | None = None

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("workload id must not be empty")
        if self.prompt_words < 1 or self.output_tokens < 1 or self.requests < 1:
            raise ValueError(f"{self.id}: prompt/output/request counts must be positive")
        if self.load_mode not in LOAD_MODES:
            raise ValueError(f"{self.id}: unsupported load mode {self.load_mode!r}")
        if self.concurrency < 1 or self.concurrency > 32:
            raise ValueError(f"{self.id}: concurrency must be between 1 and 32")
        if self.latency_slo_s <= 0:
            raise ValueError(f"{self.id}: latency SLO must be positive")
        if self.load_mode == "open_qps" and (self.target_qps or 0) <= 0:
            raise ValueError(f"{self.id}: open_qps requires a positive target_qps")
        self.quality.validate(self.id)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality"] = self.quality.as_dict()
        return payload


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    experiment_id: str
    seed: int
    baseline_candidate: str
    candidates: tuple[CandidateConfig, ...]
    workloads: tuple[WorkloadSpec, ...]
    strategies: tuple[str, ...]
    trials_per_strategy: int
    repeats_per_trial: int
    warmup_requests: int
    max_retries: int
    service_start_timeout_s: float
    request_timeout_s: float
    bootstrap_resamples: int
    confidence_level: float

    def validate(self, model_limits: dict[str, int] | None = None) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if self.trials_per_strategy < 1:
            raise ValueError("trials_per_strategy must be positive")
        if self.repeats_per_trial < 2:
            raise ValueError("repeats_per_trial must be >= 2 for stability evidence")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if self.max_retries < 0 or self.max_retries > 1:
            raise ValueError("max_retries must be 0 or 1")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        candidate_ids = [candidate.id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")
        if self.baseline_candidate not in candidate_ids:
            raise ValueError("baseline_candidate must reference a candidate")
        for candidate in self.candidates:
            limit = model_limits.get(candidate.model_id) if model_limits else None
            candidate.validate(limit)
        workload_ids = [workload.id for workload in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("workload ids must be unique")
        required = {
            "interactive",
            "rag_reading",
            "batch_generation",
            "prefix_cache_rag",
        }
        if set(workload_ids) != required:
            raise ValueError(f"workloads must be exactly {sorted(required)}")
        for workload in self.workloads:
            workload.validate()
        if len(self.strategies) != len(set(self.strategies)):
            raise ValueError("strategies must be unique")
        if not set(self.strategies).issubset(STRATEGIES):
            raise ValueError(f"unsupported strategies: {set(self.strategies) - set(STRATEGIES)}")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be >= 100")

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.as_dict())

    @property
    def default_candidate(self) -> CandidateConfig:
        return next(
            candidate for candidate in self.candidates if candidate.id == self.baseline_candidate
        )

    def candidate(self, candidate_id: str) -> CandidateConfig:
        try:
            return next(candidate for candidate in self.candidates if candidate.id == candidate_id)
        except StopIteration as exc:
            raise ValueError(f"Unknown candidate: {candidate_id}") from exc

    def experiment_hash(
        self,
        strategy: str,
        sequence_no: int,
        repeat_index: int,
        candidate_id: str,
    ) -> str:
        return stable_hash(
            {
                "spec_hash": self.spec_hash,
                "strategy": strategy,
                "sequence_no": sequence_no,
                "repeat_index": repeat_index,
                "candidate_id": candidate_id,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "baseline_candidate": self.baseline_candidate,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "workloads": [workload.as_dict() for workload in self.workloads],
            "strategies": list(self.strategies),
            "trials_per_strategy": self.trials_per_strategy,
            "repeats_per_trial": self.repeats_per_trial,
            "warmup_requests": self.warmup_requests,
            "max_retries": self.max_retries,
            "service_start_timeout_s": self.service_start_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    experiment_hash: str
    spec_id: str
    strategy: str
    sequence_no: int
    repeat_index: int
    candidate: CandidateConfig
    status: str
    failure_type: str | None
    failure_detail: str | None
    quality_passed: bool
    quality: dict[str, Any]
    performance: dict[str, Any]
    baseline_delta: dict[str, Any]
    provenance: dict[str, Any]
    artifact_path: str
    created_at: str

    def ledger_payload(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "experiment_hash": self.experiment_hash,
            "spec_id": self.spec_id,
            "strategy": self.strategy,
            "sequence_no": self.sequence_no,
            "repeat_index": self.repeat_index,
            "candidate_id": self.candidate.id,
            "candidate_json": canonical_json(self.candidate.as_dict()),
            "status": self.status,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
            "quality_passed": int(self.quality_passed),
            "quality_json": canonical_json(self.quality),
            "performance_json": canonical_json(self.performance),
            "baseline_delta_json": canonical_json(self.baseline_delta),
            **self.provenance,
            "load_id": ",".join(sorted(self.performance)),
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "experiment_hash": self.experiment_hash,
            "spec_id": self.spec_id,
            "strategy": self.strategy,
            "sequence_no": self.sequence_no,
            "repeat_index": self.repeat_index,
            "candidate": self.candidate.as_dict(),
            "status": self.status,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
            "quality_passed": self.quality_passed,
            "quality": self.quality,
            "performance": self.performance,
            "baseline_delta": self.baseline_delta,
            "provenance": self.provenance,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
        }


def _quality_from_dict(payload: dict[str, Any]) -> QualitySpec:
    return QualitySpec(
        prompt=str(payload["prompt"]),
        expected_exact=payload.get("expected_exact"),
        expected_contains=tuple(payload.get("expected_contains") or ()),
        expected_json=dict(payload.get("expected_json") or {}),
        critical_fact=payload.get("critical_fact"),
    )


def load_experiment_spec(path: Path) -> ExperimentSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    statistics = payload.get("statistics", {})
    spec = ExperimentSpec(
        schema_version=int(payload["schema_version"]),
        experiment_id=str(payload["experiment_id"]),
        seed=int(payload.get("seed", 42)),
        baseline_candidate=str(payload["baseline_candidate"]),
        candidates=tuple(
            CandidateConfig(
                id=str(item["id"]),
                model_id=str(item["model_id"]),
                context_size=int(item["context_size"]),
                parallel=int(item["parallel"]),
                batch_size=int(item.get("batch_size", 512)),
                ubatch_size=int(item.get("ubatch_size", 128)),
                flash_attn=str(item.get("flash_attn", "auto")),
                cache_type_k=str(item.get("cache_type_k", "f16")),
                cache_type_v=str(item.get("cache_type_v", "f16")),
                cache_prompt=bool(item.get("cache_prompt", True)),
            )
            for item in payload["candidates"]
        ),
        workloads=tuple(
            WorkloadSpec(
                id=str(item["id"]),
                prompt_words=int(item["prompt_words"]),
                output_tokens=int(item["output_tokens"]),
                requests=int(item["requests"]),
                load_mode=str(item["load_mode"]),
                concurrency=int(item.get("concurrency", 1)),
                latency_slo_s=float(item["latency_slo_s"]),
                target_qps=(
                    float(item["target_qps"]) if item.get("target_qps") is not None else None
                ),
                quality=_quality_from_dict(item["quality"]),
            )
            for item in payload["workloads"]
        ),
        strategies=tuple(str(item) for item in payload["strategies"]),
        trials_per_strategy=int(payload["trials_per_strategy"]),
        repeats_per_trial=int(payload["repeats_per_trial"]),
        warmup_requests=int(payload.get("warmup_requests", 1)),
        max_retries=int(payload.get("max_retries", 1)),
        service_start_timeout_s=float(payload.get("service_start_timeout_s", 120)),
        request_timeout_s=float(payload.get("request_timeout_s", 180)),
        bootstrap_resamples=int(statistics.get("bootstrap_resamples", 1000)),
        confidence_level=float(statistics.get("confidence_level", 0.95)),
    )
    spec.validate()
    return spec
