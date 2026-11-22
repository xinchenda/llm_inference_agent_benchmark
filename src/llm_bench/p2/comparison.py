from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendRun:
    backend: str
    hardware_fingerprint: str
    workload_id: str
    metrics: dict[str, float | None]


def load_backend_run(path: Path) -> BackendRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BackendRun(
        backend=str(payload["backend"]),
        hardware_fingerprint=str(payload["hardware_fingerprint"]),
        workload_id=str(payload["workload_id"]),
        metrics={
            str(key): float(value) if value is not None else None
            for key, value in payload["metrics"].items()
        },
    )


def compare_same_workload(runs: list[BackendRun]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one backend run is required")
    workload_ids = {run.workload_id for run in runs}
    if len(workload_ids) != 1:
        raise ValueError("Backend runs must use the same workload_id")
    fingerprints = {run.hardware_fingerprint for run in runs}
    same_hardware = len(fingerprints) == 1
    comparisons: dict[str, Any] = {}
    if same_hardware and len(runs) > 1:
        baseline = runs[0]
        for run in runs[1:]:
            ratios = {}
            for metric in (
                "ttft_p50_s",
                "ttft_p95_s",
                "latency_p95_s",
                "goodput_requests_per_second",
                "peak_memory_mb",
            ):
                candidate = run.metrics.get(metric)
                reference = baseline.metrics.get(metric)
                ratios[metric] = (
                    candidate / reference
                    if candidate is not None and reference not in (None, 0)
                    else None
                )
            comparisons[f"{run.backend}_vs_{baseline.backend}"] = ratios
    return {
        "workload_id": runs[0].workload_id,
        "same_hardware": same_hardware,
        "cross_hardware_ratio_available": False,
        "runs": [
            {
                "backend": run.backend,
                "hardware_fingerprint": run.hardware_fingerprint,
                "metrics": run.metrics,
            }
            for run in runs
        ],
        "ratios": comparisons,
        "note": (
            "Metrics from different hardware fingerprints are reported separately. "
            "No cross-hardware performance ratio is produced."
        ),
    }
