from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(".p2_remote/extracted/artifacts/p2")
OUTPUT = Path("reports/p2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_summary(responses: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"requests": len(responses)}
    for metric in ("ttft_s", "latency_s", "prompt_tokens", "completion_tokens", "cache_n"):
        values = [
            float(item[metric])
            for item in responses
            if item.get(metric) is not None
        ]
        output[metric] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return output


def log_facts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "available_kv_cache_gib": r"Available KV cache memory:\s*([0-9.]+)\s*GiB",
        "gpu_kv_cache_tokens": r"GPU KV cache size:\s*([0-9,]+)\s*tokens",
        "maximum_concurrency": (
            r"Maximum concurrency for\s*[0-9,]+\s*tokens per request:"
            r"\s*([0-9.]+)x"
        ),
    }
    output = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).replace(",", "")
            output[key] = float(value) if "." in value else int(value)
    return output


def summarize_vllm(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = Path(payload["model"]).name
    safe_flags = []
    skip_model = True
    for argument in payload.get("command", []):
        if skip_model:
            if argument == "serve":
                safe_flags.append(argument)
                continue
            if argument != "vllm" and argument != "serve":
                skip_model = False
            continue
        safe_flags.append(argument)
    after = payload["metrics_after"]
    return {
        "source": path.name,
        "source_sha256": sha256(path),
        "backend": "vllm",
        "model": model,
        "safe_command_flags": safe_flags,
        "cold_start_s": payload.get("cold_start_s"),
        "prefix_words": payload.get("prefix_words"),
        "max_tokens": payload.get("max_tokens"),
        "responses": response_summary(payload.get("responses", [])),
        "log_facts": log_facts(path.with_suffix(".log")),
        "serving_metrics_after": {
            key: after.get(key)
            for key in (
                "num_requests_running",
                "num_requests_waiting",
                "kv_cache_usage_perc",
                "prefix_cache_queries",
                "prefix_cache_hits",
                "prefix_cache_hit_rate",
                "prompt_tokens_total",
                "generation_tokens_total",
                "request_success_total",
                "process_resident_memory_bytes",
                "ttft_p50_s",
                "ttft_p95_s",
                "tpot_mean_s",
                "e2e_latency_p95_s",
            )
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vllm = {}
    for name in (
        "vllm_prefix",
        "vllm_no_prefix",
        "vllm_long_prefix",
        "vllm_long_no_prefix",
        "vllm_eager",
    ):
        path = ROOT / f"{name}.json"
        if path.exists():
            vllm[name] = summarize_vllm(path)
    vllm_summary = {
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 4090 D",
            "vram_gb": 24,
            "driver": "595.71.05",
        },
        "runtime": {"vllm": "0.34.0", "torch": "2.14+cu130"},
        "runs": vllm,
    }
    (OUTPUT / "vllm_summary.json").write_text(
        json.dumps(vllm_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    long_off = vllm["vllm_long_no_prefix"]
    long_on = vllm["vllm_long_prefix"]
    short_off = vllm["vllm_no_prefix"]
    short_on = vllm["vllm_prefix"]
    eager = vllm["vllm_eager"]
    graph = vllm["vllm_long_prefix"]

    def reduction(disabled: float, enabled: float) -> float:
        return (disabled - enabled) / disabled * 100.0

    comparisons = {
        "prefix_cache_short": {
            "cache_hit_rate_off": short_off["serving_metrics_after"][
                "prefix_cache_hit_rate"
            ],
            "cache_hit_rate_on": short_on["serving_metrics_after"][
                "prefix_cache_hit_rate"
            ],
            "response_ttft_mean_off_s": short_off["responses"]["ttft_s"]["mean"],
            "response_ttft_mean_on_s": short_on["responses"]["ttft_s"]["mean"],
            "response_ttft_reduction_pct": reduction(
                short_off["responses"]["ttft_s"]["mean"],
                short_on["responses"]["ttft_s"]["mean"],
            ),
        },
        "prefix_cache_long": {
            "cache_hit_rate_off": long_off["serving_metrics_after"][
                "prefix_cache_hit_rate"
            ],
            "cache_hit_rate_on": long_on["serving_metrics_after"][
                "prefix_cache_hit_rate"
            ],
            "response_ttft_mean_off_s": long_off["responses"]["ttft_s"]["mean"],
            "response_ttft_mean_on_s": long_on["responses"]["ttft_s"]["mean"],
            "response_ttft_reduction_pct": reduction(
                long_off["responses"]["ttft_s"]["mean"],
                long_on["responses"]["ttft_s"]["mean"],
            ),
            "histogram_ttft_p95_off_s": long_off["serving_metrics_after"][
                "ttft_p95_s"
            ],
            "histogram_ttft_p95_on_s": long_on["serving_metrics_after"][
                "ttft_p95_s"
            ],
            "histogram_ttft_p95_reduction_pct": reduction(
                long_off["serving_metrics_after"]["ttft_p95_s"],
                long_on["serving_metrics_after"]["ttft_p95_s"],
            ),
        },
        "cuda_graph": {
            "histogram_ttft_p95_eager_s": eager["serving_metrics_after"][
                "ttft_p95_s"
            ],
            "histogram_ttft_p95_graph_s": graph["serving_metrics_after"][
                "ttft_p95_s"
            ],
            "ttft_p95_reduction_pct": reduction(
                eager["serving_metrics_after"]["ttft_p95_s"],
                graph["serving_metrics_after"]["ttft_p95_s"],
            ),
            "tpot_mean_eager_s": eager["serving_metrics_after"]["tpot_mean_s"],
            "tpot_mean_graph_s": graph["serving_metrics_after"]["tpot_mean_s"],
            "tpot_reduction_pct": reduction(
                eager["serving_metrics_after"]["tpot_mean_s"],
                graph["serving_metrics_after"]["tpot_mean_s"],
            ),
            "e2e_p95_eager_s": eager["serving_metrics_after"][
                "e2e_latency_p95_s"
            ],
            "e2e_p95_graph_s": graph["serving_metrics_after"][
                "e2e_latency_p95_s"
            ],
            "e2e_p95_reduction_pct": reduction(
                eager["serving_metrics_after"]["e2e_latency_p95_s"],
                graph["serving_metrics_after"]["e2e_latency_p95_s"],
            ),
        },
    }
    (OUTPUT / "derived_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n",
        encoding="utf-8",
    )
    triton_path = ROOT / "triton_rmsnorm.json"
    triton = json.loads(triton_path.read_text(encoding="utf-8"))
    triton["source_sha256"] = sha256(triton_path)
    (OUTPUT / "triton_rmsnorm_gpu.json").write_text(
        json.dumps(triton, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "vllm_runs": sorted(vllm),
                "triton": triton["status"],
                "derived": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
