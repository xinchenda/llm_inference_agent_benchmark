from __future__ import annotations

import asyncio
import csv
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_bench.config import PROJECT_ROOT, load_models, resolve_path
from llm_bench.llm import LlamaServer, LocalLLM


def make_prompt(words: int) -> str:
    sentence = (
        "The inference service processes technical documents and reports measured "
        "latency, memory, citation quality, and tool status. "
    )
    repeats = max(1, words // len(sentence.split()))
    return (sentence * repeats).strip()


class PeakRSSMonitor:
    def __init__(self, server: LlamaServer, interval_s: float = 0.05) -> None:
        self.server = server
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self.server.rss_mb
            if rss:
                self.peak_mb = max(self.peak_mb, rss)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakRSSMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


async def run_batch(
    llm: LocalLLM,
    prompt: str,
    concurrency: int,
    repetitions: int,
    output_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> dict[str, Any]:
        async with semaphore:
            response = await asyncio.to_thread(
                llm.generate,
                "Answer with one short sentence. /no_think",
                prompt,
                max_tokens=output_tokens,
                seed=42 + index,
                disable_cache=True,
            )
            return {
                "latency_s": response.latency_s,
                "ttft_s": response.ttft_s,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "tokens_per_second": response.tokens_per_second,
            }

    started = time.perf_counter()
    results = await asyncio.gather(*[one(index) for index in range(repetitions)])
    wall_s = time.perf_counter() - started
    return results, wall_s


def aggregate_batch(
    model_id: str,
    run_kind: str,
    requested_words: int,
    concurrency: int,
    results: list[dict[str, Any]],
    wall_s: float,
    cold_start_s: float | None,
    peak_rss_mb: float,
    model_bytes: int,
) -> dict[str, Any]:
    latencies = [item["latency_s"] for item in results]
    ttfts = [item["ttft_s"] for item in results if item["ttft_s"] is not None]
    prompt_tokens = [item["prompt_tokens"] for item in results if item["prompt_tokens"]]
    completion_tokens = [item["completion_tokens"] for item in results if item["completion_tokens"]]
    total_output_tokens = sum(completion_tokens)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "run_kind": run_kind,
        "requested_prompt_words": requested_words,
        "mean_prompt_tokens": float(np.mean(prompt_tokens)) if prompt_tokens else None,
        "concurrency": concurrency,
        "requests": len(results),
        "wall_s": wall_s,
        "cold_start_s": cold_start_s,
        "peak_process_rss_mb": peak_rss_mb,
        "model_file_mb": model_bytes / (1024 * 1024),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "ttft_p50_s": float(np.percentile(ttfts, 50)) if ttfts else None,
        "ttft_p95_s": float(np.percentile(ttfts, 95)) if ttfts else None,
        "mean_request_tokens_per_second": float(
            np.mean(
                [
                    item["tokens_per_second"]
                    for item in results
                    if item["tokens_per_second"] is not None
                ]
            )
        ),
        "aggregate_output_tokens_per_second": (
            total_output_tokens / wall_s if wall_s > 0 else None
        ),
        "local_incremental_cost_usd_per_request": 0.0,
        "cost_note": (
            "Self-hosted open-source stack. Zero incremental software/API charge; "
            "hardware purchase and electricity are excluded."
        ),
    }


async def benchmark_model(
    model_id: str,
    model: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    model_path = PROJECT_ROOT / model["file"]
    model_bytes = model_path.stat().st_size
    perf = config["perf"]
    output_tokens = int(perf["output_tokens"])
    repetitions = int(perf["repetitions"])
    warmup = int(perf["warmup"])
    records: list[dict[str, Any]] = []

    server = LlamaServer(
        model_path=model_path,
        port=8080,
        context_size=int(config["llm"]["context_size"]),
        parallel=1,
        log_path=output_dir / "logs" / f"perf-{model_id}-context.log",
    )
    server.start()
    llm = LocalLLM(
        base_url=server.base_url,
        model_name=model_id,
        timeout_s=180,
        temperature=0.0,
        top_p=0.8,
        max_tokens=output_tokens,
    )
    try:
        for _ in range(warmup):
            await asyncio.to_thread(
                llm.generate,
                "Answer with one short sentence.",
                "Warm up the local inference server.",
                max_tokens=8,
                disable_cache=True,
            )
        for words in perf["prompt_words"]:
            with PeakRSSMonitor(server) as monitor:
                results, wall_s = await run_batch(
                    llm,
                    make_prompt(int(words)),
                    concurrency=1,
                    repetitions=repetitions,
                    output_tokens=output_tokens,
                )
            records.append(
                aggregate_batch(
                    model_id=model_id,
                    run_kind="context",
                    requested_words=int(words),
                    concurrency=1,
                    results=results,
                    wall_s=wall_s,
                    cold_start_s=server.cold_start_s,
                    peak_rss_mb=monitor.peak_mb,
                    model_bytes=model_bytes,
                )
            )
    finally:
        server.stop()

    for concurrency in perf["concurrency"]:
        server = LlamaServer(
            model_path=model_path,
            port=8080,
            context_size=int(config["llm"]["context_size"]),
            parallel=int(concurrency),
            log_path=output_dir / "logs" / f"perf-{model_id}-parallel-{concurrency}.log",
        )
        server.start()
        llm = LocalLLM(
            base_url=server.base_url,
            model_name=model_id,
            timeout_s=180,
            temperature=0.0,
            top_p=0.8,
            max_tokens=output_tokens,
        )
        try:
            for _ in range(warmup):
                await asyncio.to_thread(
                    llm.generate,
                    "Answer with one short sentence.",
                    "Warm up.",
                    max_tokens=8,
                    disable_cache=True,
                )
            with PeakRSSMonitor(server) as monitor:
                results, wall_s = await run_batch(
                    llm,
                    make_prompt(256),
                    concurrency=int(concurrency),
                    repetitions=max(repetitions, int(concurrency) * 2),
                    output_tokens=output_tokens,
                )
            records.append(
                aggregate_batch(
                    model_id=model_id,
                    run_kind="concurrency",
                    requested_words=256,
                    concurrency=int(concurrency),
                    results=results,
                    wall_s=wall_s,
                    cold_start_s=server.cold_start_s,
                    peak_rss_mb=monitor.peak_mb,
                    model_bytes=model_bytes,
                )
            )
        finally:
            server.stop()
    return records


async def run_performance(config: dict[str, Any]) -> list[dict[str, Any]]:
    model_registry = load_models()["models"]
    selected = [
        "qwen3_0_6b_q8",
        "qwen3_1_7b_q4",
        "qwen3_1_7b_q8",
    ]
    output_dir = resolve_path(config, "paths.artifacts") / "perf"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for model_id in selected:
        print(f"benchmarking {model_id}", flush=True)
        records.extend(
            await benchmark_model(
                model_id=model_id,
                model=model_registry[model_id],
                config=config,
                output_dir=output_dir,
            )
        )
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records
