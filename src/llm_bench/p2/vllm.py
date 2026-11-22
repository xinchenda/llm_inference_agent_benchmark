from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil


@dataclass(frozen=True)
class VLLMServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    max_model_len: int = 4096
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    continuous_batching: bool = True
    enable_prefix_caching: bool = True
    enable_cuda_graph: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_model_len < 256:
            raise ValueError("max_model_len must be >= 256")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if not self.continuous_batching:
            raise ValueError("vLLM V1 continuous batching cannot be disabled")

    def command(self) -> list[str]:
        self.validate()
        command = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--dtype",
            self.dtype,
            "--kv-cache-dtype",
            self.kv_cache_dtype,
        ]
        command.append(
            "--enable-prefix-caching"
            if self.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )
        if not self.enable_cuda_graph:
            command.append("--enforce-eager")
        return command


class VLLMServer:
    def __init__(
        self,
        config: VLLMServerConfig,
        log_path: Path,
    ) -> None:
        self.config = config
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self.cold_start_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def start(self, timeout_s: int = 600) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.config.command(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1"},
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                raise RuntimeError(f"vLLM exited during startup:\n{tail}")
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    self.cold_start_s = time.perf_counter() - started
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"vLLM did not become ready in {timeout_s}s")

    @property
    def rss_mb(self) -> float | None:
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            process = psutil.Process(self.process.pid)
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                total += child.memory_info().rss
            return total / (1024 * 1024)
        except psutil.Error:
            return None

    def metrics_text(self, timeout_s: float = 10) -> str:
        response = httpx.get(f"{self.base_url}/metrics", timeout=timeout_s)
        response.raise_for_status()
        return response.text

    def stop(self) -> None:
        from llm_bench.experiment.runtime import terminate_process_tree

        terminate_process_tree(self.process)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
