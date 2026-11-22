from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil


@dataclass(frozen=True)
class SGLangServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 30000
    tp_size: int = 1
    mem_fraction_static: float = 0.85
    max_running_requests: int = 8
    chunked_prefill_size: int = 8192
    enable_radix_cache: bool = True
    enable_cache_report: bool = True
    enable_cuda_graph: bool = True

    def validate(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.tp_size < 1:
            raise ValueError("tp_size must be positive")
        if not 0 < self.mem_fraction_static <= 1:
            raise ValueError("mem_fraction_static must be in (0, 1]")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be positive")
        if self.chunked_prefill_size < self.max_running_requests:
            raise ValueError(
                "chunked_prefill_size must be >= max_running_requests"
            )

    def command(self) -> list[str]:
        self.validate()
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tp-size",
            str(self.tp_size),
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--max-running-requests",
            str(self.max_running_requests),
            "--chunked-prefill-size",
            str(self.chunked_prefill_size),
        ]
        if not self.enable_radix_cache:
            command.append("--disable-radix-cache")
        if self.enable_cache_report:
            command.append("--enable-cache-report")
        if not self.enable_cuda_graph:
            command.append("--disable-cuda-graph")
        return command


class SGLangServer:
    def __init__(self, config: SGLangServerConfig, log_path: Path) -> None:
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
                raise RuntimeError(f"SGLang exited during startup:\n{tail}")
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    self.cold_start_s = time.perf_counter() - started
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"SGLang did not become ready in {timeout_s}s")

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
