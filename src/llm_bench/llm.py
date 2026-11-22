from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    timings: dict[str, Any]

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.latency_s <= 0:
            return None
        return self.completion_tokens / self.latency_s


class LlamaServer:
    """Owns one local llama.cpp server process and its lifecycle."""

    def __init__(
        self,
        model_path: Path,
        port: int = 8080,
        context_size: int = 8192,
        parallel: int = 2,
        batch_size: int = 2048,
        ubatch_size: int = 512,
        flash_attn: str = "auto",
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        log_path: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.context_size = context_size
        self.parallel = parallel
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.flash_attn = flash_attn
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.log_path = log_path or Path(f"llama-server-{port}.log")
        self.process: subprocess.Popen[str] | None = None
        self.cold_start_s: float | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self, timeout_s: int = 120) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing {self.model_path}. Run ./scripts/download_models.sh")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        command = self.command()
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "LLAMA_LOG_COLORS": "0"},
        )
        health_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"llama-server exited during startup:\n{tail}")
            try:
                response = httpx.get(health_url, timeout=1)
                if response.status_code == 200:
                    self.cold_start_s = time.perf_counter() - started
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"llama-server did not become ready in {timeout_s}s")

    def command(self) -> list[str]:
        return [
            "llama-server",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            str(self.parallel),
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--flash-attn",
            self.flash_attn,
            "--cache-type-k",
            self.cache_type_k,
            "--cache-type-v",
            self.cache_type_v,
            "--n-gpu-layers",
            "99",
            "--jinja",
            "--no-webui",
            "--metrics",
        ]

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

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


class LocalLLM:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "local",
        timeout_s: float = 120,
        temperature: float = 0.0,
        top_p: float = 0.8,
        max_tokens: int = 180,
        cache_prompt: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cache_prompt = cache_prompt

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        seed: int = 42,
        disable_cache: bool = False,
        cache_prompt: bool | None = None,
    ) -> LLMResponse:
        payload = self.build_payload(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens or self.max_tokens,
            seed=seed,
            disable_cache=disable_cache,
            cache_prompt=cache_prompt,
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        with httpx.Client(timeout=self.timeout_s) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    body = response.text[:800].replace("\n", " ")
                    raise RuntimeError(
                        f"HTTP {response.status_code} from local model service: {body}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    if event.get("timings"):
                        timings = event["timings"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        parts.append(content)
        finished = time.perf_counter()
        text = strip_thinking("".join(parts).strip())
        return LLMResponse(
            text=text,
            latency_s=finished - started,
            ttft_s=(first_token_at - started) if first_token_at else None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            timings=timings,
        )

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        seed: int,
        disable_cache: bool,
        cache_prompt: bool | None,
    ) -> dict[str, Any]:
        effective_cache_prompt = (
            self.cache_prompt if cache_prompt is None else cache_prompt
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt + "\n/no_think"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if disable_cache or effective_cache_prompt is False:
            payload["cache_prompt"] = False
        elif effective_cache_prompt is True:
            payload["cache_prompt"] = True
        return payload

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{self.base_url}/models", headers=self.headers)
            response.raise_for_status()
            return response.json()

    def metrics_text(self, timeout_s: float = 10) -> str:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(f"{self.base_url.removesuffix('/v1')}/metrics")
            response.raise_for_status()
            return response.text


def strip_thinking(text: str) -> str:
    if "<｜end▁of▁thinking｜>" in text:
        return text.split("<｜end▁of▁thinking｜>", 1)[1].strip()
    return text


def choose_device(requested: str) -> str:
    if requested != "mps":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
