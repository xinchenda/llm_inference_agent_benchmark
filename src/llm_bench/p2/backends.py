from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from llm_bench.llm import LlamaServer
from llm_bench.p2.sglang import SGLangServer, SGLangServerConfig
from llm_bench.p2.vllm import VLLMServer, VLLMServerConfig


class ServingBackend(Protocol):
    @property
    def base_url(self) -> str: ...

    def start(self, timeout_s: int = 600) -> None: ...

    def stop(self) -> None: ...

    def metrics_text(self, timeout_s: float = 10) -> str: ...


def create_backend(kind: str, config: dict[str, Any], log_path: Path) -> ServingBackend:
    if kind == "llamacpp":
        return LlamaServer(
            model_path=Path(config["model_path"]),
            port=int(config.get("port", 8080)),
            context_size=int(config.get("context_size", 8192)),
            parallel=int(config.get("parallel", 2)),
            batch_size=int(config.get("batch_size", 2048)),
            ubatch_size=int(config.get("ubatch_size", 512)),
            flash_attn=str(config.get("flash_attn", "auto")),
            cache_type_k=str(config.get("cache_type_k", "f16")),
            cache_type_v=str(config.get("cache_type_v", "f16")),
            log_path=log_path,
        )
    if kind == "vllm":
        return VLLMServer(VLLMServerConfig(**config), log_path)
    if kind == "sglang":
        return SGLangServer(SGLangServerConfig(**config), log_path)
    raise ValueError(f"Unsupported serving backend: {kind}")
