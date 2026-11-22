from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from llm_bench.llm import LocalLLM
from llm_bench.p2.metrics import summarize_serving_metrics
from llm_bench.p2.vllm import VLLMServer, VLLMServerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real vLLM smoke test and metrics probe")
    parser.add_argument("--config", default="configs/vllm_backends.yaml")
    parser.add_argument("--candidate", default="qwen3_1_7b")
    parser.add_argument("--output", type=Path, default=Path("artifacts/p2/vllm_smoke.json"))
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw = payload["vllm"][args.candidate]
    config = VLLMServerConfig(**raw)
    server = VLLMServer(config, args.output.with_suffix(".log"))
    try:
        server.start()
        llm = LocalLLM(
            f"{server.base_url}/v1",
            config.model,
            timeout_s=180,
            temperature=0,
        )
        before = summarize_serving_metrics(server.metrics_text())
        shared_prefix = "Atlas limit is 42. " + ("technical evidence " * 200)
        responses = []
        for index in range(3):
            response = llm.generate(
                "Answer briefly. /no_think",
                f"{shared_prefix}\nQuestion {index}: What is the Atlas limit?",
                max_tokens=16,
                disable_cache=False,
                cache_prompt=True,
            )
            responses.append(
                {
                    "latency_s": response.latency_s,
                    "ttft_s": response.ttft_s,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "cache_n": response.timings.get("cache_n"),
                    "text": response.text,
                }
            )
        after = summarize_serving_metrics(server.metrics_text())
        result = {
            "backend": "vllm",
            "model": config.model,
            "command": config.command(),
            "cold_start_s": server.cold_start_s,
            "metrics_before": before,
            "metrics_after": after,
            "responses": responses,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        server.stop()


if __name__ == "__main__":
    main()
