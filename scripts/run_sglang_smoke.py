from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from llm_bench.llm import LocalLLM
from llm_bench.p2.metrics import summarize_sglang_serving_metrics
from llm_bench.p2.sampling import MetricsSampler
from llm_bench.p2.sglang import SGLangServer, SGLangServerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGLang and collect real metrics")
    parser.add_argument("--config", default="configs/sglang_backends.yaml")
    parser.add_argument("--candidate", default="qwen3_0_6b_radix")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prefix-words", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p2/sglang_smoke.json"),
    )
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = SGLangServerConfig(**payload["sglang"][args.candidate])
    server = SGLangServer(config, args.output.with_suffix(".log"))
    try:
        server.start()
        client = LocalLLM(
            f"{server.base_url}/v1",
            config.model_path,
            timeout_s=180,
            temperature=0,
        )
        before = summarize_sglang_serving_metrics(server.metrics_text())
        prefix = "Shared reference: Atlas limit is 42. " + (
            "technical evidence " * args.prefix_words
        )
        responses = []
        with MetricsSampler(
            server.metrics_text,
            summarize_sglang_serving_metrics,
        ) as sampler:
            for index in range(args.requests):
                response = client.generate(
                    "Answer briefly. /no_think",
                    f"{prefix}\nQuestion {index}: What is the Atlas limit?",
                    max_tokens=args.max_tokens,
                    disable_cache=False,
                    cache_prompt=True,
                )
                responses.append(
                    {
                        "cache_n": response.timings.get("cache_n"),
                        "ttft_s": response.ttft_s,
                        "latency_s": response.latency_s,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "text": response.text,
                    }
                )
        after = summarize_sglang_serving_metrics(server.metrics_text())
        result = {
            "backend": "sglang",
            "config": config.__dict__,
            "command": config.command(),
            "cold_start_s": server.cold_start_s,
            "request_count": args.requests,
            "prefix_words": args.prefix_words,
            "max_tokens": args.max_tokens,
            "metrics_before": before,
            "metrics_after": after,
            "metrics_during": sampler.samples,
            "metrics_sampling_errors": sampler.errors,
            "responses": responses,
            "radix_attention": config.enable_radix_cache,
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
