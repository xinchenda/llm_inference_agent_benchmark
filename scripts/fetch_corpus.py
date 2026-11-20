#!/usr/bin/env python3
"""Fetch a pinned, legally reusable technical documentation corpus."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"

REPOSITORIES = {
    "fastapi": {
        "repo": "fastapi/fastapi",
        "commit": "05e6df7e906c3decba377c37548019973f40bf15",
        "license": "MIT",
        "paths": [
            "docs/en/docs/advanced/settings.md",
            "docs/en/docs/advanced/events.md",
            "docs/en/docs/advanced/middleware.md",
            "docs/en/docs/advanced/testing-dependencies.md",
            "docs/en/docs/async.md",
            "docs/en/docs/benchmarks.md",
            "docs/en/docs/deployment/concepts.md",
            "docs/en/docs/deployment/docker.md",
            "docs/en/docs/deployment/server-workers.md",
            "docs/en/docs/python-types.md",
            "docs/en/docs/tutorial/first-steps.md",
            "docs/en/docs/tutorial/dependencies/index.md",
            "docs/en/docs/tutorial/dependencies/dependencies-with-yield.md",
            "docs/en/docs/tutorial/background-tasks.md",
            "docs/en/docs/tutorial/handling-errors.md",
            "docs/en/docs/tutorial/testing.md",
            "docs/en/docs/tutorial/bigger-applications.md",
            "docs/en/docs/tutorial/security/oauth2-jwt.md",
            "docs/en/docs/tutorial/response-model.md",
            "docs/en/docs/tutorial/server-sent-events.md",
        ],
    },
    "llamacpp": {
        "repo": "ggml-org/llama.cpp",
        "commit": "a83d68b562b19fba1f8836edc93eb76d9ca1ce25",
        "license": "MIT",
        "paths": [
            "docs/function-calling.md",
            "docs/install.md",
            "docs/docker.md",
            "docs/build.md",
            "docs/models.md",
            "docs/ops.md",
            "docs/preset.md",
            "docs/speculative.md",
            "docs/backend/CUDA-FEDORA.md",
            "docs/backend/OPENVINO.md",
            "docs/backend/CANN.md",
            "docs/backend/snapdragon/README.md",
            "docs/development/token_generation_performance_tips.md",
            "docs/build-profiling.md",
            "docs/release.md",
            "docs/completions.md",
            "docs/autoparser.md",
            "docs/multimodal.md",
            "docs/multimodal/llava.md",
            "docs/multimodal/gemma3.md",
        ],
    },
    "vllm": {
        "repo": "vllm-project/vllm",
        "commit": "0aa2bb19496d9252c0dd7f8d1c0d073fc634d471",
        "license": "Apache-2.0",
        "paths": [
            "docs/usage/reproducibility.md",
            "docs/usage/metrics.md",
            "docs/benchmarking/README.md",
            "docs/serving/offline_inference.md",
            "docs/serving/online_serving/openai_compatible_server.md",
            "docs/design/paged_attention.md",
            "docs/design/prefix_caching.md",
            "docs/features/quantization/gguf.md",
            "docs/features/quantization/formats.md",
            "docs/configuration/conserving_memory.md",
            "docs/configuration/optimization.md",
            "docs/features/structured_outputs.md",
            "docs/features/tool_calling.md",
            "docs/features/prompt_embeds.md",
            "docs/features/reasoning_outputs.md",
            "docs/serving/parallelism_scaling.md",
            "docs/serving/online_serving/README.md",
            "docs/models/pooling_models/embed.md",
            "docs/design/metrics.md",
            "docs/usage/faq.md",
        ],
    },
}


def slug(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "__", path.lower()).strip("_")


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "llm-bench-corpus/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous_manifest = OUT / "source_manifest.json"
    if previous_manifest.exists():
        for record in json.loads(previous_manifest.read_text(encoding="utf-8")):
            previous = OUT / record["file"]
            if previous.exists():
                previous.unlink()
    records: list[dict[str, object]] = []
    for project, spec in REPOSITORIES.items():
        for path in spec["paths"]:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
            content = download(url)
            if len(content) < 300:
                raise RuntimeError(f"Document too short or missing: {url}")
            relative = Path(path)
            if str(relative).startswith("docs/en/docs/"):
                relative = relative.relative_to("docs/en/docs")
            elif str(relative).startswith("docs/"):
                relative = relative.relative_to("docs")
            document_id = f"{project}__{slug(relative.with_suffix('').as_posix())}"
            destination = OUT / f"{document_id}.md"
            destination.write_bytes(content)
            records.append(
                {
                    "document_id": document_id,
                    "title": Path(path).stem.replace("-", " ").title(),
                    "file": destination.name,
                    "source_url": f"https://github.com/{spec['repo']}/blob/{spec['commit']}/{path}",
                    "repository": spec["repo"],
                    "commit": spec["commit"],
                    "license": spec["license"],
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    (OUT / "source_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Fetched {len(records)} pinned documents into {OUT}")


if __name__ == "__main__":
    main()
