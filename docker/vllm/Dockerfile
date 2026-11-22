FROM vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

USER root
WORKDIR /opt/llm-inference-agent-benchmark

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps --no-cache-dir -e .

ENV HF_HUB_DISABLE_TELEMETRY=1
ENTRYPOINT ["vllm", "serve"]
