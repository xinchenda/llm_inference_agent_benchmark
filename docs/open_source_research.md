# Open-source survey and design decisions

Survey date: 2026-11-21. Stars are a maintenance-signal snapshot, not evidence
of quality. Repository activity and official source/documentation were checked.

| Area | Project | License | Observed signal | Decision |
|---|---|---:|---|---|
| Local GGUF inference | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Active; Apple Metal, CPU, CUDA, Vulkan and multiple accelerators | **Use** as the measured backend |
| Local model daemon | [Ollama](https://github.com/ollama/ollama) | MIT | Active; convenient model lifecycle and API | Not used for measurement; another daemon layer obscures the direct server lifecycle |
| Apple-native inference | [MLX-LM](https://github.com/ml-explore/mlx-lm) | MIT | Active; Apple-silicon focused | Keep as an Apple backend extension; do not duplicate the current GGUF path |
| GPU serving | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | Active; PagedAttention, continuous batching, metrics, quantization | Adapter path for Linux/NVIDIA; not claimed as executed on this M4 |
| GPU serving | [SGLang](https://github.com/sgl-project/sglang) | Apache-2.0 | Active; optimized serving and structured generation | Not selected for the first local baseline |
| RAG orchestration | [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | Active; broad document/index integrations | Not needed for a 64-document benchmark; thin local index gives tighter control |
| Agent/RAG orchestration | [LangChain](https://github.com/langchain-ai/langchain) | MIT | Active; broad agent and integration ecosystem | Avoided for the measured core to keep dependency and behavior surface small |
| RAG evaluation | [Ragas](https://github.com/vibrantlabsai/ragas) | Apache-2.0 | Active; automated RAG metrics | Not used as the only judge; the project keeps fixed labels and manual review |
| Model runtime | [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Active; broad model support | Used indirectly for embedding/reranking tooling, not the measured chat backend |
| Embeddings/reranking | [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | Apache-2.0 | Active; mature dense retrieval and cross-encoder APIs | **Use** for dense embeddings and reranking |

## Why llama.cpp for the measured path

The target machine is a 16 GB Apple M4. `llama.cpp` provides a native Metal
backend, loads GGUF files directly, exposes an OpenAI-compatible server, reports
token timing data, and lets the benchmark own process startup and shutdown.
That makes cold-start and process-RSS measurement possible without a model
training stack. It is also the same ecosystem used by the selected Qwen3 weight
cards.

## Why vLLM remains an adapter rather than an invented result

vLLM is the stronger choice for a Linux/NVIDIA deployment with concurrency,
PagedAttention, tensor parallelism, and Prometheus metrics. The local Apple
environment does not provide the same CUDA serving conditions. The project
therefore documents a GPU extension path and does not report “vLLM numbers”
from this machine.

## Why the RAG and Agent layers are thin

The benchmark needs visible retrieval ranks, document-level citations, tool
arguments, timeout status, and deterministic failure inspection. A local
implementation of BM25, reciprocal-rank fusion, cross-encoder reranking, and a
bounded state machine keeps those artifacts explicit. Framework integration is
useful when document connectors or complex planners dominate the requirement;
it is not required to prove this benchmark's core behavior.

## References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- Ollama: https://github.com/ollama/ollama
- MLX-LM: https://github.com/ml-explore/mlx-lm
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- LlamaIndex: https://github.com/run-llama/llama_index
- LangChain: https://github.com/langchain-ai/langchain
- Ragas: https://github.com/vibrantlabsai/ragas
- Transformers: https://github.com/huggingface/transformers
- Sentence Transformers: https://github.com/huggingface/sentence-transformers

## llama.cpp serving controls

The P0/P1 optimizer uses the installed `llama-server --help` and upstream
llama.cpp documentation/source to validate the following controls:

- `--batch-size` and `--ubatch-size`; batch size must be divisible by ubatch
  size.
- `--flash-attn=on|off|auto`.
- `--cache-type-k` and `--cache-type-v`; the controlled local matrix uses f16
  and q8_0.
- `--parallel` for server slots. The total context is divided across slots, so a
  long request needs enough context per slot.
- Request-level `cache_prompt` for prompt-prefix reuse.

Upstream source also requires Flash Attention for a quantized V cache. The
candidate validator enforces this rule before process startup.

Sources:

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp

No upstream implementation is claimed as original. The project contribution is
the experiment specification, candidate validation, deterministic executor,
immutable ledger, quality-first gates, RSI reflection memory, equal-budget
policy comparison, and reporting.

