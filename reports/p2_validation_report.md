# P2 GPU serving validation

Completed on a remote Linux NVIDIA environment; local contract tests run on the
Apple M4 host.

## Measured environment

- NVIDIA GeForce RTX 4090 D, 24 GB.
- Driver 595.71.05.
- vLLM 0.34.0, PyTorch 2.14+cu130.
- Model: Qwen3-0.6B.
- vLLM `/v1/chat/completions`, `/health`, and `/metrics`: HTTP 200.
- CUDA Graph vLLM startup reserved `18.57 GiB` for KV cache, reported
  `170,496` KV-cache tokens, and estimated maximum concurrency `41.62x` for
  4,096-token requests.

No credentials, SSH configuration contents, private account, or host identity
are stored in this report.

## vLLM Serving Metrics

Raw response and metrics files were summarized into
`reports/p2/vllm_summary.json`, with source hashes and selected timestamp fields.
The original JSON was not committed to avoid retaining remote paths.

### Shared-prefix cache: 1500 words, 8 requests

| Run | Metrics cache hit rate | Response TTFT mean |
| --- | ---: | ---: |
| Prefix Cache off | null / no counter | 0.12329 s |
| Prefix Cache on | 0.86522 | 0.11799 s |

The cache-on run therefore reduced response-level mean TTFT by 4.30%.
Prometheus histogram p95 TTFT was 0.092 s with caching off and 0.096 s with
caching on, so the tail latency did not improve in this sample. The safe
conclusion is that prefix-cache reuse is confirmed by counters, while the
latency benefit is small and noisy for the 0.6B model.

The shorter shared-prefix run measured hit rate `0.57919` when enabled and no
counter when disabled; response mean TTFT changed from `0.09388 s` to
`0.09024 s`.

The returned raw files sampled running/waiting requests and KV-cache usage only
before and after the workload. Those gauges were zero while idle, so they are not
reported as under-load values. The smoke scripts now include a background
metrics sampler for future runs; that additional sampling was not executed in
the shared SGLang environment.

### CUDA Graph versus eager

| Mode | Histogram TTFT p95 | TPOT mean | E2E p95 |
| --- | ---: | ---: | ---: |
| CUDA Graph | 0.096 s | 0.002425 s | 0.285 s |
| Eager | 0.235 s | 0.023692 s | 0.490 s |

Measured reductions for CUDA Graph were 59.15% TTFT p95, 89.76% TPOT, and
41.84% E2E p95. These results are for this model, GPU, request shape, and
software stack.

## Triton RMSNorm

The GPU experiment is complete for fp32, fp16, and bf16. The kernel converts
weights and row values to `tl.float32`, accumulates normalization statistics in
fp32, and casts the result back to the input/output dtype.

| Dtype | Triton ms | Speedup vs eager | Speedup vs torch.compile | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 0.038794 | 1.94x | 1.09x | 1.91e-6 |
| fp16 | 0.022723 | 2.08x | 0.88x | 0.0078125 |
| bf16 | 0.021270 | 2.31x | 1.10x | 0.0625 |

The exact GPU result is retained in `reports/p2/triton_rmsnorm_gpu.json`.
The earlier local blocked status remains in `reports/p2_triton_rmsnorm_status.json`
as evidence that the Mac cannot execute the CUDA path.

## SGLang

SGLang 0.5.10.post1 was rerun in a clean isolated virtual environment with
PyTorch 2.9.1+cu128. This prevented the SGLang dependency set from changing the
already validated vLLM environment. The two controlled arms used Triton
attention and PyTorch sampling with Qwen3-0.6B, context 4096, CUDA Graph,
eight sequential requests, a 1500-word shared prefix, and 16 output tokens per
request.

| SGLang mode | Cache hit rate | Client TTFT mean | Client TTFT p95 | Client latency mean | Client latency p95 | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RadixAttention off | 0.00000 | 0.08138 s | 0.11792 s | 0.10853 s | 0.12261 s | 147.42 |
| RadixAttention on | 0.99573 | 0.05795 s | 0.09118 s | 0.09049 s | 0.12348 s | 176.81 |

Enabling RadixAttention reduced client mean TTFT by `28.79%`, median TTFT by
`33.16%`, mean E2E latency by `16.63%`, and increased aggregate output
throughput by `19.94%`. The p95 E2E latency was effectively unchanged and was
`0.71%` worse in this sample.

### Same-GPU framework comparison

| Framework arm | Cache hit rate | Client TTFT mean | Client TTFT p95 | Client E2E p95 | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM Prefix Cache off | 0.00000 | 0.12329 s | 0.13034 s | 0.16618 s | 103.41 |
| vLLM Prefix Cache on | 0.86522 | 0.11799 s | 0.12792 s | 0.16084 s | 106.51 |
| SGLang RadixAttention off | 0.00000 | 0.08138 s | 0.11792 s | 0.12261 s | 147.42 |
| SGLang RadixAttention on | 0.99573 | 0.05795 s | 0.09118 s | 0.12348 s | 176.81 |

Both frameworks used the same GPU, model, request count, prefix material,
output limit, and context length. Their software stacks were intentionally
isolated and differed, so the table compares complete serving configurations
rather than a single kernel. vLLM reports prefix-cache hit counters; SGLang
reports `sglang:cache_hit_rate`. Those metrics are not treated as numerically
interchangeable.

## Nsight Systems

Nsight Systems 2025.3.1 captured two real vLLM traces on the same RTX 4090D.
The workload was the same eight-request shared-prefix run used for the serving
tests.

| Mode | TTFT p95 | TPOT mean | E2E p95 | `cudaLaunchKernel` calls |
| --- | ---: | ---: | ---: | ---: |
| CUDA Graph | 0.0960 s | 0.002410 s | 0.285 s | 270 |
| Eager | 0.2414 s | 0.023289 s | 0.490 s | 27,189 |

CUDA Graph reduced TTFT p95 by `60.24%`, TPOT by `89.65%`, and E2E p95 by
`41.84%`. It also reduced recorded host kernel-launch calls by `99.01%`.
The trace contains 447 graph-creation and 149 executable-graph creation
events. The selected API summary did not expose a separate Graph Launch event
count, so that value is not inferred.

The eager trace's highest-time kernels were BF16 GEMV, FlashAttention split-KV,
and three Ampere BF16 GEMM variants. Graph-node kernels are represented
differently at the default trace granularity, so reported kernel totals are not
interpreted as the complete arithmetic workload. Host-to-device traffic was
approximately `1.50 GB` in both traces. Eager mode additionally reported
`7.35 MB` of device-to-device copies and `4.72 MB` of memset activity.

Machine-readable results and trace hashes are in
[`reports/p2/nsight/summary.json`](p2/nsight/summary.json). Raw `.nsys-rep`
files remain outside Git.

## Local contract tests

- `.venv/bin/python -m pytest`: **82 passed** after the measured-result update.
- `.venv/bin/ruff check src tests scripts`: passed.
- Mocked vLLM tests cover request payload, health success, health timeout,
  metrics fetch, metrics timeout, missing fields, histogram-derived TTFT/TPOT,
  and backend factory selection.
- Local GPU validation and blocked Triton status are retained separately from
  the remote success files.

## Hardware separation

- P0/P1: llama.cpp on Apple arm64/Metal.
- P2: vLLM and Triton on Linux x86_64 with NVIDIA RTX 4090D.
- No cross-hardware latency or throughput ratio is produced.
- A same-GPU llama.cpp CUDA versus vLLM comparison is still required for a
  direct backend ratio.
