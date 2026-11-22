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

SGLang 0.5.10.post1 completed a RadixAttention run with the same Qwen3-0.6B
model, eight requests, and a 1500-word shared prefix. Context was 4096, CUDA
Graph was enabled, and `--enable-cache-report` was active.

| Backend | Cache hit metric | Client TTFT mean | Metrics TTFT p50 | Metrics TTFT p95 | E2E p95 | Cold start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM + Prefix Cache | 0.86522 | 0.11799 s | 0.07333 s | 0.096 s | 0.285 s | 34.25 s |
| SGLang + RadixAttention | 0.99573 | 0.12232 s | 0.06429 s | 0.355 s | 0.355 s | 56.72 s |

SGLang used `sglang:cache_hit_rate`; vLLM used its `prefix_cache_hits` and
`prefix_cache_queries` counters. The two values measure similar concepts but
come from different framework metric implementations.

SGLang had the lower p50 TTFT, while vLLM had materially lower p95 TTFT and E2E
tail latency. Its server log reports `#cached-token=3032` for repeated requests
and eight successful `/v1/chat/completions` responses.

The SGLang installation changed PyTorch to 2.9.1 in the shared environment and
may have invalidated vLLM after the measurements. The recorded vLLM run is
preserved; vLLM will not be rerun until its dedicated 0.34.0 environment is
restored.

## Nsight Systems

Nsight Systems 2025.3.1 is now installed on the GPU host. A vLLM profile has not
yet been captured. No timeline, kernel table, Graph-launch analysis, or
memory-transfer claim is included.

`reports/p2_nsight_status.json` records the missing evidence. The executable
command remains in `scripts/profile_vllm_nsys.sh`.

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
