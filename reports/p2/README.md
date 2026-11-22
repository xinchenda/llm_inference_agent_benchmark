# P2 remote result summaries

The remote-serving raw files remain outside Git under
`.p2_remote/extracted/artifacts/p2/`. The paired isolated SGLang raw files use
`.p2_remote/extracted/sglang/`. The project stores sanitized summaries and
SHA256 hashes instead of remote model paths or host identity.

Regenerate the tracked summaries from returned raw files:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_p2_remote.py
```

Files:

- `vllm_summary.json`: five sanitized vLLM runs, response timing aggregates,
  selected Serving Metrics, KV-cache capacity facts from startup logs, source
  hashes, and safe command flags.
- `derived_comparisons.json`: Prefix Cache and CUDA Graph/eager differences
  computed from the saved runs, including isolated SGLang on/off and
  same-GPU framework comparisons.
- `triton_rmsnorm_gpu.json`: exact Triton RMSNorm result and source hash.
- `sglang_summary.json`: paired isolated SGLang RadixAttention on/off runs,
  client latency quantiles, throughput, cache metrics, and source hashes.
- `nsight/summary.json`: real CUDA Graph and eager traces with kernel, API,
  graph-event, and memory-transfer summaries plus raw trace hashes.
- `remote_environment.json`: generic GPU/runtime identity for the P2 run.

Measured scenarios:

- Short shared-prefix cache on/off.
- 1500-word shared-prefix cache on/off with eight requests.
- CUDA Graph and eager execution with the same long-prefix workload.
- Triton RMSNorm at fp32/fp16/bf16.
- SGLang RadixAttention on and off in a clean isolated environment with
  Qwen3-0.6B.
- Nsight Systems CUDA Graph versus eager traces on Qwen3-0.6B.

Limitations:

- vLLM OpenAI responses did not expose request-level `timings.cache_n`; cache
  evidence therefore comes from the `/metrics` prefix-cache counters.
- Prefix Cache reduced mean response TTFT by 4.30%, while Prometheus p95 TTFT
  increased slightly for this 0.6B workload.
- The CUDA Graph measurements are from a single local configuration and are not
  a universal speedup claim.
- SGLang RadixAttention reduced client mean TTFT by 28.79% and increased
  measured output throughput by 19.94%; p95 E2E latency was effectively
  unchanged.
- SGLang uses Triton attention and PyTorch sampling in the isolated run. The
  vLLM and SGLang software stacks differ, so framework comparisons describe
  complete configurations rather than isolated kernel effects.
- Nsight captured CUDA Graph and eager traces. The selected API summary did not
  expose a separate Graph Launch event count, so the report uses the 99.01%
  reduction in host `cudaLaunchKernel` calls as the launch-side evidence.
- Same-GPU llama.cpp/vLLM comparison is not present in these files.
