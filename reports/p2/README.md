# P2 remote result summaries

The remote raw files remain outside Git in `.p2_remote/extracted/artifacts/p2/`.
The project stores sanitized summaries and SHA256 hashes instead of remote model
paths or host identity.

Regenerate the tracked summaries from returned raw files:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_p2_remote.py
```

Files:

- `vllm_summary.json`: five sanitized vLLM runs, response timing aggregates,
  selected Serving Metrics, KV-cache capacity facts from startup logs, source
  hashes, and safe command flags.
- `derived_comparisons.json`: Prefix Cache and CUDA Graph/eager differences
  computed from the saved runs.
- `triton_rmsnorm_gpu.json`: exact Triton RMSNorm result and source hash.
- `sglang_summary.json`: sanitized SGLang RadixAttention run and source hash.
- `remote_environment.json`: generic GPU/runtime identity for the P2 run.

Measured scenarios:

- Short shared-prefix cache on/off.
- 1500-word shared-prefix cache on/off with eight requests.
- CUDA Graph and eager execution with the same long-prefix workload.
- Triton RMSNorm at fp32/fp16/bf16.
- SGLang RadixAttention on Qwen3-0.6B.

Limitations:

- vLLM OpenAI responses did not expose request-level `timings.cache_n`; cache
  evidence therefore comes from the `/metrics` prefix-cache counters.
- Prefix Cache reduced mean response TTFT by 4.30%, while Prometheus p95 TTFT
  increased slightly for this 0.6B workload.
- The CUDA Graph measurements are from a single local configuration and are not
  a universal speedup claim.
- SGLang was measured successfully. Its later installation changed the shared
  environment's PyTorch version, so no vLLM rerun is performed in that shared
  environment.
- Nsight Systems and same-GPU llama.cpp/vLLM comparison are not present in these
  files.
