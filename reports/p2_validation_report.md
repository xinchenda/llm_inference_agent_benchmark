# P2 GPU serving validation

Generated 2026-11-23.

## Local capability audit

The local Mac is not a CUDA execution environment:

- Apple M4, macOS 26.6.2, arm64.
- `torch.cuda.is_available() == false`.
- `nvidia-smi`, `nvcc`, `nsys`, and `ncu` are absent.
- vLLM, SGLang, and Triton are not installed.
- Docker CLI is installed, but the daemon and Compose plugin are unavailable.
- Colima/Lima are installed, but no VM is running.

The full machine-readable audit is `reports/p2_capability.json`.

## vLLM implementation status

Implemented:

- `VLLMServerConfig`: validates and passes `max-model-len`, `max-num-seqs`,
  `max-num-batched-tokens`, `gpu-memory-utilization`, tensor parallel size,
  prefix caching, dtype, and CUDA Graph / eager mode selection.
- `VLLMServer`: process lifecycle, `/health` wait, `/metrics` fetch, and
  process-tree cleanup.
- `SGLangServer`: OpenAI-compatible lifecycle with `--disable-radix-cache`,
  `--enable-cache-report`, memory fraction, concurrency, and CUDA Graph controls.
- A shared backend factory/protocol for `llamacpp`, `vllm`, and `sglang`.
- `run_vllm_smoke.py`: starts vLLM, sends OpenAI-compatible streamed requests, and
  records real response timings and Prometheus samples.
- Prometheus parser: vLLM request, KV-cache, prefix-cache, token, success, TTFT,
  TPOT, and end-to-end histogram values. It preserves raw metric names and
  accepts vLLM/current and legacy GPU-cache metric names.
- Same-workload comparison gate: ratios require one matching hardware
  fingerprint; Mac/Metal and RTX/NVIDIA results stay separate.
- Pinned Dockerfile and Compose definition for vLLM 0.34.0.
- Official SGLang runtime-image runbook and RadixAttention smoke script.

Execution status: **BLOCKED**. No Linux/NVIDIA execution environment is
currently available to this project. No credentials, keys, or SSH configuration
contents were read.

```bash
python scripts/validate_vllm_environment.py \
  --output reports/p2_vllm_environment.json
# exit 2: status=blocked
```

`reports/p2_vllm_environment.json` records the concrete missing capabilities.
No vLLM latency, throughput, memory, CUDA Graph, or Serving Metrics result is
claimed.

## Triton experiment

The primary operator experiment is fused-like row-wise RMSNorm:

- PyTorch reference implementation.
- Hand-written Triton CUDA kernel.
- `torch.compile` comparison.
- fp32, fp16, and bf16 parameterized correctness gates.
- weight and accumulation use `tl.float32`; the result is cast back to the
  input/output dtype.
- `triton.testing.do_bench`, speedups, and effective bandwidth.
- Hard CUDA/Triton checks before execution.

A vector-add kernel remains as the minimal smoke test.

Local executable result:

```bash
python scripts/triton_rmsnorm.py \
  --output reports/p2_triton_rmsnorm_status.json
# exit 2: CPU reference passed; CUDA/Triton blocked; no GPU timing generated
```

The status file records:

```text
status: blocked
cuda_available: false
triton_kernel_executed: false
cpu_reference_checks: {fp32: passed, fp16: passed, bf16: passed}
```

No Triton performance result is claimed.

The local test suite executes the PyTorch reference functions and all backend
command/HTTP contract mocks. Final local test count is 80.

## Nsight Systems

Status: **NOT RUN**.

The profiling command is in `scripts/profile_vllm_nsys.sh`. It refuses to run
when `nsys` or `nvidia-smi` is absent. A timeline, kernel table, Graph launch
analysis, or memory-transfer claim will be added only from a real GPU capture.

## Hardware separation

- P0/P1 backend: llama.cpp on Apple arm64/Metal.
- P2 backend: vLLM on Linux x86_64 with an NVIDIA RTX 4090D.
- These are separate hardware fingerprints. The comparison utility intentionally
  emits no cross-hardware performance ratio.
- A valid same-hardware comparison requires running both llama.cpp CUDA and vLLM
  on the same RTX 4090D with matching model, prompt/output limits, request count,
  concurrency, seed, and quality gate.

## Required unblocking input

One of:

1. SSH access to a Linux host with an RTX 4090D and NVIDIA driver, with the user
   entering their own credentials outside this repository.
2. A temporary GPU container host with NVIDIA Container Toolkit.
3. A serving endpoint plus separate Prometheus endpoint; shell access is still
   required for Nsight and Triton.

No secret or private key should be pasted into the repository.
