# P2 GPU serving runbook

## Scope

P2 covers a real vLLM path on an NVIDIA Linux host, serving metrics, comparable
workload execution, optional Nsight Systems profiling, a lightweight Triton
kernel experiment, and an optional SGLang comparison.

The current Mac cannot execute NVIDIA CUDA workloads. No RTX 4090D execution
environment is currently available, and this repository contains no reusable
endpoint or credential.

## Capability categories

| Capability | Local Mac | Current remote | Can implement without GPU | Status |
| --- | --- | --- | --- | --- |
| NVIDIA GPU / driver | No NVIDIA GPU | RTX 4090D 24 GB, driver 595.71.05 | GPU config | Executed remotely |
| CUDA / NVCC | Not available | CUDA runtime used by torch 2.14+cu130 | Python runtime | Executed remotely |
| vLLM | Not installed locally | vLLM 0.34.0 | Adapter, config, smoke script | Executed remotely |
| SGLang | Not installed locally | 0.5.10.post1 installed in shared env | Adapter/config/smoke | Timing pending in clean env |
| Triton | No macOS distribution found | Real GPU run | Reference code and GPU script | Executed remotely |
| Nsight Systems | Not installed | Not verified | Runbook and script | GPU execution blocked |
| Docker | CLI installed; daemon and Compose plugin absent | No GPU VM | Dockerfile/Compose | Build and run blocked |
| Linux VM tooling | Colima/Lima installed, no instance | No | Setup commands | Not started |

No passwords, private keys, API keys, or SSH configuration are read by the
project audit.

## Required remote environment

Provide one of:

1. A Linux x86_64 GPU host with an RTX 4090 or RTX 4090D, NVIDIA driver, Docker
   with NVIDIA Container Toolkit, and enough disk for the model and image.
2. An SSH-accessible Linux host where the user enters credentials interactively.
   Do not send credentials in a prompt or commit them to the repository.
3. A temporary GPU instance with an HTTPS model-serving endpoint and a separate
   metrics endpoint. For full Nsight and Triton work, shell access is still
   required.

## Validation sequence

```bash
python scripts/audit_gpu_capabilities.py
python scripts/validate_vllm_environment.py
python scripts/run_vllm_smoke.py --candidate qwen3_0_6b_prefix
```

Expected evidence:

- `nvidia-smi` GPU name and 24 GB VRAM.
- vLLM version and command line.
- `/health` success.
- OpenAI-compatible chat completion.
- `/metrics` with `vllm:` prefix and prefix-cache counters.

## vLLM configuration

The example matrix is `configs/vllm_backends.yaml`.

Required search dimensions:

- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--max-model-len`
- `--gpu-memory-utilization`
- `--enable-prefix-caching` / `--no-enable-prefix-caching`
- `--tensor-parallel-size`
- CUDA Graph enabled by default; `--enforce-eager` is used only as an explicit
  ablation.

Continuous batching is the vLLM V1 scheduler behavior and cannot be disabled by
this adapter.

## Docker path

```bash
docker compose -f docker/vllm/compose.yaml up --build
```

The image is pinned to `vllm/vllm-openai:v0.34.0` and digest
`sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1`.
The image build was not run on the Mac because the Docker daemon is unavailable
and there is no NVIDIA GPU.

## Comparable llama.cpp/vLLM workloads

Run both backends on the same NVIDIA GPU with the same prompt lengths, output
token cap, request counts, concurrency, seed, and SLO.

The comparison gate in `llm_bench.p2.comparison` refuses to produce ratios when
hardware fingerprints differ. Mac/Metal llama.cpp results remain separate from
RTX 4090D vLLM results.

## Nsight Systems

Create a profile only on the GPU host:

```bash
bash scripts/profile_vllm_nsys.sh Qwen/Qwen3-0.6B artifacts/p2/nsight/vllm_trace
```

Generate workload traffic while the server starts. Analyze:

- CUDA kernel time and Graph launches.
- Prefill/decode separation and NVTX ranges.
- CPU gaps in request handling.
- Memory transfer and synchronization.

The repository will store the `.nsys-rep`, exported tables, hardware fingerprint,
command, and workload settings. A timeline is not included if it was not
actually captured.

## Optional SGLang comparison

SGLang 0.5.10.post1 was installed in the shared P2 environment, but that changed
PyTorch to 2.9.1 and may have invalidated vLLM. No SGLang timing is claimed from
that environment. In a clean GPU environment, first capture:

```bash
sglang --version
python -m sglang.launch_server --help
```

Only after verifying the installed version's flags should the same fixed
workloads be added. SGLang metrics use their own names and must not be mapped to
the vLLM field names without checking the installed version. A separate hardware
fingerprint and result file are required.

Use the official runtime image without compiling SGLang from source:

```bash
docker run --gpus all --ipc=host --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest-runtime \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --max-running-requests 8 \
  --chunked-prefill-size 8192 \
  --enable-cache-report
```

RadixAttention is enabled by default in SGLang. Add `--disable-radix-cache`
for the controlled off arm. Record the exact image digest and installed package
version in the run manifest before comparing results.

```bash
python scripts/run_sglang_smoke.py \
  --candidate qwen3_0_6b_radix
```

## Cross-hardware migration

- The P0/P1 llama.cpp measurements are Apple arm64/Metal results.
- P2 vLLM measurements must be Linux x86_64/NVIDIA RTX 4090D results.
- The same workload and settings make the experiment comparable in design, but
  hardware and runtime differences prevent a single combined throughput number.
- A ratio is emitted only when both backends were actually run on one matching
  hardware fingerprint.
- Moving the service to another GPU, driver, CUDA runtime, vLLM version, or
  quantization requires repeating correctness and performance measurements.

## Triton experiment

On the GPU host:

```bash
python scripts/triton_rmsnorm.py \
  --rows 4096 --hidden 1024 \
  --output artifacts/p2/triton_rmsnorm.json
```

The RMSNorm experiment compares PyTorch eager, `torch.compile`, and a hand-written
Triton kernel. It:

- Compares Triton and PyTorch with `torch.testing.assert_close`.
- Benchmarks all three paths with `triton.testing.do_bench`.
- Reports microseconds, speedups, and effective bandwidth.
- Refuses to produce results when CUDA or Triton is unavailable.

The smaller vector-add smoke remains available as
`python scripts/triton_vector_add.py`.

## P2 acceptance evidence

- Capability JSON and matrix.
- VLLM command/config and unit contract tests.
- Real vLLM smoke output and Prometheus metrics.
- Same-hardware llama.cpp/vLLM comparison or clearly separate rows.
- Nsight timeline and analysis, if captured.
- Triton correctness/performance JSON.
- Optional SGLang result with a separate version and hardware fingerprint.
- README and validation report updated only after real execution.
