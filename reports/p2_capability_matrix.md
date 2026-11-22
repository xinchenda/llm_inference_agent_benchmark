# P2 capability matrix

Audited 2026-11-23 without reading or outputting credentials.

## Environment findings

- Local host: Apple M4, macOS 26.6.2, arm64, 16 GiB unified memory.
- GPU: Apple M4 Metal; PyTorch 2.15 with MPS available.
- NVIDIA: `nvidia-smi` absent; CUDA unavailable in PyTorch.
- `nvcc`, `nsys`, and `ncu`: absent.
- vLLM, SGLang, and Triton Python packages: not installed.
- PyPI availability: vLLM 0.34.0 and SGLang 0.5.10.post1 listed; no macOS
  `triton` distribution was found.
- Docker CLI 29.10.0 installed; Docker daemon unavailable.
- Docker Compose plugin and standalone `docker-compose` are absent.
- Official vLLM image tag `v0.34.0` resolves to index digest
  `sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1`;
  it was not pulled or built locally.
- Colima and Lima binaries installed; no running VM or instance.
- No currently available Linux/NVIDIA serving endpoint or execution entry was
  found in this repository.

| Capability | Local | Remote/current | Implemented | Execution status |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 4090D | No | Not currently available | GPU profile/config | Blocked |
| CUDA/NVCC | No | Not verified | Environment validator | Blocked |
| vLLM | No | Not verified | Adapter, config, smoke script, metrics parser | Code ready; run blocked |
| vLLM Serving Metrics | No live server | Not verified | `/metrics` parser and report fields | Run blocked |
| Same-GPU llama.cpp vs vLLM | No | Not verified | Hardware-separated comparison gate | Run blocked |
| Nsight Systems | No | Not verified | Profiling script and analysis runbook | Run blocked |
| Triton | No macOS wheel | Not verified | Correctness reference, CUDA kernel script | Local run impossible |
| SGLang | No | Not verified | Adapter, RadixAttention config, smoke script | Code ready; run blocked |
| Docker/NVIDIA container | CLI only; no daemon | No GPU VM | Pinned Dockerfile and Compose | Build/run blocked |

## Classification

### Already available

- Python 3.12 test environment.
- Apple Metal llama.cpp results from P0/P1.
- Docker client, Colima, and Lima executables.

### Implemented but uncompiled/unmeasured

- vLLM lifecycle adapter and command construction.
- SGLang OpenAI-compatible lifecycle adapter with RadixAttention controls.
- vLLM Prometheus metric parsing.
- Cross-hardware comparison guard.
- Docker image and Compose definition.
- Nsight Systems profiling command.
- Triton correctness/performance script.

### Blocked

- Any claim of RTX 4090D vLLM latency, throughput, memory, CUDA Graph benefit,
  Nsight timeline, or Triton kernel performance.
- Building or running the CUDA container on this Mac.
- Same-hardware llama.cpp/vLLM comparison.

No P2 GPU result is reported until a Linux/NVIDIA execution environment is
provided and the scripts complete on that environment.
