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
- A remote Linux RTX 4090D session was used for the P2 vLLM and Triton
  measurements. No host identity is stored in the repository.

| Capability | Local | GPU host | Implemented | Execution status |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 4090D | No | Yes, 24 GB; driver 595.71.05 | GPU profile/config | Executed |
| CUDA/NVCC | No | CUDA runtime used by torch 2.14+cu130 | Environment validator | Executed through Python runtime |
| vLLM | No | 0.34.0 | Adapter, config, smoke script, metrics parser | Executed |
| vLLM Serving Metrics | No live local server | `/metrics` HTTP 200 | `/metrics` parser and report fields | Executed |
| Same-GPU llama.cpp vs vLLM | No | Not verified | Hardware-separated comparison gate | Run blocked |
| Nsight Systems | No | 2025.3.1; two real vLLM traces captured | Profiling script, summaries, and runbook | Executed |
| Triton | No macOS wheel | Real fp32/fp16/bf16 GPU run | Correctness reference, CUDA kernel script | Executed |
| SGLang | No | 0.5.10.post1 in an isolated environment | Adapter, RadixAttention config, smoke script | Executed on/off |
| Docker/NVIDIA container | CLI only; no daemon | No GPU VM | Pinned Dockerfile and Compose | Build/run blocked |

## Classification

### Already available

- Python 3.12 test environment.
- Apple Metal llama.cpp results from P0/P1.
- Docker client, Colima, and Lima executables.

### Implemented and measured

- vLLM Prometheus metric parsing.
- vLLM Prefix Cache on/off and CUDA Graph/eager comparisons.
- SGLang RadixAttention on/off with the same 8-request shared-prefix workload
  in an isolated environment.
- Nsight Systems CUDA Graph versus eager vLLM traces.
- Triton RMSNorm correctness and performance for fp32/fp16/bf16.

### Implemented but not measured

- Docker image build and Compose launch.
- Same-GPU llama.cpp CUDA versus vLLM comparison.

### Blocked

- Same-hardware llama.cpp/vLLM comparison.
- Docker GPU build on this Mac.
