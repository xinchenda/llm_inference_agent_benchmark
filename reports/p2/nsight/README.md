# Nsight Systems P2 trace summary

Two real vLLM traces were captured on the same RTX 4090D with Qwen3-0.6B,
an eight-request 1500-word shared-prefix workload, 16 output tokens per
request, context length 4096, and Prefix Cache enabled.

| Mode | TTFT p95 | TPOT mean | E2E p95 | `cudaLaunchKernel` calls |
| --- | ---: | ---: | ---: | ---: |
| CUDA Graph | 0.0960 s | 0.002410 s | 0.285 s | 270 |
| Eager | 0.2414 s | 0.023289 s | 0.490 s | 27,189 |

CUDA Graph reduced measured TTFT p95 by 60.24%, TPOT by 89.65%, and E2E p95
by 41.84%. The trace also recorded 447 graph-creation and 149 executable-graph
creation events. The selected Nsight API summary did not expose a separate
Graph Launch event count. The concrete launch-side evidence is the 99.01%
reduction in `cudaLaunchKernel` calls, from 27,189 to 270.

The highest-time eager kernels were BF16 GEMV, FlashAttention split-KV, and
three Ampere BF16 GEMM variants. In graph mode Nsight reports graph-node
activity differently at the default graph granularity, so reported kernel
totals are not treated as the complete arithmetic workload.

Host-to-device copies totaled approximately 1.50 GB in both modes. Eager mode
additionally reported 1,711 device-to-device copies totaling 7.35 MB and 108
memsets totaling 4.72 MB. Device-to-host activity was dominated by two large
graph-mode transfers and 183 eager-mode transfers, while aggregate transfer
time differed by less than 5%.

The machine-readable kernel distribution, API counts, memory-transfer totals,
workload metadata, and trace hashes are in `summary.json`. Raw large `.nsys-rep`
files stay outside Git.
