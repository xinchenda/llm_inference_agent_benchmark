from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_bench.p2.triton_ref import rmsnorm_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm on CUDA")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--output", default="artifacts/p2/triton_rmsnorm.json")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        values = torch.ones((2, 8))
        reference = rmsnorm_reference(values, torch.ones(8))
        torch.testing.assert_close(reference, torch.ones_like(reference), atol=1e-4, rtol=1e-4)
        blocked = {
            "kernel": "triton_rmsnorm",
            "status": "blocked",
            "cuda_available": False,
            "triton_kernel_executed": False,
            "cpu_reference_check": "passed",
            "reason": "CUDA is unavailable; no GPU benchmark was produced",
        }
        output_path.write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(blocked, indent=2))
        raise SystemExit(2)

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise SystemExit("Triton is not installed on this GPU host") from exc

    @triton.jit
    def rmsnorm_kernel(
        values,
        weight,
        output,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        row_values = tl.load(values + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(row_values * row_values, axis=0) / hidden
        scale = tl.rsqrt(variance + eps)
        weights = tl.load(weight + offsets, mask=mask, other=0.0)
        tl.store(
            output + row * hidden + offsets,
            row_values * scale * weights,
            mask=mask,
        )

    rows = args.rows
    hidden = args.hidden
    block = triton.next_power_of_2(hidden)
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn((rows, hidden), device="cuda", generator=generator)
    weight = torch.randn((hidden,), device="cuda", generator=generator)
    output = torch.empty_like(values)
    rmsnorm_kernel[(rows,)](
        values,
        weight,
        output,
        rows=rows,
        hidden=hidden,
        eps=1e-6,
        block=block,
    )
    reference = rmsnorm_reference(values, weight)
    torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-4)
    compiled = torch.compile(rmsnorm_reference, fullgraph=True)

    def triton_run():
        rmsnorm_kernel[(rows,)](
            values,
            weight,
            output,
            rows=rows,
            hidden=hidden,
            eps=1e-6,
            block=block,
        )

    triton_ms = float(triton.testing.do_bench(triton_run))
    eager_ms = float(triton.testing.do_bench(lambda: rmsnorm_reference(values, weight)))
    compile_ms = float(triton.testing.do_bench(lambda: compiled(values, weight)))
    bytes_moved = rows * hidden * values.element_size() * 3
    result = {
        "kernel": "triton_rmsnorm",
        "status": "complete",
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "hidden": hidden,
        "max_abs_error": float((output - reference).abs().max().item()),
        "pytorch_eager_ms": eager_ms,
        "torch_compile_ms": compile_ms,
        "triton_ms": triton_ms,
        "triton_speedup_vs_eager": eager_ms / triton_ms,
        "triton_speedup_vs_compile": compile_ms / triton_ms,
        "effective_bandwidth_gb_s": bytes_moved / triton_ms / 1e6,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
