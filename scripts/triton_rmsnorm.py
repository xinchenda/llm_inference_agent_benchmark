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
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("fp32", "fp16", "bf16"),
        default=("fp32", "fp16", "bf16"),
    )
    parser.add_argument("--output", default="artifacts/p2/triton_rmsnorm.json")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        cpu_checks = {}
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        for dtype_name in args.dtypes:
            dtype = dtype_map[dtype_name]
            values = torch.ones((2, 8), dtype=dtype)
            reference = rmsnorm_reference(values, torch.ones(8, dtype=dtype))
            tolerance = 1e-4 if dtype_name == "fp32" else 2e-2
            torch.testing.assert_close(
                reference,
                torch.ones_like(reference),
                atol=tolerance,
                rtol=tolerance,
            )
            cpu_checks[dtype_name] = "passed"
        blocked = {
            "kernel": "triton_rmsnorm",
            "status": "blocked",
            "cuda_available": False,
            "triton_kernel_executed": False,
            "cpu_reference_checks": cpu_checks,
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
        row_values = tl.load(
            values + row * hidden + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(row_values * row_values, axis=0) / hidden
        scale = tl.rsqrt(variance + eps)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        result = row_values * scale * weights
        tl.store(
            output + row * hidden + offsets,
            result.to(values.dtype.element_ty),
            mask=mask,
        )

    rows = args.rows
    hidden = args.hidden
    block = triton.next_power_of_2(hidden)
    generator = torch.Generator(device="cuda").manual_seed(42)
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    results = []
    for dtype_name in args.dtypes:
        dtype = dtype_map[dtype_name]
        values = torch.randn(
            (rows, hidden),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        weight = torch.randn(
            (hidden,),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
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
        tolerance = 1e-4 if dtype_name == "fp32" else 2e-2
        torch.testing.assert_close(
            output,
            reference,
            atol=tolerance,
            rtol=tolerance,
        )
        compiled = torch.compile(rmsnorm_reference, fullgraph=True)

        def triton_run(
            values=values,
            weight=weight,
            output=output,
        ):
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
        eager_ms = float(
            triton.testing.do_bench(
                lambda values=values, weight=weight: rmsnorm_reference(values, weight)
            )
        )
        compile_ms = float(
            triton.testing.do_bench(
                lambda compiled=compiled, values=values, weight=weight: compiled(
                    values,
                    weight,
                )
            )
        )
        bytes_moved = rows * hidden * values.element_size() * 3
        results.append(
            {
                "dtype": dtype_name,
                "max_abs_error": float((output - reference).abs().max().item()),
                "pytorch_eager_ms": eager_ms,
                "torch_compile_ms": compile_ms,
                "triton_ms": triton_ms,
                "triton_speedup_vs_eager": eager_ms / triton_ms,
                "triton_speedup_vs_compile": compile_ms / triton_ms,
                "effective_bandwidth_gb_s": bytes_moved / triton_ms / 1e6,
            }
        )
    result = {
        "kernel": "triton_rmsnorm",
        "status": "complete",
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "hidden": hidden,
        "results": results,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
