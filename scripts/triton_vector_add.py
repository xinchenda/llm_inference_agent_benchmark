from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from llm_bench.p2.triton_ref import vector_add_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a lightweight Triton CUDA kernel")
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="artifacts/p2/triton_vector_add.json")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        left = torch.tensor([1.0, 2.0, 3.0])
        right = torch.tensor([4.0, 5.0, 6.0])
        reference = vector_add_reference(left, right)
        expected = torch.tensor([5.0, 7.0, 9.0])
        torch.testing.assert_close(reference, expected)
        blocked = {
            "kernel": "triton_vector_add",
            "status": "blocked",
            "cuda_available": False,
            "triton_kernel_executed": False,
            "cpu_reference_check": "passed",
            "reason": "CUDA is unavailable; no GPU timing was produced",
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
    def kernel(left, right, output, size: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < size
        left_values = tl.load(left + offsets, mask=mask)
        right_values = tl.load(right + offsets, mask=mask)
        tl.store(output + offsets, left_values + right_values, mask=mask)

    size = args.size
    block = 1024
    generator = torch.Generator(device="cuda").manual_seed(42)
    left = torch.randn(size, device="cuda", generator=generator)
    right = torch.randn(size, device="cuda", generator=generator)
    output = torch.empty_like(left)
    grid = (triton.cdiv(size, block),)
    kernel[grid](left, right, output, size=size, block=block)
    reference = vector_add_reference(left, right)
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-6)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        kernel[grid](left, right, output, size=size, block=block)
    torch.cuda.synchronize()
    start.record()
    started = time.perf_counter()
    for _ in range(args.iterations):
        kernel[grid](left, right, output, size=size, block=block)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    wall_s = time.perf_counter() - started
    bytes_moved = size * 3 * left.element_size() * args.iterations
    result = {
        "kernel": "triton_vector_add",
        "device": torch.cuda.get_device_name(0),
        "size": size,
        "iterations": args.iterations,
        "max_abs_error": float((output - reference).abs().max().item()),
        "cuda_ms_total": elapsed_ms,
        "wall_s": wall_s,
        "effective_bandwidth_gb_s": bytes_moved / elapsed_ms / 1e6,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
