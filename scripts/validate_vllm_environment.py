from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Linux/NVIDIA/vLLM readiness")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = {
        "linux": platform.system() == "Linux",
        "x86_64": platform.machine() in {"x86_64", "amd64"},
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "nvcc": bool(shutil.which("nvcc")),
        "vllm": bool(importlib.util.find_spec("vllm")),
        "triton": bool(importlib.util.find_spec("triton")),
        "nsys": bool(shutil.which("nsys")),
        "ncu": bool(shutil.which("ncu")),
    }
    gpu = None
    if checks["nvidia_smi"]:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu = process.stdout.strip() if process.returncode == 0 else None
    required = all(checks[name] for name in ("linux", "nvidia_smi", "vllm"))
    result = {
        "checks": checks,
        "gpu": gpu,
        "required_vllm_execution_ready": required,
        "status": "ready" if required else "blocked",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if required else 2)


if __name__ == "__main__":
    main()
