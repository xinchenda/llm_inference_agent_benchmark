from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _command_path(name: str) -> str | None:
    return shutil.which(name)


def _probe_gpu() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found"}
    process = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    return {
        "available": process.returncode == 0,
        "command": executable,
        "returncode": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def _docker_daemon() -> dict[str, Any]:
    executable = shutil.which("docker")
    if not executable:
        return {"available": False, "reason": "docker not found"}
    process = subprocess.run(
        [executable, "info", "--format", "{{json .ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    return {
        "available": process.returncode == 0,
        "compose_plugin": shutil.which("docker-compose") is not None
        or _docker_compose_plugin_available(executable),
        "returncode": process.returncode,
        "server_version": process.stdout.strip(),
        "error": process.stderr.strip()[:500],
    }


def _docker_compose_plugin_available(executable: str) -> bool:
    process = subprocess.run(
        [executable, "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    return process.returncode == 0


def collect_capabilities() -> dict[str, Any]:
    package_names = ("vllm", "sglang", "triton", "torch")
    packages = {
        name: bool(importlib.util.find_spec(name)) for name in package_names
    }
    tools = {
        name: _command_path(name)
        for name in ("nvidia-smi", "nvcc", "nsys", "ncu", "docker", "colima", "limactl")
    }
    is_linux = platform.system() == "Linux"
    docker_state = _docker_daemon()
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "tools": tools,
        "python_packages": packages,
        "nvidia_gpu": _probe_gpu(),
        "docker_daemon": docker_state,
        "classification": {
            "local_cuda_execution": bool(
                is_linux and tools["nvidia-smi"] and packages["vllm"]
            ),
            "nsight_execution": bool(tools["nsys"] and tools["nvidia-smi"]),
            "triton_execution": bool(packages["triton"] and tools["nvidia-smi"]),
            "docker_gpu_execution": bool(
                tools["docker"]
                and docker_state["available"]
                and docker_state["compose_plugin"]
            ),
        },
        "notes": [
            "A GPU-capable Mac is not assumed to provide CUDA.",
            "Historical RTX 4090D training does not imply a reachable serving host.",
            "Credentials and SSH configuration are intentionally excluded from this audit.",
        ],
    }


def write_capability_report(destination: Path) -> dict[str, Any]:
    payload = collect_capabilities()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
