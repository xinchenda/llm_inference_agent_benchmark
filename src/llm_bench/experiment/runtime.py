from __future__ import annotations

import errno
import fcntl
import os
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil


class ResourceLock:
    """Exclusive local resource lock for model processes and experiment runs."""

    def __init__(self, path: Path, timeout_s: float = 30) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self._handle = None

    def __enter__(self) -> ResourceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(f"pid={os.getpid()} acquired={time.time():.6f}\n")
                self._handle.flush()
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(
                        f"Could not acquire experiment lock: {self.path}"
                    ) from None
                time.sleep(0.1)

    def __exit__(self, *_: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.05)
        return client.connect_ex((host, port)) != 0


def allocate_port(start: int = 18080, end: int = 18120) -> int:
    for port in range(start, end + 1):
        if port_is_free(port):
            return port
    raise OSError(errno.EADDRINUSE, f"No free port in range {start}-{end}")


def terminate_process_tree(process: Any, timeout_s: float = 10) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        processes = parent.children(recursive=True)
        processes.append(parent)
    except psutil.Error:
        processes = []
    for item in processes:
        try:
            item.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=timeout_s)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    try:
        process.wait(timeout=1)
    except Exception:
        pass


OOM_PATTERN = re.compile(
    r"\b(?:out of memory|oom|insufficient memory|cannot allocate memory|"
    r"failed to allocate|memory allocation failed)\b",
    re.IGNORECASE,
)
CRASH_PATTERN = re.compile(
    r"\b(?:connection reset|broken pipe|unexpected eof|segmentation fault|"
    r"process exited|server exited|terminated unexpectedly)\b",
    re.IGNORECASE,
)
INVALID_PATTERN = re.compile(
    r"\b(?:invalid|unsupported|must be|validation|exceeds the available context)\b"
    r"|HTTP 400",
    re.IGNORECASE,
)


def classify_failure(error: str | BaseException) -> str:
    message = str(error)
    if isinstance(error, TimeoutError) or "timeout" in message.lower():
        return "timeout"
    if OOM_PATTERN.search(message):
        return "oom"
    if "startup" in message.lower() or "health" in message.lower():
        return "service_start_failure"
    if CRASH_PATTERN.search(message):
        return "process_crash"
    if INVALID_PATTERN.search(message):
        return "invalid_config"
    if "port" in message.lower() or "address already in use" in message.lower():
        return "port_occupied"
    return "runtime_error"


def retryable_failure(failure_type: str) -> bool:
    return failure_type in {"service_start_failure", "process_crash", "timeout"}


@contextmanager
def clean_process(process: Any):
    try:
        yield process
    finally:
        terminate_process_tree(process)
