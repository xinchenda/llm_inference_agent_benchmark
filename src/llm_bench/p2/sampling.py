from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class MetricsSampler:
    def __init__(
        self,
        fetch: Callable[[], str],
        summarize: Callable[[str], dict[str, Any]],
        interval_s: float = 0.1,
    ) -> None:
        self.fetch = fetch
        self.summarize = summarize
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.summarize(self.fetch()))
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MetricsSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
