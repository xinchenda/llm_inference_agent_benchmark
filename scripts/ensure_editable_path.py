#!/usr/bin/env python3
"""Ensure the editable src path stays visible on this non-ASCII macOS path."""

from __future__ import annotations

import site
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site_dir = Path(site.getsitepackages()[0])
marker = site_dir / "zz_llm_bench_src.pth"
expected = str(ROOT / "src")
if not marker.exists() or marker.read_text(encoding="utf-8").strip() != expected:
    marker.write_text(expected + "\n", encoding="utf-8")
if hasattr(stat, "UF_HIDDEN"):
    # This Python build skips macOS-hidden .pth files during startup.
    for path in site_dir.glob("*.pth"):
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            path.chflags(flags & ~stat.UF_HIDDEN)
sys.path.insert(0, expected)
