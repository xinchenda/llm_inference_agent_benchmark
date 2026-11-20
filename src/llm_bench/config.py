from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    override = os.getenv("LLM_BENCH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "default.yaml").exists():
        return current
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "configs" / "default.yaml").exists():
        return source_candidate
    raise RuntimeError("Cannot locate project root. Run from the repository or set LLM_BENCH_ROOT.")


PROJECT_ROOT = _find_project_root()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], dotted_key: str) -> Path:
    value = config
    for part in dotted_key.split("."):
        value = value[part]
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_models(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return load_yaml(model_path)


def parse_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse dotted CLI overrides such as eval.max_questions=5."""
    result: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value form: {item}")
        dotted, raw = item.split("=", 1)
        parsed: Any = yaml.safe_load(raw)
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parsed
    return result


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    return _merge(config, parse_overrides(overrides))
