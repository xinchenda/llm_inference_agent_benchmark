from __future__ import annotations

import json
from typing import Any


class OutputFormatError(ValueError):
    """Raised when a model emits malformed JSON or the wrong top-level shape."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model response that must contain exactly one JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise OutputFormatError("Unclosed Markdown code fence")
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputFormatError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise OutputFormatError("Expected a JSON object")
    return parsed
