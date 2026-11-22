from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_bench.p2.capability import write_capability_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local GPU serving capabilities")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p2/capability.json"),
    )
    args = parser.parse_args()
    payload = write_capability_report(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
