#!/usr/bin/env python3
"""Capture a deterministic, human-readable package environment record."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    output = Path("reports/environment-freeze.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
