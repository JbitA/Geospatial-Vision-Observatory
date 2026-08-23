#!/usr/bin/env python3
"""Verify that built wheels contain the runtime files required by the published project."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_WHEEL_MEMBERS = {
    "geo_vision/__init__.py",
    "geo_vision/api.py",
    "geo_vision/source.py",
    "geo_vision/storage.py",
    "geo_vision/dashboard.html",
    "geo_vision/dashboard.css",
    "geo_vision/dashboard.js",
    "geo_vision/ml/train.py",
    "geo_vision/ml/inference.py",
    "geo_vision/ml/integrity.py",
}


def verify_wheel(path: Path) -> None:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"wheel does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_WHEEL_MEMBERS - names)
        if missing:
            raise ValueError(f"wheel is missing required runtime files: {missing}")
        metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one distribution METADATA file")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    if "Requires-Python: >=3.12,<3.13" not in metadata:
        raise ValueError("wheel Python compatibility metadata does not match the repository contract")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = sorted(args.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {args.dist}, found {len(wheels)}")
    try:
        verify_wheel(wheels[0])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(wheels[0])


if __name__ == "__main__":
    main()
