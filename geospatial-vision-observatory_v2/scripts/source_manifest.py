#!/usr/bin/env python3
"""Write or verify the deterministic SHA-256 source manifest used by GitHub releases."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".showcase",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "data",
    "dist",
    "build",
}
EXCLUDED_NAMES = {".env", ".coverage", "release-validation.json", MANIFEST.name}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3"}
EXCLUDED_REPORT_SUFFIXES = {".tif", ".tiff", ".npz"}
_LINE = re.compile(r"^([0-9a-f]{64})  ([^\\\r\n]+)$")


def _include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    if (
        len(relative.parts) >= 2
        and relative.parts[0] == "reports"
        and path.suffix.lower() in EXCLUDED_REPORT_SUFFIXES
    ):
        return False
    return True


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and _include(path):
            if path.is_symlink():
                raise ValueError(f"source manifest refuses symlink: {path.relative_to(ROOT)}")
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in source_files()
    }


def write_manifest() -> None:
    expected = build_manifest()
    MANIFEST.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in expected.items()),
        encoding="utf-8",
        newline="\n",
    )


def read_manifest() -> dict[str, str]:
    if not MANIFEST.is_file() or MANIFEST.is_symlink():
        raise ValueError("SOURCE_MANIFEST.sha256 must be a regular non-symlink file")
    result: dict[str, str] = {}
    for index, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid source manifest line {index}")
        digest, relative = match.groups()
        if relative in result:
            raise ValueError(f"duplicate source manifest path: {relative}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative != path.as_posix():
            raise ValueError(f"unsafe source manifest path: {relative}")
        result[relative] = digest
    return result


def verify_manifest() -> None:
    recorded = read_manifest()
    expected = build_manifest()
    missing = sorted(set(expected) - set(recorded))
    extra = sorted(set(recorded) - set(expected))
    mismatched = sorted(
        relative
        for relative in set(expected) & set(recorded)
        if expected[relative] != recorded[relative]
    )
    if missing or extra or mismatched:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        if mismatched:
            details.append("hash mismatch: " + ", ".join(mismatched))
        raise ValueError("source manifest verification failed; " + "; ".join(details))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="regenerate SOURCE_MANIFEST.sha256")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        print(MANIFEST)
        return
    verify_manifest()
    print(f"source manifest verified: {len(read_manifest())} files")


if __name__ == "__main__":
    main()
