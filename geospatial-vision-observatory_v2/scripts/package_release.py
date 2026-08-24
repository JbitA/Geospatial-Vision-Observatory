#!/usr/bin/env python3
"""Build a deterministic, cache-free source handoff archive with an integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "geospatial-vision-observatory"
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
EXCLUDED_NAMES = {".env", ".coverage", "release-validation.json", "SOURCE_MANIFEST.sha256"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3"}
# Raw/large inference artifacts are reproducible locally and should not enter a GitHub source handoff.
EXCLUDED_REPORT_SUFFIXES = {".tif", ".tiff", ".npz"}
FIXED_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or not _include(path):
            continue
        if path.is_symlink():
            raise SystemExit(f"release source must not contain symlinks: {path.relative_to(ROOT)}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _run_preflight() -> None:
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "src"],
        [sys.executable, "-m", "coverage", "erase"],
        [sys.executable, "-m", "coverage", "run", "--branch", "-m", "pytest", "-q"],
        [sys.executable, "-m", "coverage", "report", "--fail-under=80"],
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "-m", "pip_audit"],
        [sys.executable, "-m", "build"],
        [sys.executable, "scripts/check_distribution.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    if shutil.which("node") is not None:
        subprocess.run(["node", "--check", "src/geo_vision/dashboard.js"], cwd=ROOT, check=True)
    command = [sys.executable, "scripts/validate_release.py", "--skip-tests"]
    bundle_path = ROOT / "models/landcover/bundle.json"
    if not bundle_path.is_file():
        command.append("--allow-untrained")
    else:
        gate_path = ROOT / "reports/landcover/release-gate.json"
        if gate_path.is_file():
            try:
                gate = json.loads(gate_path.read_text())
            except (OSError, ValueError):
                gate = {}
            if gate.get("eligible_for_github_showcase") is False:
                command.append("--allow-nonpublishable")
    subprocess.run(command, cwd=ROOT, check=True)


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.create_system = 0
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "geospatial-vision-observatory-windows-github-ready.zip",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="package without rerunning tests/coverage/release validation",
    )
    args = parser.parse_args()
    if not args.skip_preflight:
        _run_preflight()

    files = _source_files()
    if not files:
        raise SystemExit("no release files found")
    manifest_lines: list[str] = []
    payloads: list[tuple[str, bytes, int]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        manifest_lines.append(f"{_sha256_bytes(payload)}  {relative}")
        mode = 0o755 if os.access(path, os.X_OK) else 0o644
        payloads.append((relative, payload, mode))
    manifest = ("\n".join(manifest_lines) + "\n").encode()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for relative, payload, mode in payloads:
                zf.writestr(_zip_info(f"{ARCHIVE_ROOT}/{relative}", mode), payload)
            zf.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.sha256", 0o644),
                manifest,
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    archive_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{archive_sha}  {output.name}\n")
    print(output)
    print(sidecar)
    print(f"files={len(files) + 1} sha256={archive_sha}")


if __name__ == "__main__":
    main()
