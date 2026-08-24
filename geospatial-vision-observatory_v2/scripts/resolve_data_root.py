#!/usr/bin/env python3
"""Select a writable/readable curated-data root for resilient Windows execution."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

AOIS = (
    "helsinki_metro",
    "north_karelia_forest",
    "turku_coast",
    "oulu_mixed",
    "tampere_growth",
    "jyvaskyla_validation",
    "stockholm_external",
    "tallinn_external",
)
KNOWN_SCENE_FILES = (
    "manifest.json",
    "sentinel2_multispectral.tif",
    "worldcover_2021_on_sentinel.tif",
)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str = ""


def _probe_existing_path(path: Path) -> None:
    """Touch metadata/content lightly so Windows ACL errors surface before the pipeline starts."""

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise OSError(f"symlinked cache entry is not accepted: {path}")
    if stat.S_ISDIR(info.st_mode):
        with os.scandir(path) as entries:
            for entry in entries:
                entry.stat(follow_symlinks=False)
        return
    if stat.S_ISREG(info.st_mode):
        with path.open("rb") as stream:
            stream.read(1)
        return
    raise OSError(f"unsupported cache entry type: {path}")


def probe_root(root: Path) -> ProbeResult:
    """Return whether the root is safe enough to reuse for this run."""

    try:
        root.mkdir(parents=True, exist_ok=True)
        _probe_existing_path(root)

        probe = root / f".gvo-write-probe-{uuid.uuid4().hex}.tmp"
        try:
            probe.write_bytes(b"gvo")
            if probe.read_bytes() != b"gvo":
                return ProbeResult(False, "write/read probe returned unexpected content")
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

        for name in ("selection-split.json", "training-split.json"):
            _probe_existing_path(root / name)
        for aoi in AOIS:
            scene = root / aoi
            _probe_existing_path(scene)
            for name in KNOWN_SCENE_FILES:
                _probe_existing_path(scene / name)
    except (OSError, PermissionError) as error:
        return ProbeResult(False, f"{type(error).__name__}: {error}")
    return ProbeResult(True)


def select_root(primary: Path, state_root: Path, *, max_fallbacks: int = 20) -> Path:
    primary = primary.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    first = probe_root(primary)
    if first.ok:
        return primary

    print(
        f"WARNING: curated cache is not safely accessible and will not be reused: {primary}\n"
        f"         {first.reason}",
        file=sys.stderr,
    )

    candidates = [state_root / "curated-windows"]
    candidates.extend(state_root / f"curated-windows-{index}" for index in range(2, max_fallbacks + 1))
    for candidate in candidates:
        result = probe_root(candidate)
        if result.ok:
            print(f"INFO: using healthy fallback curated cache: {candidate}", file=sys.stderr)
            return candidate.resolve()
        print(
            f"WARNING: fallback curated cache is unusable: {candidate} ({result.reason})",
            file=sys.stderr,
        )
    raise RuntimeError("no writable/readable curated cache directory could be selected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args()
    print(select_root(args.primary, args.state_root))


if __name__ == "__main__":
    main()
