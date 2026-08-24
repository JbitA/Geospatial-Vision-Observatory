#!/usr/bin/env python3
"""Prepare the spatially isolated real-data split used by the showcase training pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from geo_vision.ml.data import dataset_signature
from geo_vision.ml.schema import SHOWCASE_SPLIT

EXPECTED_COLLECTION = "sentinel-2-c1-l2a"
REQUIRED_OUTPUTS = frozenset({"sentinel2_multispectral.tif", "worldcover_2021_on_sentinel.tif"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_metadata(directory: Path, name: object, metadata: object) -> bool:
    if not isinstance(name, str) or not name or Path(name).name != name:
        return False
    if not isinstance(metadata, dict):
        return False
    digest = metadata.get("sha256")
    expected_bytes = metadata.get("bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        return False
    path = directory / name
    if not path.is_file() or path.is_symlink():
        return False
    try:
        return path.stat().st_size == expected_bytes and sha256_file(path) == digest
    except OSError:
        return False


def scene_complete(
    root: Path,
    key: str,
    *,
    start_date: date,
    end_date: date,
    max_cloud: float,
    max_pixels: int,
) -> bool:
    """Return True only when a cached AOI exactly matches the requested curation policy."""

    directory = root / key
    manifest_path = directory / "manifest.json"
    try:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return False
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {3, 4}:
        return False
    aoi = manifest.get("aoi")
    sentinel = manifest.get("sentinel2")
    integrity = manifest.get("integrity")
    if not isinstance(aoi, dict) or aoi.get("key") != key:
        return False
    if not isinstance(sentinel, dict) or not isinstance(integrity, dict):
        return False
    if sentinel.get("collection") != EXPECTED_COLLECTION:
        return False
    requested_start = sentinel.get("requested_start")
    requested_end = sentinel.get("requested_end")
    if not isinstance(requested_start, str) or not requested_start.startswith(start_date.isoformat()):
        return False
    if not isinstance(requested_end, str) or not requested_end.startswith(end_date.isoformat()):
        return False
    if sentinel.get("cloud_threshold_required") is not True:
        return False
    try:
        cloud = float(sentinel.get("cloud_cover"))
        aoi_obscured = float(sentinel.get("aoi_scl_obscured_percent"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(cloud) or not 0.0 <= cloud <= 100.0:
        return False
    if sentinel.get("cloud_quality_basis") != "aoi_scl_obscured_percent":
        return False
    if not math.isfinite(aoi_obscured) or not 0.0 <= aoi_obscured <= max_cloud:
        return False
    if integrity.get("max_pixels") != max_pixels:
        return False
    outputs = integrity.get("outputs")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUTS.issubset(outputs):
        return False
    return all(_safe_output_metadata(directory, name, metadata) for name, metadata in outputs.items())


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.start_date >= args.end_date:
        parser.error("--start-date must be before --end-date")
    if not math.isfinite(args.max_cloud) or not 0.0 <= args.max_cloud <= 100.0:
        parser.error("--max-cloud must be a finite percentage between 0 and 100")
    if not 256 <= args.max_pixels <= 2048:
        parser.error("--max-pixels must be between 256 and 2048")



def authorize_external_curation(
    root: Path, evidence_path: Path = Path("reports/landcover/seed-selection.json")
) -> None:
    """Require immutable validation-only seed-selection evidence before holdout acquisition."""

    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise RuntimeError(
            "external holdout curation is locked until validation-only seed selection completes"
        )
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("seed-selection evidence is unreadable") from error
    if not isinstance(evidence, dict):
        raise RuntimeError("seed-selection evidence has an invalid structure")
    if evidence.get("selection_dataset") != "validation AOIs only":
        raise RuntimeError("seed-selection evidence does not identify validation-only selection")
    if evidence.get("external_test_used_for_selection") is not False:
        raise RuntimeError("seed-selection evidence does not prove holdout isolation")
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 3:
        raise RuntimeError("external holdout curation requires at least three selection candidates")
    signatures: set[str] = set()
    seeds: set[int] = set()
    for row in candidates:
        if not isinstance(row, dict):
            raise RuntimeError("seed-selection candidate is invalid")
        seed = row.get("seed")
        signature = row.get("dataset_signature")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in seeds:
            raise RuntimeError("seed-selection candidates require unique integer seeds")
        if not isinstance(signature, str) or len(signature) != 64:
            raise RuntimeError("seed-selection candidate dataset signature is invalid")
        seeds.add(seed)
        signatures.add(signature)
    selected = evidence.get("selected_seed")
    if selected not in seeds:
        raise RuntimeError("selected seed is not one of the recorded candidates")
    current = dataset_signature(
        [root / key for key in (*SHOWCASE_SPLIT.train, *SHOWCASE_SPLIT.validation)]
    )
    if signatures != {current}:
        raise RuntimeError(
            "train/validation data changed after seed selection; external holdout remains locked"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/curated"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2021, 5, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2021, 9, 30))
    parser.add_argument("--max-cloud", type=float, default=15.0)
    parser.add_argument("--max-pixels", type=int, default=768)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="prepare only train/validation AOIs; external holdout remains absent",
    )
    args = parser.parse_args()
    _validate_args(parser, args)
    assignments = {
        **{key: "train" for key in SHOWCASE_SPLIT.train},
        **{key: "validation" for key in SHOWCASE_SPLIT.validation},
    }
    if not args.selection_only:
        authorize_external_curation(args.output)
        assignments.update({key: "external_test" for key in SHOWCASE_SPLIT.external_test})
    prepared: list[dict[str, str]] = []
    for key, split in assignments.items():
        complete = scene_complete(
            args.output,
            key,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud=args.max_cloud,
            max_pixels=args.max_pixels,
        )
        if complete and not args.force:
            status = "reused"
        else:
            command = [
                sys.executable,
                "scripts/prepare_geospatial_dataset.py",
                "--aoi",
                key,
                "--output",
                str(args.output),
                "--start-date",
                args.start_date.isoformat(),
                "--end-date",
                args.end_date.isoformat(),
                "--max-cloud",
                str(args.max_cloud),
                "--require-cloud-threshold",
                "--max-pixels",
                str(args.max_pixels),
            ]
            subprocess.run(command, check=True)
            status = "prepared"
        manifest_path = args.output / key / "manifest.json"
        prepared.append(
            {
                "aoi": key,
                "split": split,
                "status": status,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "temporal_window": [args.start_date.isoformat(), args.end_date.isoformat()],
        "max_cloud_percent": args.max_cloud,
        "cloud_quality_basis": "aoi_scl_obscured_percent",
        "max_pixels": args.max_pixels,
        "assignment": prepared,
        "leakage_control": (
            "AOIs are assigned wholly to train, validation, or external_test; no crop from one "
            "source observation may cross a split boundary."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("selection-split.json" if args.selection_only else "training-split.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
