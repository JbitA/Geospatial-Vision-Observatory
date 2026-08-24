#!/usr/bin/env python3
"""Execute AOI v2 Iteration 2 as deterministic projected processing cells."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from geo_vision.aoi import load_aoi_document  # noqa: E402
from scripts.prepare_geospatial_dataset import build_processing_cell_collection  # noqa: E402


def _utc_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _utc_end(value: date) -> datetime:
    return datetime.combine(value, time.max, tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic AOI v2 processing-cell artifacts with multi-source mosaics"
    )
    parser.add_argument("input", type=Path, help="GeoJSON Feature or FeatureCollection")
    parser.add_argument("--aoi-id", help="execute only one AOI id from a FeatureCollection")
    parser.add_argument("--output", type=Path, default=Path("data/processing-cells"))
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--max-cloud", type=float, default=15.0)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--require-cloud-threshold", action="store_true")
    parser.add_argument("--resolution-m", type=float, default=10.0)
    parser.add_argument("--cell-pixels", type=int, default=1024)
    parser.add_argument("--halo-pixels", type=int, default=32)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="rebuild cells even when their recorded artifacts pass integrity checks",
    )
    parser.add_argument(
        "--fresh-window",
        action="store_true",
        help="for dynamic lookback runs, start a new frozen acquisition window instead of resuming",
    )
    args = parser.parse_args()

    if not args.input.is_file() or args.input.is_symlink():
        parser.error("input must be a regular GeoJSON file")
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")
    if args.start_date is not None and args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AOI document must be a JSON object")
        aois = load_aoi_document(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"AOI loading failed: {error}") from error

    if args.aoi_id is not None:
        aois = [aoi for aoi in aois if aoi.aoi_id == args.aoi_id]
        if not aois:
            raise SystemExit(f"AOI id not found: {args.aoi_id}")

    start = _utc_start(args.start_date) if args.start_date is not None else None
    end = _utc_end(args.end_date) if args.end_date is not None else None
    for aoi in aois:
        result = build_processing_cell_collection(
            aoi,
            args.output,
            args.lookback_days,
            args.max_cloud,
            resolution_m=args.resolution_m,
            cell_pixels=args.cell_pixels,
            halo_pixels=args.halo_pixels,
            start=start,
            end=end,
            require_cloud_threshold=args.require_cloud_threshold,
            force_rebuild=args.force_rebuild,
            fresh_window=args.fresh_window,
        )
        print(result)


if __name__ == "__main__":
    main()
