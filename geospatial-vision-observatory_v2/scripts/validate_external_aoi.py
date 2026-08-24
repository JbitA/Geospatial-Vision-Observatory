#!/usr/bin/env python3
"""Validate proposed untouched external geography against known scientific evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geo_vision.aoi import load_aoi_document
from geo_vision.experiment import validate_external_candidate


def _load(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read AOI document: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"AOI document must be a JSON object: {path}")
    return load_aoi_document(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one external_unobserved candidate against existing scientific AOIs. "
            "The minimum separation is explicit because independence distance is study-specific."
        )
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--known-aois", type=Path, default=Path("config/aois-baseline.geojson")
    )
    parser.add_argument("--minimum-separation-km", type=float, required=True)
    parser.add_argument(
        "--exposure-ledger",
        type=Path,
        default=Path("reports/experiment-state/holdout-exposure.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates = _load(args.candidate)
    if len(candidates) != 1:
        parser.error("candidate document must contain exactly one AOI")
    known = _load(args.known_aois)
    result = validate_external_candidate(
        candidates[0],
        known,
        minimum_separation_km=args.minimum_separation_km,
        exposure_ledger=args.exposure_ledger if args.exposure_ledger.exists() else None,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
