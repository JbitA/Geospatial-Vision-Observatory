#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from geo_vision.aoi import load_aoi_document  # noqa: E402
from geo_vision.planning import build_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AOI v2 GeoJSON and emit a deterministic plan")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/aoi-plan.json"))
    parser.add_argument("--recipe", default="landcover-v1")
    parser.add_argument("--resolution-m", type=float, default=10.0)
    parser.add_argument("--cell-pixels", type=int, default=1024)
    parser.add_argument("--halo-pixels", type=int, default=32)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.is_symlink():
        parser.error("input must be a regular GeoJSON file")
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AOI document must be a JSON object")
        aois = load_aoi_document(payload)
        plan = build_plan(
            aois,
            recipe=args.recipe,
            resolution_m=args.resolution_m,
            cell_pixels=args.cell_pixels,
            halo_pixels=args.halo_pixels,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"AOI planning failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
