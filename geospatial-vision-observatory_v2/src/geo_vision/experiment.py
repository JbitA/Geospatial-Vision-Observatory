from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyproj import CRS, Geod, Transformer
from shapely.geometry import shape
from shapely.ops import nearest_points, transform

from .aoi import AOI, SCIENTIFIC_EVIDENCE_ROLES, ScientificRole

LEDGER_SCHEMA_VERSION = 1
_HEX = frozenset("0123456789abcdef")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def verify_holdout_exposure_ledger(path: Path) -> list[dict[str, Any]]:
    """Verify and return the complete tamper-evident holdout exposure chain.

    The ledger is append-only. A malformed line, broken hash chain, duplicate exposure identity, or
    invalid experiment/AOI identity fails closed so later experiments cannot silently ignore an
    ambiguous holdout history.
    """

    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise ValueError("holdout exposure ledger must be a regular non-symlink file")

    events: list[dict[str, Any]] = []
    previous: str | None = None
    exposures: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("holdout exposure ledger is unreadable") from error
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"holdout exposure ledger contains an empty line at {index}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"holdout exposure ledger line {index} is invalid JSON") from error
        if not isinstance(raw, dict):
            raise ValueError(f"holdout exposure ledger line {index} must be an object")
        event = dict(raw)
        event_hash = event.pop("event_sha256", None)
        if not _valid_sha256(event_hash):
            raise ValueError(f"holdout exposure ledger line {index} has an invalid event hash")
        if _canonical_sha256(event) != event_hash:
            raise ValueError(f"holdout exposure ledger line {index} failed SHA-256 verification")
        if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ValueError(f"holdout exposure ledger line {index} has an unsupported schema")
        if event.get("sequence") != index:
            raise ValueError(f"holdout exposure ledger line {index} has a broken sequence")
        if event.get("previous_event_sha256") != previous:
            raise ValueError(f"holdout exposure ledger line {index} has a broken hash chain")
        for name in ("experiment_signature", "dataset_signature", "exposure_id"):
            if not _valid_sha256(event.get(name)):
                raise ValueError(f"holdout exposure ledger line {index} has invalid {name}")
        aoi_ids = event.get("aoi_ids")
        geometry_ids = event.get("aoi_geometry_ids")
        if (
            not isinstance(aoi_ids, list)
            or not aoi_ids
            or not all(isinstance(value, str) and value for value in aoi_ids)
            or aoi_ids != sorted(set(aoi_ids))
        ):
            raise ValueError(f"holdout exposure ledger line {index} has invalid AOI ids")
        if (
            not isinstance(geometry_ids, list)
            or len(geometry_ids) != len(aoi_ids)
            or not all(_valid_sha256(value) for value in geometry_ids)
            or geometry_ids != sorted(set(geometry_ids))
        ):
            raise ValueError(f"holdout exposure ledger line {index} has invalid AOI geometry ids")
        exposure_id = str(event["exposure_id"])
        if exposure_id in exposures:
            raise ValueError("holdout exposure ledger contains a duplicate exposure identity")
        exposures.add(exposure_id)
        previous = str(event_hash)
        events.append({**event, "event_sha256": event_hash})
    return events


def holdout_exposure_id(
    *,
    experiment_signature: str,
    dataset_signature: str,
    aois: Iterable[AOI],
    purpose: str,
) -> str:
    ordered = sorted(aois, key=lambda aoi: (aoi.aoi_id, aoi.geometry_id))
    payload = {
        "experiment_signature": experiment_signature,
        "dataset_signature": dataset_signature,
        "aois": [
            {"aoi_id": aoi.aoi_id, "geometry_id": aoi.geometry_id, "role": aoi.role.value}
            for aoi in ordered
        ],
        "purpose": purpose,
    }
    return _canonical_sha256(payload)


def record_holdout_exposure(
    path: Path,
    *,
    experiment_signature: str,
    dataset_signature: str,
    aois: Iterable[AOI],
    purpose: str,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    """Append one idempotent, hash-chained external-pixel exposure event."""

    if not _valid_sha256(experiment_signature) or not _valid_sha256(dataset_signature):
        raise ValueError("experiment and dataset signatures must be lowercase SHA-256 values")
    ordered = sorted(aois, key=lambda aoi: (aoi.aoi_id, aoi.geometry_id))
    if not ordered:
        raise ValueError("at least one external AOI is required")
    if any(aoi.role not in {ScientificRole.EXTERNAL_UNOBSERVED, ScientificRole.EXTERNAL_OBSERVED} for aoi in ordered):
        raise ValueError("holdout exposure events may contain only external AOIs")
    if len({aoi.aoi_id for aoi in ordered}) != len(ordered):
        raise ValueError("holdout exposure event contains duplicate AOI ids")
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 1000:
        raise ValueError("holdout exposure purpose must be 1-1000 characters")

    existing = verify_holdout_exposure_ledger(path)
    exposure_id = holdout_exposure_id(
        experiment_signature=experiment_signature,
        dataset_signature=dataset_signature,
        aois=ordered,
        purpose=purpose.strip(),
    )
    for event in existing:
        if event["exposure_id"] == exposure_id:
            return event

    timestamp = recorded_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    core: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(existing) + 1,
        "previous_event_sha256": existing[-1]["event_sha256"] if existing else None,
        "recorded_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "event_type": "external_pixels_exposure",
        "experiment_signature": experiment_signature,
        "dataset_signature": dataset_signature,
        "exposure_id": exposure_id,
        "aoi_ids": sorted(aoi.aoi_id for aoi in ordered),
        "aoi_geometry_ids": sorted(aoi.geometry_id for aoi in ordered),
        "roles": sorted({aoi.role.value for aoi in ordered}),
        "purpose": purpose.strip(),
    }
    event = {**core, "event_sha256": _canonical_sha256(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    verified = verify_holdout_exposure_ledger(path)
    if not verified or verified[-1]["event_sha256"] != event["event_sha256"]:
        raise RuntimeError("holdout exposure ledger append did not verify")
    return verified[-1]


def exposed_geometry_ids(path: Path) -> set[str]:
    return {
        geometry_id
        for event in verify_holdout_exposure_ledger(path)
        for geometry_id in event["aoi_geometry_ids"]
    }


def _pair_distance_km(left: AOI, right: AOI) -> float:
    left_geom = shape(left.geometry)
    right_geom = shape(right.geometry)
    if left_geom.intersects(right_geom):
        return 0.0
    left_centroid = left_geom.centroid
    right_centroid = right_geom.centroid
    lon0 = (float(left_centroid.x) + float(right_centroid.x)) / 2.0
    lat0 = (float(left_centroid.y) + float(right_centroid.y)) / 2.0
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} +datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local, always_xy=True).transform
    inverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True).transform
    projected_left = transform(forward, left_geom)
    projected_right = transform(forward, right_geom)
    point_left, point_right = nearest_points(projected_left, projected_right)
    lon_left, lat_left = inverse(point_left.x, point_left.y)
    lon_right, lat_right = inverse(point_right.x, point_right.y)
    _az12, _az21, distance_m = Geod(ellps="WGS84").inv(
        lon_left, lat_left, lon_right, lat_right
    )
    return max(float(distance_m), 0.0) / 1000.0


def validate_external_candidate(
    candidate: AOI,
    known_aois: Iterable[AOI],
    *,
    minimum_separation_km: float,
    exposure_ledger: Path | None = None,
) -> dict[str, Any]:
    """Validate new untouched external geography against known scientific evidence.

    The minimum separation is caller-supplied because an appropriate geographic-independence
    threshold is study-specific. The project deliberately does not hide an arbitrary scientific
    threshold in code.
    """

    if candidate.role is not ScientificRole.EXTERNAL_UNOBSERVED:
        raise ValueError("candidate must have scientific role external_unobserved")
    if (
        isinstance(minimum_separation_km, bool)
        or not math.isfinite(float(minimum_separation_km))
        or float(minimum_separation_km) < 0.0
    ):
        raise ValueError("minimum_separation_km must be a finite non-negative number")
    minimum = float(minimum_separation_km)

    known = [aoi for aoi in known_aois if aoi.role in SCIENTIFIC_EVIDENCE_ROLES]
    if any(aoi.aoi_id == candidate.aoi_id for aoi in known):
        raise ValueError(f"candidate AOI id already exists: {candidate.aoi_id}")
    if any(aoi.geometry_id == candidate.geometry_id for aoi in known):
        raise ValueError("candidate geometry is already present in scientific evidence")
    if exposure_ledger is not None and candidate.geometry_id in exposed_geometry_ids(exposure_ledger):
        raise ValueError("candidate geometry already appears in the holdout exposure ledger")

    distances: list[dict[str, Any]] = []
    for aoi in sorted(known, key=lambda item: item.aoi_id):
        distance_km = _pair_distance_km(candidate, aoi)
        row = {
            "aoi_id": aoi.aoi_id,
            "role": aoi.role.value,
            "geometry_id": aoi.geometry_id,
            "distance_km": distance_km,
        }
        distances.append(row)
        if distance_km + 1e-9 < minimum:
            raise ValueError(
                f"candidate {candidate.aoi_id} is {distance_km:.3f} km from {aoi.aoi_id}; "
                f"minimum required separation is {minimum:.3f} km"
            )

    minimum_observed = min((row["distance_km"] for row in distances), default=None)
    return {
        "schema_version": 1,
        "candidate": {
            "aoi_id": candidate.aoi_id,
            "geometry_id": candidate.geometry_id,
            "role": candidate.role.value,
        },
        "minimum_required_separation_km": minimum,
        "minimum_observed_separation_km": minimum_observed,
        "comparisons": distances,
        "status": "passed",
    }
