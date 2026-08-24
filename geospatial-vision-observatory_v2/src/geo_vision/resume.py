from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

RUN_STATE_SCHEMA_VERSION = 1
CELL_RECIPE_VERSION = "processing-cell-recipe-v2"
CELL_MANIFEST_SCHEMA_VERSION = 6
CELL_DATASET_LAYOUT = "processing_cell_v2"
COLLECTION_SCHEMA_VERSION = 2
COLLECTION_DATASET_LAYOUT = "processing_cell_collection_v2"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def resolve_run_window(
    lookback_days: int,
    *,
    start: datetime | None,
    end: datetime | None,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be an integer between 1 and 3650")
    if (start is None) != (end is None):
        raise ValueError("start and end must either both be supplied or both be omitted")
    if start is None:
        resolved_end = now or datetime.now(UTC)
        if resolved_end.tzinfo is None or resolved_end.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        resolved_end = resolved_end.astimezone(UTC)
        resolved_start = resolved_end - timedelta(days=lookback_days)
    else:
        assert end is not None
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")
        resolved_start = start.astimezone(UTC)
        resolved_end = end.astimezone(UTC)
    if resolved_start >= resolved_end:
        raise ValueError("run window start must be before end")
    return resolved_start, resolved_end


def request_identity(
    *,
    aoi_id: str,
    geometry_id: str,
    grid_signature: str,
    lookback_days: int,
    max_cloud: float,
    require_cloud_threshold: bool,
    explicit_start: datetime | None,
    explicit_end: datetime | None,
    pipeline_version: str,
) -> tuple[str, dict[str, Any]]:
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    if (explicit_start is None) != (explicit_end is None):
        raise ValueError("explicit_start and explicit_end must be supplied together")
    window_request: dict[str, Any]
    if explicit_start is None:
        window_request = {"mode": "lookback", "lookback_days": lookback_days}
    else:
        assert explicit_end is not None
        window_request = {
            "mode": "explicit",
            "start": utc_iso(explicit_start),
            "end": utc_iso(explicit_end),
        }
    payload = {
        "recipe_version": CELL_RECIPE_VERSION,
        "pipeline_version": pipeline_version,
        "aoi_id": aoi_id,
        "geometry_id": geometry_id,
        "grid_signature": grid_signature,
        "window_request": window_request,
        "max_cloud": float(max_cloud),
        "require_cloud_threshold": bool(require_cloud_threshold),
    }
    return canonical_sha256(payload), payload


def run_identity(
    request_id: str,
    *,
    resolved_start: datetime,
    resolved_end: datetime,
) -> str:
    return canonical_sha256(
        {
            "request_id": request_id,
            "resolved_start": utc_iso(resolved_start),
            "resolved_end": utc_iso(resolved_end),
        }
    )


def cell_recipe_identity(run_id: str, cell_id: str) -> str:
    return canonical_sha256(
        {
            "recipe_version": CELL_RECIPE_VERSION,
            "manifest_schema_version": CELL_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "cell_id": cell_id,
        }
    )


def source_selection_identity(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def prior_source_selection_id(
    manifest_path: Path,
    *,
    expected_cell_id: str,
    expected_recipe_id: str,
) -> str | None:
    payload = load_json_object(manifest_path)
    if payload is None:
        return None
    cell = payload.get("cell")
    if not isinstance(cell, dict) or cell.get("cell_id") != expected_cell_id:
        return None
    if payload.get("recipe_id") != expected_recipe_id:
        return None
    value = payload.get("source_selection_id")
    return value if isinstance(value, str) and len(value) == 64 else None


def validate_processing_cell_manifest(
    manifest_path: Path,
    *,
    expected_cell_id: str,
    expected_recipe_id: str,
    expected_run_id: str,
) -> dict[str, Any] | None:
    payload = load_json_object(manifest_path)
    if payload is None:
        return None
    if payload.get("schema_version") != CELL_MANIFEST_SCHEMA_VERSION:
        return None
    if payload.get("dataset_layout") != CELL_DATASET_LAYOUT:
        return None
    if payload.get("run_id") != expected_run_id or payload.get("recipe_id") != expected_recipe_id:
        return None
    cell = payload.get("cell")
    if not isinstance(cell, dict) or cell.get("cell_id") != expected_cell_id:
        return None
    source_id = payload.get("source_selection_id")
    if not isinstance(source_id, str) or len(source_id) != 64:
        return None
    integrity = payload.get("integrity")
    outputs = integrity.get("outputs") if isinstance(integrity, dict) else None
    if not isinstance(outputs, dict) or not outputs:
        return None

    root = manifest_path.parent
    expected_files: set[str] = set()
    for name, metadata in outputs.items():
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
            return None
        if not isinstance(metadata, dict):
            return None
        expected_size = metadata.get("bytes")
        expected_hash = metadata.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            return None
        artifact = root / name
        if not artifact.is_file() or artifact.is_symlink():
            return None
        try:
            if artifact.stat().st_size != expected_size or file_sha256(artifact) != expected_hash:
                return None
        except OSError:
            return None
        expected_files.add(name)

    try:
        actual_files = {
            path.name
            for path in root.iterdir()
            if path.name != manifest_path.name and path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            return None
        if any(path.is_symlink() for path in root.iterdir()):
            return None
    except OSError:
        return None
    return payload
