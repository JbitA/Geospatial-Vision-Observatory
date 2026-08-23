from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 1_000_000
MAX_MODEL_BYTES = 250_000_000
MAX_SUPPORT_BYTES = 5_000_000
REQUIRED_SUPPORT_FILES = frozenset(
    {"normalization.json", "classes.json", "training-config.json", "MODEL_CARD.md"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_bundle_name(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("model bundle contains an unsafe file name")
    if value in {".", ".."} or "\\" in value or "/" in value:
        raise ValueError("model bundle contains an unsafe file name")
    return value


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify the complete local deployment bundle before any model deserialization.

    The manifest itself is intentionally outside its own hash set. Every referenced support/model
    file must be a regular non-symlink file whose byte count and SHA-256 match exactly.
    """

    manifest_path = bundle_dir / "bundle.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise ValueError("model bundle manifest is missing or exceeds the safety limit")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model bundle manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("unsupported model bundle schema")
    if payload.get("task") != "worldcover_landcover_segmentation":
        raise ValueError("unsupported model bundle task")
    for signature_name in ("dataset_signature", "experiment_signature"):
        signature = payload.get(signature_name)
        if not isinstance(signature, str) or not SHA256_PATTERN.fullmatch(signature):
            raise ValueError(f"model bundle {signature_name} is invalid")

    model_file = _safe_bundle_name(payload.get("model_file"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("model bundle file manifest is invalid")
    expected_names = set(REQUIRED_SUPPORT_FILES) | {model_file}
    if set(files) != expected_names:
        raise ValueError("model bundle file set is incomplete or contains unexpected files")
    actual_names = {path.name for path in bundle_dir.iterdir()}
    if actual_names != expected_names | {"bundle.json"}:
        raise ValueError("model bundle directory contains unexpected files")

    for raw_name, raw_metadata in files.items():
        name = _safe_bundle_name(raw_name)
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"model bundle metadata is invalid: {name}")
        expected_sha = raw_metadata.get("sha256")
        expected_bytes = raw_metadata.get("bytes")
        if not isinstance(expected_sha, str) or not SHA256_PATTERN.fullmatch(expected_sha):
            raise ValueError(f"model bundle SHA-256 is invalid: {name}")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise ValueError(f"model bundle byte count is invalid: {name}")
        limit = MAX_MODEL_BYTES if name == model_file else MAX_SUPPORT_BYTES
        if expected_bytes > limit:
            raise ValueError(f"model bundle file exceeds safety limit: {name}")
        path = bundle_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"model bundle file is missing or not a regular file: {name}")
        stat = path.stat()
        if stat.st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise ValueError(f"model bundle integrity failure: {name}")

    return payload
