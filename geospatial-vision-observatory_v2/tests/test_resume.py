from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from geo_vision.aoi import aoi_from_bbox
from geo_vision.resume import (
    CELL_DATASET_LAYOUT,
    CELL_MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
    canonical_sha256,
    cell_recipe_identity,
    prior_source_selection_id,
    request_identity,
    resolve_run_window,
    run_identity,
    validate_processing_cell_manifest,
)
from scripts import prepare_geospatial_dataset as prep

ROOT = Path(__file__).parents[1]


def _fake_manifest_builder(calls: list[dict[str, object]], *, fail_call: int | None = None):
    def build(aoi, grid, cell, output_root, *_args, **kwargs):
        calls.append(
            {
                "cell_id": cell.cell_id,
                "start": kwargs["start"],
                "end": kwargs["end"],
                "run_id": kwargs["run_id"],
                "recipe_id": kwargs["recipe_id"],
                "expected_source_selection_id": kwargs["expected_source_selection_id"],
            }
        )
        if fail_call is not None and len(calls) == fail_call:
            raise RuntimeError("simulated interruption")
        cells_root = kwargs["cells_root"]
        destination = cells_root / cell.cell_id
        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / "fixture.bin"
        artifact.write_bytes(f"artifact:{cell.cell_id}".encode())
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        source_id = "b" * 64
        prior = kwargs["expected_source_selection_id"]
        if prior is not None:
            assert prior == source_id
        manifest = {
            "schema_version": CELL_MANIFEST_SCHEMA_VERSION,
            "dataset_layout": CELL_DATASET_LAYOUT,
            "run_id": kwargs["run_id"],
            "recipe_id": kwargs["recipe_id"],
            "execution_mode": "collection_resumable",
            "source_selection_id": source_id,
            "cell": {"cell_id": cell.cell_id},
            "integrity": {
                "outputs": {
                    artifact.name: {"sha256": digest, "bytes": artifact.stat().st_size}
                }
            },
        }
        path = destination / "manifest.json"
        atomic_write_json(path, manifest)
        return path

    return build


def _aoi():
    return aoi_from_bbox("resume-aoi", (24.80, 60.10, 25.05, 60.22))


def _collection(tmp_path: Path, **kwargs) -> Path:
    return prep.build_processing_cell_collection(
        _aoi(),
        tmp_path / "out",
        30,
        20.0,
        resolution_m=100.0,
        cell_pixels=128,
        halo_pixels=8,
        **kwargs,
    )


def test_resolve_run_window_freezes_dynamic_time() -> None:
    now = datetime(2026, 8, 24, 17, 30, tzinfo=UTC)
    start, end = resolve_run_window(30, start=None, end=None, now=now)
    assert end == now
    assert start == now - timedelta(days=30)
    with pytest.raises(ValueError, match="both"):
        resolve_run_window(30, start=now, end=None)


def test_request_and_run_identity_are_deterministic() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    request_id, payload = request_identity(
        aoi_id="a",
        geometry_id="1" * 64,
        grid_signature="2" * 64,
        lookback_days=30,
        max_cloud=20.0,
        require_cloud_threshold=True,
        explicit_start=start,
        explicit_end=end,
        pipeline_version="2.0.0",
    )
    assert request_id == canonical_sha256(payload)
    assert run_identity(request_id, resolved_start=start, resolved_end=end) == run_identity(
        request_id, resolved_start=start, resolved_end=end
    )
    assert cell_recipe_identity("a" * 64, "b" * 64) != cell_recipe_identity(
        "a" * 64, "c" * 64
    )


def test_processing_cell_manifest_validator_detects_corruption_and_extras(tmp_path: Path) -> None:
    cell_id = "c" * 64
    recipe_id = "d" * 64
    run_id = "e" * 64
    root = tmp_path / cell_id
    root.mkdir()
    artifact = root / "data.bin"
    artifact.write_bytes(b"abc")
    manifest = {
        "schema_version": CELL_MANIFEST_SCHEMA_VERSION,
        "dataset_layout": CELL_DATASET_LAYOUT,
        "run_id": run_id,
        "recipe_id": recipe_id,
        "source_selection_id": "f" * 64,
        "cell": {"cell_id": cell_id},
        "integrity": {
            "outputs": {
                "data.bin": {
                    "bytes": 3,
                    "sha256": hashlib.sha256(b"abc").hexdigest(),
                }
            }
        },
    }
    manifest_path = root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    assert validate_processing_cell_manifest(
        manifest_path,
        expected_cell_id=cell_id,
        expected_recipe_id=recipe_id,
        expected_run_id=run_id,
    ) is not None
    assert prior_source_selection_id(
        manifest_path, expected_cell_id=cell_id, expected_recipe_id=recipe_id
    ) == "f" * 64

    artifact.write_bytes(b"abd")
    assert validate_processing_cell_manifest(
        manifest_path,
        expected_cell_id=cell_id,
        expected_recipe_id=recipe_id,
        expected_run_id=run_id,
    ) is None
    artifact.write_bytes(b"abc")
    (root / "stale.bin").write_bytes(b"stale")
    assert validate_processing_cell_manifest(
        manifest_path,
        expected_cell_id=cell_id,
        expected_recipe_id=recipe_id,
        expected_run_id=run_id,
    ) is None


def test_collection_rerun_reuses_valid_cells_and_same_frozen_window(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    resolutions = 0
    fixed_start = datetime(2026, 1, 1, tzinfo=UTC)
    fixed_end = datetime(2026, 1, 31, tzinfo=UTC)

    def frozen(*_args, **_kwargs):
        nonlocal resolutions
        resolutions += 1
        return fixed_start, fixed_end

    monkeypatch.setattr(prep, "resolve_run_window", frozen)
    monkeypatch.setattr(prep, "build_processing_cell_dataset", _fake_manifest_builder(calls))
    first = _collection(tmp_path)
    first_payload = json.loads(first.read_text())
    first_call_count = len(calls)
    assert first_call_count > 1
    assert resolutions == 1
    assert first_payload["execution"]["built_cells"] == first_call_count

    second = _collection(tmp_path)
    second_payload = json.loads(second.read_text())
    assert second == first
    assert len(calls) == first_call_count
    assert resolutions == 1
    assert second_payload["execution"]["built_cells"] == 0
    assert second_payload["execution"]["reused_cells"] == first_call_count
    assert second_payload["resolved_window"] == first_payload["resolved_window"]

    state = json.loads((tmp_path / "out" / "resume-aoi" / "run-state.json").read_text())
    assert state["status"] == "complete"
    assert state["run_id"] == first_payload["run_id"]


def test_corrupt_cell_is_rebuilt_while_other_cells_are_reused(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fixed = (
        datetime(2026, 2, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(prep, "resolve_run_window", lambda *_args, **_kwargs: fixed)
    monkeypatch.setattr(prep, "build_processing_cell_dataset", _fake_manifest_builder(calls))
    first = _collection(tmp_path)
    first_payload = json.loads(first.read_text())
    initial = len(calls)

    victim_manifest = tmp_path / "out" / "resume-aoi" / first_payload["cells"][0]["manifest"]
    (victim_manifest.parent / "fixture.bin").write_bytes(b"corrupt")
    second = _collection(tmp_path)
    second_payload = json.loads(second.read_text())
    assert len(calls) == initial + 1
    assert calls[-1]["expected_source_selection_id"] == "b" * 64
    assert second_payload["execution"]["built_cells"] == 1
    assert second_payload["execution"]["reused_cells"] == initial - 1


def test_interrupted_collection_resumes_only_incomplete_cells(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fixed = (
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 4, 1, tzinfo=UTC),
    )
    resolutions = 0

    def frozen(*_args, **_kwargs):
        nonlocal resolutions
        resolutions += 1
        return fixed

    monkeypatch.setattr(prep, "resolve_run_window", frozen)
    monkeypatch.setattr(
        prep, "build_processing_cell_dataset", _fake_manifest_builder(calls, fail_call=2)
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _collection(tmp_path)
    state_path = tmp_path / "out" / "resume-aoi" / "run-state.json"
    interrupted = json.loads(state_path.read_text())
    assert interrupted["status"] == "running"
    assert interrupted["completed_cells"] == 1
    assert interrupted["last_error"]["type"] == "RuntimeError"
    assert resolutions == 1

    resumed_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        prep, "build_processing_cell_dataset", _fake_manifest_builder(resumed_calls)
    )
    result = _collection(tmp_path)
    completed = json.loads(result.read_text())
    assert resolutions == 1
    assert completed["execution"]["reused_cells"] == 1
    assert completed["execution"]["built_cells"] == len(completed["cells"]) - 1
    assert json.loads(state_path.read_text())["status"] == "complete"


def test_fresh_window_creates_new_run_without_deleting_old_run(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    windows = iter(
        [
            (datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)),
            (datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC)),
        ]
    )
    monkeypatch.setattr(prep, "resolve_run_window", lambda *_args, **_kwargs: next(windows))
    monkeypatch.setattr(prep, "build_processing_cell_dataset", _fake_manifest_builder(calls))
    first = _collection(tmp_path)
    first_payload = json.loads(first.read_text())
    second = _collection(tmp_path, fresh_window=True)
    second_payload = json.loads(second.read_text())
    assert first_payload["request_id"] == second_payload["request_id"]
    assert first_payload["run_id"] != second_payload["run_id"]
    aoi_root = tmp_path / "out" / "resume-aoi"
    assert (aoi_root / "runs" / first_payload["run_id"] / "cells-manifest.json").is_file()
    assert (aoi_root / "runs" / second_payload["run_id"] / "cells-manifest.json").is_file()


def test_iteration_3_schemas_are_valid() -> None:
    for name in (
        "processing-cell.schema.json",
        "processing-cell-collection.schema.json",
        "processing-run-state.schema.json",
    ):
        schema = json.loads((ROOT / "schemas" / name).read_text())
        Draft202012Validator.check_schema(schema)
