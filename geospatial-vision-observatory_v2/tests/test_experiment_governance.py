from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from geo_vision.aoi import ScientificRole, aoi_from_bbox
from geo_vision.experiment import (
    exposed_geometry_ids,
    record_holdout_exposure,
    validate_external_candidate,
    verify_holdout_exposure_ledger,
)
from geo_vision.ml.metrics import spatial_block_bootstrap_macro_iou_ci


EXPERIMENT = "a" * 64
DATASET = "b" * 64


def _external(aoi_id: str, bbox: tuple[float, float, float, float], *, observed: bool = True):
    role = ScientificRole.EXTERNAL_OBSERVED if observed else ScientificRole.EXTERNAL_UNOBSERVED
    return aoi_from_bbox(aoi_id, bbox, role=role)


def test_holdout_exposure_ledger_is_idempotent_and_hash_chained(tmp_path) -> None:
    path = tmp_path / "holdout.jsonl"
    first = _external("external-a", (10.0, 60.0, 10.1, 60.1))
    second = _external("external-b", (11.0, 60.0, 11.1, 60.1))
    timestamp = datetime(2026, 8, 24, 18, 0, tzinfo=UTC)

    event1 = record_holdout_exposure(
        path,
        experiment_signature=EXPERIMENT,
        dataset_signature=DATASET,
        aois=[first],
        purpose="final external evaluation",
        recorded_at=timestamp,
    )
    repeated = record_holdout_exposure(
        path,
        experiment_signature=EXPERIMENT,
        dataset_signature=DATASET,
        aois=[first],
        purpose="final external evaluation",
        recorded_at=timestamp,
    )
    event2 = record_holdout_exposure(
        path,
        experiment_signature="c" * 64,
        dataset_signature=DATASET,
        aois=[second],
        purpose="independent follow-up evaluation",
        recorded_at=timestamp,
    )

    assert repeated == event1
    events = verify_holdout_exposure_ledger(path)
    assert len(events) == 2
    assert event2["previous_event_sha256"] == event1["event_sha256"]
    assert exposed_geometry_ids(path) == {first.geometry_id, second.geometry_id}


def test_holdout_exposure_ledger_fails_closed_after_tampering(tmp_path) -> None:
    path = tmp_path / "holdout.jsonl"
    external = _external("external-a", (10.0, 60.0, 10.1, 60.1))
    record_holdout_exposure(
        path,
        experiment_signature=EXPERIMENT,
        dataset_signature=DATASET,
        aois=[external],
        purpose="final external evaluation",
    )
    payload = json.loads(path.read_text())
    payload["purpose"] = "rewritten after inspection"
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="SHA-256 verification"):
        verify_holdout_exposure_ledger(path)


def test_external_candidate_checks_scientific_distance_and_prior_exposure(tmp_path) -> None:
    known = aoi_from_bbox(
        "training-a",
        (24.90, 60.15, 25.00, 60.25),
        role=ScientificRole.TRAIN,
    )
    far = _external("candidate-far", (27.0, 62.0, 27.1, 62.1), observed=False)
    result = validate_external_candidate(far, [known], minimum_separation_km=100.0)
    assert result["status"] == "passed"
    assert result["minimum_observed_separation_km"] >= 100.0

    near = _external("candidate-near", (25.01, 60.15, 25.11, 60.25), observed=False)
    with pytest.raises(ValueError, match="minimum required separation"):
        validate_external_candidate(near, [known], minimum_separation_km=50.0)

    ledger = tmp_path / "holdout.jsonl"
    observed = _external("observed-copy", (29.0, 63.0, 29.1, 63.1))
    record_holdout_exposure(
        ledger,
        experiment_signature=EXPERIMENT,
        dataset_signature=DATASET,
        aois=[observed],
        purpose="external evaluation",
    )
    same_geometry = _external("future-copy", (29.0, 63.0, 29.1, 63.1), observed=False)
    with pytest.raises(ValueError, match="holdout exposure ledger"):
        validate_external_candidate(
            same_geometry,
            [],
            minimum_separation_km=0.0,
            exposure_ledger=ledger,
        )


def test_spatial_block_bootstrap_is_deterministic_and_not_patch_bootstrap_alias() -> None:
    blocks = [
        np.array([[20, 2], [1, 17]], dtype=np.int64),
        np.array([[8, 8], [2, 12]], dtype=np.int64),
        np.array([[15, 1], [9, 5]], dtype=np.int64),
    ]
    first = spatial_block_bootstrap_macro_iou_ci(blocks, iterations=300, seed=42)
    second = spatial_block_bootstrap_macro_iou_ci(blocks, iterations=300, seed=42)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0
