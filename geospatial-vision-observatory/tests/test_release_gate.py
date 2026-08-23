from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.release_gate as gate
from geo_vision.ml.schema import SHOWCASE_SPLIT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curated_scene(
    root: Path,
    key: str,
    *,
    cloud: object = 5.0,
    aoi_obscured: object = 5.0,
    strict: bool = True,
    collection: str = "sentinel-2-c1-l2a",
) -> str:
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for name in ("sentinel2_multispectral.tif", "worldcover_2021_on_sentinel.tif"):
        path = directory / name
        path.write_bytes(f"{key}:{name}".encode())
        outputs[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 3,
        "aoi": {"key": key},
        "sentinel2": {
            "collection": collection,
            "requested_start": "2021-05-01T00:00:00+00:00",
            "requested_end": "2021-09-30T23:59:59.999999+00:00",
            "cloud_cover": cloud,
            "cloud_threshold_required": strict,
            "cloud_quality_basis": "aoi_scl_obscured_percent",
            "aoi_scl_obscured_percent": aoi_obscured,
        },
        "integrity": {"max_pixels": 768, "outputs": outputs},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return _sha(manifest_path)


def _split(
    root: Path,
    *,
    bad_key: str | None = None,
    cloud: object = 5.0,
    aoi_obscured: object = 5.0,
    strict: bool = True,
    collection: str = "sentinel-2-c1-l2a",
) -> None:
    assignments = []
    for split_name, keys in (
        ("train", SHOWCASE_SPLIT.train),
        ("validation", SHOWCASE_SPLIT.validation),
        ("external_test", SHOWCASE_SPLIT.external_test),
    ):
        for key in keys:
            scene_cloud = cloud if key == bad_key else 5.0
            scene_aoi_obscured = aoi_obscured if key == bad_key else 5.0
            scene_strict = strict if key == bad_key else True
            assignments.append(
                {
                    "aoi": key,
                    "split": split_name,
                    "status": "prepared",
                    "manifest_sha256": _curated_scene(
                        root,
                        key,
                        cloud=scene_cloud,
                        aoi_obscured=scene_aoi_obscured,
                        strict=scene_strict,
                        collection=collection if key == bad_key else "sentinel-2-c1-l2a",
                    ),
                }
            )
    (root / "training-split.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "temporal_window": ["2021-05-01", "2021-09-30"],
                "max_cloud_percent": 15.0,
                "cloud_quality_basis": "aoi_scl_obscured_percent",
                "max_pixels": 768,
                "assignment": assignments,
            }
        )
    )


def test_release_gate_curated_split_fails_closed_on_scene_policy(tmp_path: Path) -> None:
    policy = {
        "temporal_window": ["2021-05-01", "2021-09-30"],
        "max_cloud_percent": 15.0,
        "cloud_quality_basis": "aoi_scl_obscured_percent",
        "max_pixels": 768,
    }
    _split(tmp_path)
    failures: list[str] = []
    gate._verify_curated_split(tmp_path, policy, failures)
    assert failures == []

    _split(tmp_path, bad_key="helsinki_metro", cloud="nan")
    failures = []
    gate._verify_curated_split(tmp_path, policy, failures)
    assert any("cloud-cover metadata invalid" in item for item in failures)

    _split(tmp_path, bad_key="helsinki_metro", strict=False)
    failures = []
    gate._verify_curated_split(tmp_path, policy, failures)
    assert any("strict cloud-threshold mode" in item for item in failures)

    _split(tmp_path, bad_key="helsinki_metro", aoi_obscured=22.0)
    failures = []
    gate._verify_curated_split(tmp_path, policy, failures)
    assert any("AOI exceeds obscured-pixel threshold" in item for item in failures)

    _split(tmp_path, bad_key="helsinki_metro", collection="sentinel-2-l2a")
    failures = []
    gate._verify_curated_split(tmp_path, policy, failures)
    assert any("unexpected Sentinel collection" in item for item in failures)


def test_seed_selection_requires_validation_only_winner() -> None:
    splits = {
        "train": list(SHOWCASE_SPLIT.train),
        "validation": list(SHOWCASE_SPLIT.validation),
    }
    rows = [
        {
            "seed": seed,
            "macro_iou": macro,
            "weighted_iou": weighted,
            "dataset_signature": "a" * 64,
            "experiment_signature": f"{seed:064x}",
            "splits": splits,
            "patch_counts": {"train": 10, "validation": 5},
        }
        for seed, macro, weighted in ((1, 0.4, 0.6), (2, 0.5, 0.55), (3, 0.5, 0.65))
    ]
    report = {
        "schema_version": "1.0",
        "selection_dataset": "validation AOIs only",
        "selection_rule": "maximum validation macro IoU, then weighted IoU",
        "external_test_used_for_selection": False,
        "candidates": rows,
        "selected_seed": 3,
    }
    failures: list[str] = []
    gate._verify_seed_selection(report, {"minimum_validation_seeds": 3}, failures)
    assert failures == []

    report["selected_seed"] = 1
    failures = []
    gate._verify_seed_selection(report, {"minimum_validation_seeds": 3}, failures)
    assert "recorded selected seed is not the declared validation-metric winner" in failures


def test_seed_selection_detects_dataset_drift() -> None:
    splits = {
        "train": list(SHOWCASE_SPLIT.train),
        "validation": list(SHOWCASE_SPLIT.validation),
    }
    rows = [
        {
            "seed": seed,
            "macro_iou": 0.4 + seed * 0.01,
            "weighted_iou": 0.6,
            "dataset_signature": "a" * 64,
            "experiment_signature": f"{seed:064x}",
            "splits": splits,
            "patch_counts": {"train": 10, "validation": 5},
        }
        for seed in (1, 2, 3)
    ]
    report = {
        "schema_version": "1.0",
        "selection_dataset": "validation AOIs only",
        "selection_rule": "maximum validation macro IoU, then weighted IoU",
        "external_test_used_for_selection": False,
        "candidates": rows,
        "selected_seed": 3,
    }
    failures: list[str] = []
    gate._verify_seed_selection(
        report,
        {"minimum_validation_seeds": 3},
        failures,
        expected_dataset_signature="b" * 64,
    )
    assert "seed-selection dataset no longer matches current train/validation data" in failures
