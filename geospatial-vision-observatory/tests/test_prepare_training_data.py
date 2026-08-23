from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import scripts.prepare_training_data as prep


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(root: Path, *, cloud: float = 4.0, max_pixels: int = 768) -> Path:
    directory = root / "helsinki_metro"
    directory.mkdir(parents=True)
    outputs: dict[str, dict[str, object]] = {}
    for name in ("sentinel2_multispectral.tif", "worldcover_2021_on_sentinel.tif"):
        path = directory / name
        path.write_bytes(name.encode())
        outputs[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 3,
        "aoi": {"key": "helsinki_metro"},
        "sentinel2": {
            "collection": "sentinel-2-c1-l2a",
            "requested_start": "2021-05-01T00:00:00+00:00",
            "requested_end": "2021-09-30T23:59:59.999999+00:00",
            "cloud_cover": cloud,
            "cloud_threshold_required": True,
            "cloud_quality_basis": "aoi_scl_obscured_percent",
            "aoi_scl_obscured_percent": 4.0,
        },
        "integrity": {"max_pixels": max_pixels, "outputs": outputs},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _complete(root: Path, **kwargs: object) -> bool:
    defaults = {
        "start_date": date(2021, 5, 1),
        "end_date": date(2021, 9, 30),
        "max_cloud": 15.0,
        "max_pixels": 768,
    }
    defaults.update(kwargs)
    return prep.scene_complete(root, "helsinki_metro", **defaults)  # type: ignore[arg-type]


def test_scene_complete_requires_exact_policy_and_integrity(tmp_path: Path) -> None:
    directory = _scene(tmp_path)
    assert _complete(tmp_path)
    assert not _complete(tmp_path, max_pixels=1024)
    assert not _complete(tmp_path, max_cloud=3.0)
    assert not _complete(tmp_path, start_date=date(2021, 6, 1))

    stack = directory / "sentinel2_multispectral.tif"
    stack.write_bytes(b"tampered")
    assert not _complete(tmp_path)


def test_scene_complete_rejects_unsafe_or_nonfinite_metadata(tmp_path: Path) -> None:
    directory = _scene(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sentinel2"]["cloud_cover"] = "nan"
    manifest_path.write_text(json.dumps(manifest))
    assert not _complete(tmp_path)

    directory = _scene(tmp_path / "other")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["integrity"]["outputs"].pop("sentinel2_multispectral.tif")
    manifest["integrity"]["outputs"]["../sentinel2_multispectral.tif"] = metadata
    manifest_path.write_text(json.dumps(manifest))
    assert not _complete(tmp_path / "other")


def test_scene_complete_rejects_wrong_sentinel_collection(tmp_path: Path) -> None:
    directory = _scene(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sentinel2"]["collection"] = "sentinel-2-l2a"
    manifest_path.write_text(json.dumps(manifest))
    assert not _complete(tmp_path)


def _selection_scene(root: Path, key: str) -> None:
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("sentinel2_multispectral.tif", "worldcover_2021_on_sentinel.tif"):
        (directory / name).write_bytes(f"{key}:{name}".encode())


def test_external_curation_requires_frozen_selection_evidence(tmp_path: Path) -> None:
    for key in (*prep.SHOWCASE_SPLIT.train, *prep.SHOWCASE_SPLIT.validation):
        _selection_scene(tmp_path, key)

    evidence = tmp_path / "seed-selection.json"
    try:
        prep.authorize_external_curation(tmp_path, evidence)
    except RuntimeError as error:
        assert "locked" in str(error)
    else:
        raise AssertionError("holdout curation should be locked without selection evidence")

    signature = prep.dataset_signature(
        [tmp_path / key for key in (*prep.SHOWCASE_SPLIT.train, *prep.SHOWCASE_SPLIT.validation)]
    )
    evidence.write_text(
        json.dumps(
            {
                "selection_dataset": "validation AOIs only",
                "external_test_used_for_selection": False,
                "candidates": [
                    {"seed": 11, "dataset_signature": signature},
                    {"seed": 12, "dataset_signature": signature},
                    {"seed": 13, "dataset_signature": signature},
                ],
                "selected_seed": 12,
            }
        )
    )
    prep.authorize_external_curation(tmp_path, evidence)

    (tmp_path / prep.SHOWCASE_SPLIT.train[0] / "sentinel2_multispectral.tif").write_bytes(b"drift")
    try:
        prep.authorize_external_curation(tmp_path, evidence)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("holdout curation should lock when selection data drifts")
