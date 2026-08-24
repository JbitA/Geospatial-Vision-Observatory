from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from geo_vision.ml.inference import _block_size, _runtime_contract, colorize_prediction
from geo_vision.ml.integrity import REQUIRED_SUPPORT_FILES, verify_bundle
from geo_vision.ml.schema import CLASS_NAMES, INPUT_BANDS, WORLDCOVER_CODES


def _write_bundle(
    root: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    normalization: dict[str, object] | None = None,
    classes: dict[str, object] | None = None,
    training_config: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    root.mkdir(parents=True)
    model_name = "landcover_multispectral.pt2"
    payloads: dict[str, bytes] = {
        model_name: b"not-loaded-by-contract-tests",
        "normalization.json": (
            json.dumps(
                normalization
                or {
                    "bands": INPUT_BANDS,
                    "mean": [0.1] * len(INPUT_BANDS),
                    "std": [0.2] * len(INPUT_BANDS),
                }
            )
            + "\n"
        ).encode(),
        "classes.json": (
            json.dumps(
                classes
                or {"class_names": CLASS_NAMES, "worldcover_codes": WORLDCOVER_CODES}
            )
            + "\n"
        ).encode(),
        "training-config.json": (json.dumps(training_config or {"patch_size": 64}) + "\n").encode(),
        "MODEL_CARD.md": b"# Model card\n",
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    files = {
        name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        for name, content in payloads.items()
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "task": "worldcover_landcover_segmentation",
        "dataset_signature": "a" * 64,
        "experiment_signature": "b" * 64,
        "model_file": model_name,
        "input_bands": list(INPUT_BANDS),
        "class_names": list(CLASS_NAMES),
        "files": files,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (root / "bundle.json").write_text(json.dumps(manifest))
    return root, manifest


def test_bundle_verifier_rejects_manifest_and_file_metadata_tampering(tmp_path: Path) -> None:
    bundle, manifest = _write_bundle(tmp_path / "bundle")
    assert verify_bundle(bundle)["task"] == "worldcover_landcover_segmentation"

    (bundle / "bundle.json").write_text("{")
    with pytest.raises(ValueError, match="unreadable"):
        verify_bundle(bundle)

    (bundle / "bundle.json").write_text(json.dumps(manifest | {"schema_version": "2.0"}))
    with pytest.raises(ValueError, match="schema"):
        verify_bundle(bundle)

    invalid = dict(manifest)
    invalid["model_file"] = "../escape.pt2"
    (bundle / "bundle.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="unsafe"):
        verify_bundle(bundle)


def test_bundle_verifier_rejects_unexpected_hash_size_and_symlink(tmp_path: Path) -> None:
    bundle, manifest = _write_bundle(tmp_path / "bundle")
    files = dict(manifest["files"])  # type: ignore[arg-type]
    files["unexpected.txt"] = {"sha256": "0" * 64, "bytes": 0}
    (bundle / "bundle.json").write_text(json.dumps(manifest | {"files": files}))
    with pytest.raises(ValueError, match="file set"):
        verify_bundle(bundle)

    bundle, manifest = _write_bundle(tmp_path / "bundle2")
    files = dict(manifest["files"])  # type: ignore[arg-type]
    model_name = str(manifest["model_file"])
    files[model_name] = {"sha256": "invalid", "bytes": 1}
    (bundle / "bundle.json").write_text(json.dumps(manifest | {"files": files}))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle(bundle)

    bundle, manifest = _write_bundle(tmp_path / "bundle3")
    files = dict(manifest["files"])  # type: ignore[arg-type]
    support = next(iter(REQUIRED_SUPPORT_FILES))
    support_meta = dict(files[support])  # type: ignore[arg-type]
    support_meta["bytes"] = True
    files[support] = support_meta
    (bundle / "bundle.json").write_text(json.dumps(manifest | {"files": files}))
    with pytest.raises(ValueError, match="byte count"):
        verify_bundle(bundle)

    bundle, manifest = _write_bundle(tmp_path / "bundle4")
    target = tmp_path / "real-card.md"
    target.write_text("card")
    card = bundle / "MODEL_CARD.md"
    card.unlink()
    card.symlink_to(target)
    files = dict(manifest["files"])  # type: ignore[arg-type]
    files["MODEL_CARD.md"] = {
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
    }
    (bundle / "bundle.json").write_text(json.dumps(manifest | {"files": files}))
    with pytest.raises(ValueError, match="regular file"):
        verify_bundle(bundle)


def test_runtime_contract_rejects_semantically_incompatible_support_files(tmp_path: Path) -> None:
    bundle, manifest = _write_bundle(
        tmp_path / "bands",
        manifest_overrides={"input_bands": ["blue"]},
    )
    with pytest.raises(ValueError, match="input bands"):
        _runtime_contract(bundle, verify_bundle(bundle))

    bundle, manifest = _write_bundle(
        tmp_path / "classes",
        classes={"class_names": ["wrong"], "worldcover_codes": WORLDCOVER_CODES},
    )
    with pytest.raises(ValueError, match="class metadata"):
        _runtime_contract(bundle, verify_bundle(bundle))

    bundle, manifest = _write_bundle(
        tmp_path / "normalization",
        normalization={"bands": INPUT_BANDS, "mean": [0.1], "std": [0.2]},
    )
    with pytest.raises(ValueError, match="normalization shape"):
        _runtime_contract(bundle, verify_bundle(bundle))

    bundle, manifest = _write_bundle(
        tmp_path / "std",
        normalization={"bands": INPUT_BANDS, "mean": [0.1] * 6, "std": [0.0] * 6},
    )
    with pytest.raises(ValueError, match="normalization values"):
        _runtime_contract(bundle, verify_bundle(bundle))

    bundle, manifest = _write_bundle(tmp_path / "patch", training_config={"patch_size": True})
    with pytest.raises(ValueError, match="patch size"):
        _runtime_contract(bundle, verify_bundle(bundle))


def test_small_raster_blocking_and_colorization_are_deterministic() -> None:
    assert _block_size(15) == 0
    assert _block_size(16) == 16
    assert _block_size(400) == 256
    prediction = np.array([[0, 4, 255]], dtype=np.uint8)
    rgb = colorize_prediction(prediction)
    assert rgb.shape == (1, 3, 3)
    assert tuple(rgb[0, 0]) != tuple(rgb[0, 1])
    assert tuple(rgb[0, 2]) == (0, 0, 0)
