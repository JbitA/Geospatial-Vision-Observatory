from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
torch = pytest.importorskip("torch")
from rasterio.transform import from_origin  # noqa: E402

from geo_vision.ml.data import (  # noqa: E402
    class_weights,
    compute_normalization,
    load_scene,
    map_worldcover,
    patch_index,
)
from geo_vision.ml.inference import predict_scene, verify_bundle  # noqa: E402
from geo_vision.ml.metrics import confusion_matrix, metrics_from_confusion  # noqa: E402
from geo_vision.ml.model import build_model  # noqa: E402
from geo_vision.ml.schema import CLASS_NAMES, SHOWCASE_SPLIT  # noqa: E402
from geo_vision.ml.train import TrainConfig, train_landcover  # noqa: E402


def _write_scene(root: Path, key: str, phase: float) -> Path:
    directory = root / key
    directory.mkdir(parents=True)
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    tree = xx < 32
    labels = np.where(tree, 10, 50).astype(np.uint8)
    # Six reflectance bands with a learnable tree/built spectral distinction.
    red = np.where(tree, 0.12, 0.24) + phase + yy * 0.0001
    green = np.where(tree, 0.18, 0.22) + phase
    blue = np.where(tree, 0.08, 0.20) + phase
    nir = np.where(tree, 0.48, 0.18) + phase
    swir16 = np.where(tree, 0.20, 0.34) + phase
    swir22 = np.where(tree, 0.16, 0.30) + phase
    stack = np.stack((red, green, blue, nir, swir16, swir22)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 6,
        "dtype": "float32",
        "crs": "EPSG:32635",
        "transform": from_origin(500000, 7000000, 10, 10),
        "nodata": -9999.0,
    }
    stack_path = directory / "sentinel2_multispectral.tif"
    with rasterio.open(stack_path, "w", **profile) as dst:
        dst.write(stack)
        for index, name in enumerate(("red", "green", "blue", "nir", "swir16", "swir22"), 1):
            dst.set_band_description(index, name)
    label_profile = profile | {"count": 1, "dtype": "uint8", "nodata": 0}
    labels_path = directory / "worldcover_2021_on_sentinel.tif"
    with rasterio.open(labels_path, "w", **label_profile) as dst:
        dst.write(labels, 1)
    ImageProfile = profile | {"count": 1, "dtype": "uint8", "nodata": 0}
    scl_path = directory / "sentinel2_scl.tif"
    with rasterio.open(scl_path, "w", **ImageProfile) as dst:
        dst.write(np.full((height, width), 4, dtype=np.uint8), 1)
    preview = np.stack(
        [
            np.clip(red * 255, 0, 255),
            np.clip(green * 255, 0, 255),
            np.clip(blue * 255, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    from PIL import Image

    Image.fromarray(preview).save(directory / "sentinel2_preview.png")
    integrity = {}
    for path in (stack_path, labels_path, scl_path):
        integrity[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "aoi": {"key": key},
                "sentinel2": {"item_id": key},
                "integrity": {"outputs": integrity},
            }
        )
    )
    return directory


def _all_split_scenes(root: Path) -> None:
    keys = (*SHOWCASE_SPLIT.train, *SHOWCASE_SPLIT.validation, *SHOWCASE_SPLIT.external_test)
    for index, key in enumerate(keys):
        _write_scene(root, key, index * 0.001)


def test_model_shape_and_parameter_efficiency() -> None:
    model = build_model(base_channels=8)
    output = model(torch.zeros(2, 6, 64, 64))
    assert output.shape == (2, len(CLASS_NAMES), 64, 64)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_scene_loading_normalization_and_patch_index(tmp_path: Path) -> None:
    directory = _write_scene(tmp_path, "scene", 0.0)
    scene = load_scene(directory)
    assert scene.image.shape == (6, 64, 64)
    assert set(np.unique(scene.target)) == {0, 4}
    refs = patch_index([scene], patch_size=32, stride=32)
    assert len(refs) == 4
    mean, std = compute_normalization([scene], sample_pixels_per_scene=500)
    assert mean.shape == (6,)
    assert np.all(std > 0)
    weights = class_weights([scene])
    assert weights[0] > 0
    assert weights[4] > 0
    assert weights[1] == 0


def test_segmentation_metrics() -> None:
    truth = np.array([[0, 0, 4, 4], [0, 255, 4, 4]], dtype=np.uint8)
    pred = np.array([[0, 4, 4, 4], [0, 0, 0, 4]], dtype=np.uint8)
    matrix = confusion_matrix(truth, pred, len(CLASS_NAMES))
    metrics = metrics_from_confusion(matrix)
    assert matrix.sum() == 7
    assert 0 < metrics["macro_iou"] < 1
    mapped = map_worldcover(np.array([[10, 50, 0]], dtype=np.uint8))
    assert mapped.tolist() == [[0, 4, 255]]


def test_end_to_end_training_bundle_and_inference(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _all_split_scenes(data_root)
    bundle = tmp_path / "models" / "landcover"
    reports = tmp_path / "reports" / "landcover"
    result = train_landcover(
        data_root,
        bundle,
        reports,
        config=TrainConfig(
            epochs=1,
            batch_size=2,
            patch_size=32,
            stride=32,
            base_channels=4,
            patience=1,
            bootstrap_iterations=20,
        ),
        requested_device="cpu",
        force=True,
    )
    assert result["external_test"]["patches"] == 8
    manifest = verify_bundle(bundle)
    assert manifest["task"] == "worldcover_landcover_segmentation"
    output = tmp_path / "prediction.tif"
    inference = predict_scene(
        data_root / SHOWCASE_SPLIT.external_test[0] / "sentinel2_multispectral.tif",
        bundle,
        output,
        device="cpu",
        tile_size=32,
    )
    assert output.is_file()
    assert len(inference["prediction_sha256"]) == 64


def test_training_config_enforces_resource_and_numeric_bounds() -> None:
    invalid = (
        {"epochs": 0},
        {"batch_size": 0},
        {"patch_size": 16},
        {"stride": 8},
        {"stride": 256, "patch_size": 128},
        {"learning_rate": float("nan")},
        {"weight_decay": -1.0},
        {"base_channels": 2},
        {"label_smoothing": 1.0},
        {"min_valid_fraction": 0.0},
        {"patience": 15},
        {"workers": 99},
        {"bootstrap_iterations": 0},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)


def test_inference_rejects_wrong_band_order_and_missing_crs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _all_split_scenes(data_root)
    bundle = tmp_path / "models" / "landcover"
    reports = tmp_path / "reports" / "landcover"
    train_landcover(
        data_root,
        bundle,
        reports,
        config=TrainConfig(
            epochs=1,
            batch_size=2,
            patch_size=32,
            stride=32,
            base_channels=4,
            patience=1,
            bootstrap_iterations=20,
        ),
        requested_device="cpu",
        force=True,
    )
    source = data_root / SHOWCASE_SPLIT.external_test[0] / "sentinel2_multispectral.tif"

    wrong_order = tmp_path / "wrong-order.tif"
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        values = src.read()
    with rasterio.open(wrong_order, "w", **profile) as dst:
        dst.write(values)
        for index, name in enumerate(("blue", "green", "red", "nir", "swir16", "swir22"), 1):
            dst.set_band_description(index, name)
    with pytest.raises(ValueError, match="band descriptions"):
        predict_scene(wrong_order, bundle, tmp_path / "wrong.tif", tile_size=32)

    no_crs = tmp_path / "no-crs.tif"
    profile.pop("crs", None)
    with rasterio.open(no_crs, "w", **profile) as dst:
        dst.write(values)
        for index, name in enumerate(("red", "green", "blue", "nir", "swir16", "swir22"), 1):
            dst.set_band_description(index, name)
    with pytest.raises(ValueError, match="coordinate reference system"):
        predict_scene(no_crs, bundle, tmp_path / "no-crs-out.tif", tile_size=32)


def test_lazy_scene_patch_and_statistics_match_eager_scene(tmp_path: Path) -> None:
    from geo_vision.ml.data import load_lazy_scene

    directory = _write_scene(tmp_path, "lazy-scene", 0.002)
    eager = load_scene(directory)
    lazy = load_lazy_scene(directory)
    assert lazy.width == eager.image.shape[2]
    assert lazy.height == eager.image.shape[1]
    image, target, valid = lazy.read_patch(8, 12, 32, 32)
    np.testing.assert_allclose(image, eager.image[:, 8:40, 12:44], equal_nan=True)
    np.testing.assert_array_equal(target, eager.target[8:40, 12:44])
    np.testing.assert_array_equal(valid, eager.valid[8:40, 12:44])

    eager_refs = patch_index([eager], patch_size=32, stride=32)
    lazy_refs = patch_index([lazy], patch_size=32, stride=32)
    assert lazy_refs == eager_refs
    eager_mean, eager_std = compute_normalization([eager], sample_pixels_per_scene=100_000, seed=7)
    lazy_mean, lazy_std = compute_normalization([lazy], sample_pixels_per_scene=100_000, seed=7)
    np.testing.assert_allclose(lazy_mean, eager_mean, rtol=0, atol=1e-7)
    np.testing.assert_allclose(lazy_std, eager_std, rtol=0, atol=1e-7)
    np.testing.assert_allclose(class_weights([lazy]), class_weights([eager]), rtol=0, atol=1e-7)


def test_lazy_scene_pixel_reads_are_window_bounded(tmp_path: Path, monkeypatch) -> None:
    from geo_vision.ml import data as ml_data

    directory = _write_scene(tmp_path, "windowed-scene", 0.0)
    lazy = ml_data.load_lazy_scene(directory)
    real_open = rasterio.open
    reads: list[object] = []

    class ReaderProxy:
        def __init__(self, dataset):
            self._dataset = dataset

        def __enter__(self):
            self._dataset.__enter__()
            return self

        def __exit__(self, *args):
            return self._dataset.__exit__(*args)

        def read(self, *args, **kwargs):
            reads.append(kwargs.get("window"))
            return self._dataset.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    monkeypatch.setattr(rasterio, "open", lambda *args, **kwargs: ReaderProxy(real_open(*args, **kwargs)))
    image, target, valid = lazy.read_patch(4, 5, 32, 32)
    assert image.shape == (6, 32, 32)
    assert target.shape == valid.shape == (32, 32)
    assert len(reads) == 3
    assert all(window is not None for window in reads)


def test_lazy_scene_sampled_normalization_matches_eager_scene(tmp_path: Path) -> None:
    from geo_vision.ml.data import load_lazy_scene

    directory = _write_scene(tmp_path, "lazy-sampled", 0.003)
    eager = load_scene(directory)
    lazy = load_lazy_scene(directory)

    eager_mean, eager_std = compute_normalization([eager], sample_pixels_per_scene=257, seed=91)
    lazy_mean, lazy_std = compute_normalization([lazy], sample_pixels_per_scene=257, seed=91)

    np.testing.assert_array_equal(lazy_mean, eager_mean)
    np.testing.assert_array_equal(lazy_std, eager_std)
