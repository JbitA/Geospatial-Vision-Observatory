from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import scripts.prepare_geospatial_dataset as prep
from geo_vision.geodata import CURATED_AOIS


def _write_raster(path: Path, data: np.ndarray, *, dtype: str) -> None:
    aoi = CURATED_AOIS["helsinki_metro"]
    transform = from_bounds(*aoi.bbox, width=data.shape[1], height=data.shape[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def test_build_dataset_offline_alignment_and_manifest(tmp_path: Path, monkeypatch) -> None:
    shape = (32, 32)
    band_values = {
        "red": 2000,
        "green": 3000,
        "blue": 1800,
        "nir": 4000,
        "swir16": 3000,
        "swir22": 2500,
    }
    assets: dict[str, dict[str, object]] = {}
    for name, value in band_values.items():
        path = tmp_path / f"{name}.tif"
        _write_raster(path, np.full(shape, value, dtype=np.uint16), dtype="uint16")
        assets[name] = {
            "href": str(path),
            "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
        }

    scl_path = tmp_path / "scl.tif"
    scl = np.full(shape, 4, dtype=np.uint8)
    scl[0, :] = 9
    _write_raster(scl_path, scl, dtype="uint8")
    assets["scl"] = {"href": str(scl_path)}

    worldcover_path = tmp_path / "worldcover.tif"
    worldcover = np.full(shape, 10, dtype=np.uint8)
    worldcover[:, 16:] = 50
    _write_raster(worldcover_path, worldcover, dtype="uint8")

    tree_path = tmp_path / "treecover.tif"
    _write_raster(tree_path, np.full(shape, 70, dtype=np.uint8), dtype="uint8")
    loss_path = tmp_path / "lossyear.tif"
    lossyear = np.zeros(shape, dtype=np.uint8)
    lossyear[:4, :4] = 25
    _write_raster(loss_path, lossyear, dtype="uint8")

    item = {
        "id": "synthetic-sentinel-item",
        "properties": {
            "datetime": "2026-08-01T10:00:00Z",
            "eo:cloud_cover": 4.5,
        },
        "assets": assets,
    }
    monkeypatch.setattr(prep, "stac_item", lambda *_args, **_kwargs: item)
    # Local GeoTIFF fixtures deliberately bypass the production HTTPS asset-host validator.
    monkeypatch.setattr(prep, "validated_asset_href", lambda value: str(value))
    monkeypatch.setattr(prep, "worldcover_map_url", lambda *_args: str(worldcover_path))
    monkeypatch.setattr(prep, "scl_asset_href", lambda *_args: str(scl_path))
    monkeypatch.setattr(
        prep,
        "hansen_url",
        lambda layer, *_args: str(tree_path if layer == "treecover2000" else loss_path),
    )

    manifest_path = prep.build_dataset("helsinki_metro", tmp_path / "out", 30, 20.0)
    manifest = json.loads(manifest_path.read_text())
    output = manifest_path.parent

    assert (output / "sentinel2_multispectral.tif").is_file()
    assert (output / "sentinel2_indices.tif").is_file()
    assert (output / "sentinel2_preview.png").is_file()
    assert (output / "sentinel2_scl.tif").is_file()
    assert (output / "worldcover_2021_on_sentinel.tif").is_file()
    assert (output / "hansen_treecover2000_on_sentinel.tif").is_file()
    assert (output / "hansen_lossyear_on_sentinel.tif").is_file()
    assert manifest["sentinel2"]["bands"] == list(prep.SENTINEL_BANDS)
    assert manifest["sentinel2"]["band_calibration"]["red"]["scale"] == 0.0001
    assert manifest["sentinel2"]["band_calibration"]["red"]["offset"] == -0.1
    assert manifest["sentinel2"]["scl_valid_fraction"] == 0.96875
    assert manifest["worldcover"]["class_fractions"] == {
        "built_up": 0.5,
        "tree_cover": 0.5,
    }
    assert manifest["hansen"]["loss_pixels_2001_2025"] == 16
    assert manifest["sentinel2"]["indices"]["ndvi"]["mean"] == 0.5

    with rasterio.open(output / "sentinel2_multispectral.tif") as src:
        assert src.count == 6
        assert src.descriptions == prep.SENTINEL_BANDS
        assert np.isclose(src.read(1)[1, 1], 0.1)
    with rasterio.open(output / "sentinel2_indices.tif") as src:
        assert src.count == 3
        assert src.descriptions == ("ndvi", "ndwi", "ndbi")


def test_asset_url_and_calibration_policy() -> None:
    assert prep.validated_asset_href(
        "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/example.tif"
    ).endswith("example.tif")
    for unsafe in (
        "http://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/example.tif",
        "https://user@e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/example.tif",
        "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com:444/example.tif",
        "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/example.tif#fragment",
        "https://example.com/example.tif",
    ):
        try:
            prep.validated_asset_href(unsafe)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion helper
            raise AssertionError(f"unsafe asset URL accepted: {unsafe}")

    assert prep.asset_calibration({"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}) == {
        "scale": 0.0001,
        "offset": -0.1,
        "nodata": 0.0,
    }
    for bad in (
        {"raster:bands": [{"scale": 0.0}]},
        {"raster:bands": [{"scale": float("nan")}]},
        {"raster:bands": [{"offset": float("inf")}]},
        {"raster:bands": [{"nodata": float("nan")}]},
    ):
        try:
            prep.asset_calibration(bad)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion helper
            raise AssertionError(f"invalid calibration accepted: {bad}")


def test_stac_item_rejects_invalid_metadata_and_respects_window(monkeypatch) -> None:
    asset = {"href": "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/example.tif"}
    assets = {name: asset for name in prep.SENTINEL_BANDS}
    assets["scl"] = asset
    payload = {
        "features": [
            {
                "id": "naive-time",
                "properties": {"datetime": "2021-06-01T10:00:00", "eo:cloud_cover": 1},
                "assets": assets,
            },
            {
                "id": "nan-cloud",
                "properties": {"datetime": "2021-06-02T10:00:00Z", "eo:cloud_cover": "nan"},
                "assets": assets,
            },
            {
                "id": "outside-window",
                "properties": {"datetime": "2021-10-01T10:00:00Z", "eo:cloud_cover": 0},
                "assets": assets,
            },
            {
                "id": "valid",
                "properties": {"datetime": "2021-07-01T10:00:00Z", "eo:cloud_cover": 5},
                "assets": assets,
            },
        ]
    }
    monkeypatch.setattr(prep, "bounded_json_get", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(prep, "scl_obscured_percent", lambda *_args, **_kwargs: 2.0)
    from datetime import UTC, datetime

    selected = prep.stac_item(
        CURATED_AOIS["helsinki_metro"].bbox,
        30,
        10,
        start=datetime(2021, 5, 1, tzinfo=UTC),
        end=datetime(2021, 9, 30, tzinfo=UTC),
        require_cloud_threshold=True,
    )
    assert selected["id"] == "valid"


def test_stac_item_uses_aoi_scl_quality_when_scene_metadata_is_cloudy(monkeypatch) -> None:
    base = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"

    def item(item_id: str, stamp: str, scene_cloud: float, scl_name: str) -> dict[str, object]:
        assets = {
            name: {"href": f"{base}/{item_id}-{name}.tif"}
            for name in prep.SENTINEL_BANDS
        }
        assets["scl"] = {"href": f"{base}/{scl_name}.tif"}
        return {
            "id": item_id,
            "properties": {"datetime": stamp, "eo:cloud_cover": scene_cloud},
            "assets": assets,
        }

    payload = {
        "features": [
            item("cloudy-granule-clear-aoi", "2021-07-10T10:00:00Z", 28.0, "clear"),
            item("lower-scene-cloud-bad-aoi", "2021-07-11T10:00:00Z", 18.0, "obscured"),
        ]
    }
    monkeypatch.setattr(prep, "bounded_json_get", lambda *_args, **_kwargs: payload)

    def quality(href: str, *_args: object, **_kwargs: object) -> float:
        return 4.0 if href.endswith("clear.tif") else 35.0

    monkeypatch.setattr(prep, "scl_obscured_percent", quality)
    from datetime import UTC, datetime

    selected = prep.stac_item(
        CURATED_AOIS["north_karelia_forest"].bbox,
        30,
        15.0,
        start=datetime(2021, 5, 1, tzinfo=UTC),
        end=datetime(2021, 9, 30, tzinfo=UTC),
        require_cloud_threshold=True,
    )
    assert selected["id"] == "cloudy-granule-clear-aoi"
    assert selected["properties"]["gvo:aoi_scl_obscured_percent"] == 4.0


def test_rgb_preview_replaces_non_finite_values_before_uint8_cast() -> None:
    red = np.array([[0.1, np.nan], [np.inf, 0.3]], dtype=np.float32)
    green = np.array([[0.2, 0.25], [0.3, 0.35]], dtype=np.float32)
    blue = np.array([[0.15, 0.2], [0.25, -np.inf]], dtype=np.float32)
    with np.errstate(all="raise"):
        preview = prep.robust_rgb_preview(red, green, blue)
    assert preview.dtype == np.uint8
    assert preview.shape == (2, 2, 3)
    assert np.isfinite(preview).all()
