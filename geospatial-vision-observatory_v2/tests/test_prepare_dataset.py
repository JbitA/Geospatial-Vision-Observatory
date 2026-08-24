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


def test_stac_item_set_selects_same_overpass_tiles_to_cover_cell(monkeypatch) -> None:
    base = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"

    def item(item_id: str, west: float, east: float, cloud: float) -> dict[str, object]:
        assets = {name: {"href": f"{base}/{item_id}-{name}.tif"} for name in prep.SENTINEL_BANDS}
        assets["scl"] = {"href": f"{base}/{item_id}-scl.tif"}
        return {
            "type": "Feature",
            "id": item_id,
            "bbox": [west, 60.0, east, 60.5],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[west, 60.0], [east, 60.0], [east, 60.5], [west, 60.5], [west, 60.0]]],
            },
            "properties": {"datetime": "2021-07-01T10:00:00Z", "eo:cloud_cover": cloud},
            "assets": assets,
        }

    payload = {"features": [item("right", 24.5, 25.0, 4.0), item("left", 24.0, 24.5, 5.0)]}
    monkeypatch.setattr(prep, "bounded_json_get", lambda *_args, **_kwargs: payload)
    from datetime import UTC, datetime

    selected = prep.stac_item_set(
        (24.0, 60.0, 25.0, 60.5),
        30,
        10.0,
        start=datetime(2021, 6, 1, tzinfo=UTC),
        end=datetime(2021, 8, 1, tzinfo=UTC),
    )
    assert {row["id"] for row in selected} == {"left", "right"}


def test_stac_item_set_fails_closed_when_no_overpass_fully_covers_cell(monkeypatch) -> None:
    base = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"

    def item(item_id: str, west: float, east: float, stamp: str) -> dict[str, object]:
        assets = {name: {"href": f"{base}/{item_id}-{name}.tif"} for name in prep.SENTINEL_BANDS}
        return {
            "type": "Feature",
            "id": item_id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[west, 60.0], [east, 60.0], [east, 60.5], [west, 60.5], [west, 60.0]]],
            },
            "properties": {"datetime": stamp, "eo:cloud_cover": 2.0},
            "assets": assets,
        }

    payload = {
        "features": [
            item("left", 24.0, 24.5, "2021-07-01T10:00:00Z"),
            item("right-next-day", 24.5, 25.0, "2021-07-02T10:00:00Z"),
        ]
    }
    monkeypatch.setattr(prep, "bounded_json_get", lambda *_args, **_kwargs: payload)
    from datetime import UTC, datetime
    import pytest

    with pytest.raises(RuntimeError, match="fully covers"):
        prep.stac_item_set(
            (24.0, 60.0, 25.0, 60.5),
            30,
            10.0,
            start=datetime(2021, 6, 1, tzinfo=UTC),
            end=datetime(2021, 8, 1, tzinfo=UTC),
        )


def test_build_processing_cell_dataset_executes_multi_reference_tiles_offline(
    tmp_path: Path, monkeypatch
) -> None:
    from geo_vision.aoi import aoi_from_bbox
    from geo_vision.cells import plan_processing_grid

    aoi = aoi_from_bbox("scaled", (19.8, 59.8, 24.2, 60.2))
    grid = plan_processing_grid(aoi, resolution_m=1000.0, cell_pixels=256, halo_pixels=0)
    cell = next(
        row
        for row in grid.cells
        if len(row.worldcover_tiles) > 1 and len(row.hansen_tiles) > 1
    )
    west, south, east, north = cell.wgs84_bbox
    middle = (west + east) / 2.0

    def write_bounds(path: Path, value: int, bounds: tuple[float, float, float, float], *, nodata: int = 255) -> None:
        data = np.full((64, 64), value, dtype=np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_bounds(*bounds, width=64, height=64),
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)

    item_rows = []
    for index, bounds in enumerate(((west, south, middle, north), (middle, south, east, north))):
        assets: dict[str, dict[str, object]] = {}
        for band_index, name in enumerate(prep.SENTINEL_BANDS, start=1):
            path = tmp_path / f"sentinel-{index}-{name}.tif"
            write_bounds(path, 1000 + (band_index * 100) + (index * 10), bounds)
            assets[name] = {
                "href": str(path),
                "raster:bands": [{"scale": 0.0001, "offset": 0.0, "nodata": 255}],
            }
        scl_path = tmp_path / f"sentinel-{index}-scl.tif"
        write_bounds(scl_path, 4, bounds)
        assets["scl"] = {"href": str(scl_path)}
        item_rows.append(
            {
                "id": f"item-{index}",
                "properties": {"datetime": "2021-07-01T10:00:00Z", "eo:cloud_cover": 2.0},
                "assets": assets,
            }
        )

    def parse_worldcover(tile: str) -> tuple[float, float, float, float]:
        lat = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
        lon = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
        return (lon, lat, lon + 3, lat + 3)

    wc_paths: dict[str, Path] = {}
    for index, tile in enumerate(cell.worldcover_tiles, start=1):
        path = tmp_path / f"wc-{tile}.tif"
        write_bounds(path, 10 if index % 2 else 50, parse_worldcover(tile))
        wc_paths[tile] = path

    def parse_hansen(tile: str) -> tuple[float, float, float, float]:
        lat_top = int(tile[0:2]) * (1 if tile[2] == "N" else -1)
        lon_left = int(tile[4:7]) * (1 if tile[7] == "E" else -1)
        return (lon_left, lat_top - 10, lon_left + 10, lat_top)

    tree_paths: dict[str, Path] = {}
    loss_paths: dict[str, Path] = {}
    for index, tile in enumerate(cell.hansen_tiles, start=1):
        bounds = parse_hansen(tile)
        tree = tmp_path / f"tree-{tile}.tif"
        loss = tmp_path / f"loss-{tile}.tif"
        write_bounds(tree, 30 + index, bounds)
        write_bounds(loss, index, bounds)
        tree_paths[tile] = tree
        loss_paths[tile] = loss

    monkeypatch.setattr(prep, "stac_item_set", lambda *_args, **_kwargs: item_rows)
    monkeypatch.setattr(prep, "validated_asset_href", lambda value: str(value))
    monkeypatch.setattr(prep, "scl_asset_href", lambda assets: str(assets["scl"]["href"]))
    monkeypatch.setattr(prep, "worldcover_map_url_for_tile", lambda tile: str(wc_paths[tile]))
    monkeypatch.setattr(
        prep,
        "hansen_url_for_tile",
        lambda layer, tile: str(tree_paths[tile] if layer == "treecover2000" else loss_paths[tile]),
    )

    manifest_path = prep.build_processing_cell_dataset(
        aoi,
        grid,
        cell,
        tmp_path / "out",
        30,
        20.0,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 6
    assert manifest["dataset_layout"] == "processing_cell_v2"
    assert len(manifest["run_id"]) == 64
    assert len(manifest["recipe_id"]) == 64
    assert len(manifest["source_selection_id"]) == 64
    assert len(manifest["sentinel2"]["item_ids"]) == 2
    assert len(manifest["worldcover"]["sources"]) == len(cell.worldcover_tiles) > 1
    assert len(manifest["hansen"]["treecover2000_sources"]) == len(cell.hansen_tiles) > 1
    assert manifest["cell"]["cell_id"] == cell.cell_id
    with rasterio.open(manifest_path.parent / "sentinel2_multispectral.tif") as src:
        red = src.read(1)
        valid = red != -9999.0
        assert valid.any()
        assert len(np.unique(np.round(red[valid], 4))) >= 2
    with rasterio.open(manifest_path.parent / "worldcover_2021_on_cell.tif") as src:
        labels = src.read(1)
        assert np.any(labels != 0)


def test_processing_cell_collection_manifest_preserves_aoi_as_scientific_unit(
    tmp_path: Path, monkeypatch
) -> None:
    from jsonschema import Draft202012Validator
    from geo_vision.aoi import aoi_from_bbox

    aoi = aoi_from_bbox("collection", (24.8, 60.1, 24.9, 60.2))

    def fake_cell_builder(aoi_arg, grid, cell, output_root, *_args, **kwargs):
        import hashlib

        assert aoi_arg.aoi_id == aoi.aoi_id
        cells_root = kwargs["cells_root"]
        run_id = kwargs["run_id"]
        recipe_id = kwargs["recipe_id"]
        destination = cells_root / cell.cell_id / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifact = destination.parent / "fixture.bin"
        artifact.write_bytes(cell.cell_id.encode())
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        destination.write_text(
            json.dumps(
                {
                    "schema_version": 6,
                    "dataset_layout": "processing_cell_v2",
                    "run_id": run_id,
                    "recipe_id": recipe_id,
                    "source_selection_id": "a" * 64,
                    "cell": {"cell_id": cell.cell_id},
                    "integrity": {
                        "outputs": {
                            "fixture.bin": {"sha256": digest, "bytes": artifact.stat().st_size}
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        return destination

    monkeypatch.setattr(prep, "build_processing_cell_dataset", fake_cell_builder)
    manifest_path = prep.build_processing_cell_collection(
        aoi,
        tmp_path / "out",
        30,
        20.0,
        resolution_m=100.0,
        cell_pixels=128,
        halo_pixels=8,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["dataset_layout"] == "processing_cell_collection_v2"
    assert manifest["scientific_unit"] == "AOI"
    assert manifest["aoi"]["role"] == "inference_only"
    assert manifest["execution"]["built_cells"] == len(manifest["cells"])
    assert manifest["execution"]["reused_cells"] == 0
    assert len(manifest["cells"]) == len(manifest["processing_grid"]["cells"])
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas/processing-cell-collection.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(manifest)


def test_stac_item_set_checks_overlapping_sliding_time_windows(monkeypatch) -> None:
    base = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"

    def item(item_id: str, west: float, east: float, stamp: str) -> dict[str, object]:
        assets = {name: {"href": f"{base}/{item_id}-{name}.tif"} for name in prep.SENTINEL_BANDS}
        return {
            "type": "Feature",
            "id": item_id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[west, 60.0], [east, 60.0], [east, 60.5], [west, 60.5], [west, 60.0]]
                ],
            },
            "properties": {"datetime": stamp, "eo:cloud_cover": 2.0},
            "assets": assets,
        }

    # The 10:00 item is irrelevant. A fixed window anchored only at 10:00 would split 10:40
    # away from 10:20 even though those latter two are a valid <=30-minute coverage set.
    payload = {
        "features": [
            item("irrelevant", 23.0, 23.5, "2021-07-01T10:00:00Z"),
            item("left", 24.0, 24.5, "2021-07-01T10:20:00Z"),
            item("right", 24.5, 25.0, "2021-07-01T10:40:00Z"),
        ]
    }
    monkeypatch.setattr(prep, "bounded_json_get", lambda *_args, **_kwargs: payload)
    from datetime import UTC, datetime

    selected = prep.stac_item_set(
        (24.0, 60.0, 25.0, 60.5),
        30,
        10.0,
        start=datetime(2021, 6, 1, tzinfo=UTC),
        end=datetime(2021, 8, 1, tzinfo=UTC),
        max_time_span_minutes=30,
    )
    assert {row["id"] for row in selected} == {"left", "right"}
