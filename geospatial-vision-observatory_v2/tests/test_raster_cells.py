from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from geo_vision.aoi import aoi_from_bbox
from geo_vision.cells import plan_processing_grid
from geo_vision.raster_cells import cell_aoi_mask, cell_profile, reproject_sources


def _write(path: Path, value: int, bounds: tuple[float, float, float, float]) -> None:
    data = np.full((64, 64), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, width=64, height=64),
        nodata=255,
    ) as dst:
        dst.write(data, 1)


def test_reproject_sources_assembles_adjacent_tiles_deterministically(tmp_path: Path) -> None:
    left = tmp_path / "z-left.tif"
    right = tmp_path / "a-right.tif"
    _write(left, 10, (24.0, 60.0, 24.5, 60.5))
    _write(right, 50, (24.5, 60.0, 25.0, 60.5))
    profile = {
        "crs": "EPSG:4326",
        "transform": from_bounds(24.0, 60.0, 25.0, 60.5, width=128, height=64),
        "width": 128,
        "height": 64,
    }
    first = reproject_sources(
        [str(left), str(right)],
        profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    second = reproject_sources(
        [str(right), str(left), str(right)],
        profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    assert np.array_equal(first, second)
    assert np.all(first[:, :64] == 10)
    assert np.all(first[:, 64:] == 50)


def test_reproject_sources_preserves_valid_zero_values(tmp_path: Path) -> None:
    source = tmp_path / "zero.tif"
    _write(source, 0, (24.0, 60.0, 25.0, 61.0))
    profile = {
        "crs": "EPSG:4326",
        "transform": from_bounds(24.0, 60.0, 25.0, 61.0, width=32, height=32),
        "width": 32,
        "height": 32,
    }
    result = reproject_sources(
        [str(source)],
        profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=255,
    )
    assert np.all(result == 0)


def test_cell_profile_has_exact_resolution_and_halo_dimensions() -> None:
    aoi = aoi_from_bbox("profile", (24.8, 60.1, 24.9, 60.2))
    grid = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=128, halo_pixels=16)
    cell = grid.cells[0]
    core = cell_profile(grid, cell, dtype="uint8", nodata=0)
    halo = cell_profile(grid, cell, dtype="uint8", nodata=0, include_halo=True)
    assert core["width"] == core["height"] == 128
    assert halo["width"] == halo["height"] == 160
    assert core["transform"].a == 100.0
    assert core["transform"].e == -100.0


def test_cell_mask_excludes_pixels_outside_irregular_aoi() -> None:
    from shapely.geometry import Polygon, mapping
    from geo_vision.aoi import aoi_from_feature

    aoi = aoi_from_feature(
        {
            "type": "Feature",
            "id": "triangle-cell",
            "properties": {"role": "inference_only", "name": "triangle-cell"},
            "geometry": mapping(Polygon([(24.8, 60.1), (25.0, 60.1), (24.8, 60.3)])),
        }
    )
    grid = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=128, halo_pixels=0)
    partial = next(cell for cell in grid.cells if cell.coverage_fraction < 0.9)
    profile = cell_profile(grid, partial, dtype="uint8", nodata=0)
    mask = cell_aoi_mask(aoi, grid, partial, profile)
    assert 0 < int(mask.sum()) < mask.size


def test_reproject_calibrated_sources_merges_tiles_with_per_source_calibration(tmp_path: Path) -> None:
    from geo_vision.raster_cells import reproject_calibrated_sources

    left = tmp_path / "left-reflectance.tif"
    right = tmp_path / "right-reflectance.tif"
    _write(left, 20, (24.0, 60.0, 24.5, 60.5))
    _write(right, 40, (24.5, 60.0, 25.0, 60.5))
    profile = {
        "crs": "EPSG:4326",
        "transform": from_bounds(24.0, 60.0, 25.0, 60.5, width=128, height=64),
        "width": 128,
        "height": 64,
    }
    result = reproject_calibrated_sources(
        [
            (str(left), 0.01, -0.1, 255.0),
            (str(right), 0.02, 0.0, 255.0),
        ],
        profile,
    )
    assert np.allclose(result[:, :64], 0.1)
    assert np.allclose(result[:, 64:], 0.8)


def test_reproject_sources_exposes_coverage_separately_from_valid_zero(tmp_path: Path) -> None:
    from geo_vision.raster_cells import reproject_sources_with_coverage

    source = tmp_path / "half-zero.tif"
    _write(source, 0, (24.0, 60.0, 24.5, 61.0))
    profile = {
        "crs": "EPSG:4326",
        "transform": from_bounds(24.0, 60.0, 25.0, 61.0, width=64, height=32),
        "width": 64,
        "height": 32,
    }
    result, coverage = reproject_sources_with_coverage(
        [str(source)],
        profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    assert np.all(result == 0)
    assert coverage[:, :32].all()
    assert not coverage[:, 32:].any()
