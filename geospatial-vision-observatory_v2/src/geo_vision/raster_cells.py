from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom

from .aoi import AOI
from .cells import ProcessingCell, ProcessingGrid, cell_core_geometry


def cell_profile(
    grid: ProcessingGrid,
    cell: ProcessingCell,
    *,
    dtype: str,
    nodata: int | float,
    include_halo: bool = False,
) -> dict[str, Any]:
    bounds = cell.halo_bounds_m if include_halo else cell.core_bounds_m
    left, _bottom, _right, top = bounds
    padding = grid.halo_pixels if include_halo else 0
    width = grid.cell_pixels + (2 * padding)
    height = grid.cell_pixels + (2 * padding)
    transform = Affine(grid.resolution_m, 0.0, left, 0.0, -grid.resolution_m, top)
    return {
        "driver": "GTiff",
        "crs": grid.working_crs.definition,
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }


def reproject_sources_with_coverage(
    hrefs: Iterable[str],
    destination_profile: dict[str, Any],
    *,
    dtype: np.dtype[Any],
    resampling: Resampling,
    destination_nodata: int | float = 0,
) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """Reproject a logical layer and return both values and explicit source coverage."""

    ordered = tuple(sorted(set(hrefs)))
    if not ordered:
        raise ValueError("at least one raster source is required")
    destination = np.full(
        (int(destination_profile["height"]), int(destination_profile["width"])),
        destination_nodata,
        dtype=dtype,
    )
    covered = np.zeros(destination.shape, dtype=bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for href in ordered:
            with rasterio.open(href) as src:
                if src.crs is None:
                    raise RuntimeError("source COG is missing a CRS")
                temporary = np.full(destination.shape, np.nan, dtype=np.float64)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=temporary,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=destination_profile["transform"],
                    dst_crs=destination_profile["crs"],
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
                valid = np.isfinite(temporary) & ~covered
                destination[valid] = temporary[valid].astype(dtype, copy=False)
                covered[valid] = True
    return destination, covered


def reproject_sources(
    hrefs: Iterable[str],
    destination_profile: dict[str, Any],
    *,
    dtype: np.dtype[Any],
    resampling: Resampling,
    destination_nodata: int | float = 0,
) -> npt.NDArray[Any]:
    """Reproject one logical layer from one or more deterministic source rasters."""

    destination, _covered = reproject_sources_with_coverage(
        hrefs,
        destination_profile,
        dtype=dtype,
        resampling=resampling,
        destination_nodata=destination_nodata,
    )
    return destination


def reproject_calibrated_sources(
    sources: Iterable[tuple[str, float, float, float | None]],
    destination_profile: dict[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
) -> npt.NDArray[np.float32]:
    """Warp prioritized calibrated sources and fill each target pixel from the first valid source."""

    ordered: list[tuple[str, float, float, float | None]] = []
    seen: set[str] = set()
    for source in sources:
        href, scale, offset, nodata = source
        if href in seen:
            continue
        if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(offset):
            raise ValueError("source calibration must contain finite positive scale and finite offset")
        if nodata is not None and not np.isfinite(nodata):
            raise ValueError("source nodata must be finite when provided")
        seen.add(href)
        ordered.append(source)
    if not ordered:
        raise ValueError("at least one calibrated raster source is required")

    shape = (int(destination_profile["height"]), int(destination_profile["width"]))
    destination = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for href, scale, offset, nodata in ordered:
            with rasterio.open(href) as src:
                if src.crs is None:
                    raise RuntimeError("source COG is missing a CRS")
                temporary = np.full(shape, np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=temporary,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=nodata if nodata is not None else src.nodata,
                    dst_transform=destination_profile["transform"],
                    dst_crs=destination_profile["crs"],
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
                temporary = temporary * float(scale) + float(offset)
                valid = np.isfinite(temporary) & ~np.isfinite(destination)
                destination[valid] = temporary[valid]
    return destination


def cell_aoi_mask(aoi: AOI, grid: ProcessingGrid, cell: ProcessingCell, profile: dict[str, Any]) -> npt.NDArray[np.bool_]:
    geometry = cell_core_geometry(aoi, grid, cell)
    projected = transform_geom("EPSG:4326", profile["crs"], geometry)
    mask = geometry_mask(
        [projected],
        out_shape=(int(profile["height"]), int(profile["width"])),
        transform=profile["transform"],
        invert=True,
    )
    if not mask.any():
        raise RuntimeError("processing cell AOI intersection has no pixels on its target grid")
    return mask
