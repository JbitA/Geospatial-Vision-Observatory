#!/usr/bin/env python3
"""Build a compact Sentinel-2 + WorldCover + Hansen urban/forestry dataset.

The command reads Cloud-Optimized GeoTIFF windows rather than downloading global datasets.
It creates a six-band Sentinel-2 stack, spectral indices, WorldCover labels and Hansen forestry
layers on one common grid, plus a preview and a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image

try:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError as error:  # pragma: no cover - runtime guard
    raise SystemExit("Install the geospatial extra: pip install -e '.[geo]'") from error

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from geo_vision.geodata import (  # noqa: E402
    CURATED_AOIS,
    WORLDCOVER_CLASSES,
    assert_single_worldcover_tile,
    bbox_center,
    hansen_url,
    spectral_indices,
    worldcover_map_url,
)

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ALLOWED_ASSET_HOSTS = {
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
}
MAX_STAC_BYTES = 8 * 1024 * 1024
STAC_ITEM_LIMIT = 200
MAX_SCL_QUALITY_CANDIDATES = 24
SENTINEL_BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")
SCL_EXCLUDED_CLASSES = frozenset({0, 1, 3, 7, 8, 9, 10})


def _validate_https_url(value: object, allowed_hosts: set[str], *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} URL is missing")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"{label} URL violates HTTPS/host policy")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} URL must not contain userinfo")
    if parsed.port not in (None, 443):
        raise ValueError(f"{label} URL must use the default HTTPS port")
    if parsed.fragment:
        raise ValueError(f"{label} URL must not contain a fragment")
    return value


def validated_asset_href(value: object) -> str:
    return _validate_https_url(value, ALLOWED_ASSET_HOSTS, label="Sentinel asset")


def bounded_json_get(url: str, params: dict[str, str]) -> dict[str, Any]:
    _validate_https_url(url, {"earth-search.aws.element84.com"}, label="STAC endpoint")
    headers = {
        "Accept": "application/geo+json,application/json",
        "User-Agent": "geospatial-vision-observatory/1.0 dataset-builder",
    }
    chunks: list[bytes] = []
    received = 0
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        with client.stream("GET", url, params=params, headers=headers) as response:
            if 300 <= response.status_code < 400:
                raise RuntimeError("STAC redirects are rejected")
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type not in {"application/json", "application/geo+json"}:
                raise RuntimeError("STAC response has an unexpected media type")
            declared = response.headers.get("content-length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as error:
                    raise RuntimeError("STAC response has invalid Content-Length") from error
                if declared_bytes < 0:
                    raise RuntimeError("STAC response has invalid Content-Length")
                if declared_bytes > MAX_STAC_BYTES:
                    raise RuntimeError("STAC metadata exceeds the size ceiling")
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > MAX_STAC_BYTES:
                    raise RuntimeError("STAC metadata exceeds the size ceiling")
                chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except json.JSONDecodeError as error:
        raise RuntimeError("STAC response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("STAC response must be a JSON object")
    return payload


def scl_obscured_percent(
    href: str,
    bbox: tuple[float, float, float, float],
    *,
    max_pixels: int = 512,
) -> float:
    """Measure SCL-obscured pixels over the requested AOI, not the whole granule."""

    if not 64 <= max_pixels <= 2048:
        raise ValueError("SCL quality max_pixels must be between 64 and 2048")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("Sentinel SCL COG is missing a CRS")
            projected = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = (
                window_from_bounds(*projected, transform=src.transform)
                .round_offsets()
                .round_lengths()
            )
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = max(width / max_pixels, height / max_pixels, 1.0)
            out_width = max(1, int(round(width / scale)))
            out_height = max(1, int(round(height / scale)))
            scl = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
            )
    if scl.size == 0:
        raise RuntimeError("Sentinel SCL AOI window is empty")
    obscured = np.isin(scl, list(SCL_EXCLUDED_CLASSES))
    return round(float(obscured.mean() * 100.0), 6)


def stac_item(
    bbox: tuple[float, float, float, float],
    lookback_days: int,
    max_cloud: float,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
    quality_max_pixels: int = 512,
) -> dict[str, Any]:
    if not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    now = datetime.now(UTC)
    end = end or now
    start = start or (end - timedelta(days=lookback_days))
    for label, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"Sentinel search {label} must be timezone-aware")
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    if start >= end:
        raise ValueError("Sentinel search start must be before end")
    params = {
        "bbox": ",".join(str(value) for value in bbox),
        "datetime": (
            f"{start.isoformat().replace('+00:00', 'Z')}/"
            f"{end.isoformat().replace('+00:00', 'Z')}"
        ),
        # Earth Search documents sorting on item properties. Asking for a bounded, larger page
        # prevents a long seasonal window from accidentally considering only the newest 50 items.
        "limit": str(STAC_ITEM_LIMIT),
        "sortby": "+properties.eo:cloud_cover,+properties.datetime",
    }
    payload = bounded_json_get(EARTH_SEARCH, params)
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise RuntimeError("STAC response has an invalid features collection")
    candidates: list[tuple[datetime, float, dict[str, Any]]] = []
    for item in features:
        if not isinstance(item, dict):
            continue
        properties = item.get("properties", {})
        assets = item.get("assets", {})
        if not isinstance(properties, dict) or not isinstance(assets, dict):
            continue
        required = {name: assets.get(name, {}) for name in SENTINEL_BANDS}
        if any(not isinstance(asset, dict) for asset in required.values()):
            continue
        try:
            for asset in required.values():
                validated_asset_href(asset.get("href"))
        except ValueError:
            continue
        if not isinstance(item.get("id"), str) or not item["id"]:
            continue
        stamp = properties.get("datetime")
        if not isinstance(stamp, str):
            continue
        try:
            acquired = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if acquired.tzinfo is None or acquired.utcoffset() is None:
                continue
            acquired = acquired.astimezone(UTC)
            cloud = float(properties.get("eo:cloud_cover", 100.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(cloud) or not 0.0 <= cloud <= 100.0:
            continue
        if not start <= acquired <= end:
            continue
        candidates.append((acquired, cloud, item))
    if not candidates:
        raise RuntimeError("Earth Search returned no usable Sentinel-2 multispectral COGs")

    midpoint = start + (end - start) / 2
    candidates.sort(
        key=lambda row: (
            row[1],
            abs((row[0] - midpoint).total_seconds()),
            -row[0].timestamp(),
        )
    )
    if not require_cloud_threshold:
        acceptable = [row for row in candidates if row[1] <= max_cloud]
        return (acceptable or candidates)[0][2]

    # eo:cloud_cover describes the STAC item/granule. For a small AOI it is only a discovery
    # heuristic; publication quality is measured from the Sentinel Scene Classification Layer
    # over the actual AOI. This permits a clear AOI inside an otherwise cloudy granule without
    # weakening the 15% quality gate.
    quality_rows: list[tuple[float, float, float, dict[str, Any]]] = []
    inspected = 0
    for acquired, scene_cloud, item in candidates[:MAX_SCL_QUALITY_CANDIDATES]:
        assets = item.get("assets", {})
        if not isinstance(assets, dict):
            continue
        href = scl_asset_href(assets)
        if href is None:
            continue
        inspected += 1
        try:
            obscured = scl_obscured_percent(
                href,
                bbox,
                max_pixels=min(max(quality_max_pixels, 64), 2048),
            )
        except (OSError, RuntimeError, rasterio.errors.RasterioError):
            continue
        quality_rows.append(
            (
                obscured,
                scene_cloud,
                abs((acquired - midpoint).total_seconds()),
                item,
            )
        )
    if not quality_rows:
        raise RuntimeError(
            "no Sentinel-2 candidate exposed a readable SCL layer for AOI cloud-quality checks "
            f"(inspected {inspected} candidates)"
        )
    acceptable_quality = [row for row in quality_rows if row[0] <= max_cloud]
    if not acceptable_quality:
        best = min(quality_rows, key=lambda row: (row[0], row[1], row[2]))
        raise RuntimeError(
            "no Sentinel-2 scene meets the required AOI SCL obscured-pixel threshold "
            f"({max_cloud}%); best AOI obscured fraction was {best[0]}% "
            f"with scene-wide eo:cloud_cover={best[1]}%"
        )
    selected = min(acceptable_quality, key=lambda row: (row[0], row[1], row[2]))
    chosen = dict(selected[3])
    properties = dict(chosen.get("properties", {}))
    properties["gvo:aoi_scl_obscured_percent"] = selected[0]
    chosen["properties"] = properties
    return chosen



def asset_calibration(asset: dict[str, Any]) -> dict[str, float | int | None]:
    """Extract STAC raster scale, offset and nodata without assuming processing baseline."""

    raster_bands = asset.get("raster:bands")
    metadata = raster_bands[0] if isinstance(raster_bands, list) and raster_bands else {}
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        scale = float(metadata.get("scale", 1.0))
        offset = float(metadata.get("offset", 0.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Sentinel raster calibration is not numeric") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Sentinel raster scale must be finite and positive")
    if not math.isfinite(offset):
        raise ValueError("Sentinel raster offset must be finite")
    nodata_value = metadata.get("nodata", 0)
    nodata: float | None = None
    if isinstance(nodata_value, int | float) and not isinstance(nodata_value, bool):
        nodata = float(nodata_value)
        if not math.isfinite(nodata):
            raise ValueError("Sentinel raster nodata must be finite")
    return {"scale": scale, "offset": offset, "nodata": nodata}


def calibrated_reflectance(array: np.ndarray, asset: dict[str, Any]) -> np.ndarray:
    calibration = asset_calibration(asset)
    result = array.astype(np.float32) * float(calibration["scale"]) + float(
        calibration["offset"]
    )
    nodata = calibration["nodata"]
    if isinstance(nodata, int | float):
        result[array == nodata] = np.nan
    return result


def scl_asset_href(assets: dict[str, Any]) -> str | None:
    for key in ("scl", "SCL"):
        asset = assets.get(key)
        if isinstance(asset, dict) and isinstance(asset.get("href"), str):
            try:
                return validated_asset_href(asset["href"])
            except ValueError:
                continue
    return None


def read_reference_window(
    href: str,
    bbox: tuple[float, float, float, float],
    max_pixels: int = 1536,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read band 1 from a COG and return it with an AOI-scoped target profile."""

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("source COG is missing a CRS")
            projected = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = (
                window_from_bounds(*projected, transform=src.transform)
                .round_offsets()
                .round_lengths()
            )
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = max(width / max_pixels, height / max_pixels, 1.0)
            out_width = max(1, int(round(width / scale)))
            out_height = max(1, int(round(height / scale)))
            array = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.bilinear,
            )
            transform = src.window_transform(window) * Affine.scale(
                window.width / out_width, window.height / out_height
            )
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=out_width,
                height=out_height,
                count=1,
                transform=transform,
                compress="deflate",
                tiled=True,
            )
            return array, profile


def write_tif(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    descriptions: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=array.shape[0])
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array)
        if descriptions:
            if len(descriptions) != array.shape[0]:
                raise ValueError("band descriptions do not match output count")
            for index, description in enumerate(descriptions, start=1):
                dst.set_band_description(index, description)


def reproject_single_band(
    href: str,
    destination_profile: dict[str, Any],
    *,
    dtype: np.dtype[Any],
    resampling: Resampling,
) -> np.ndarray:
    destination = np.zeros(
        (destination_profile["height"], destination_profile["width"]), dtype=dtype
    )
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("source COG is missing a CRS")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=destination_profile["transform"],
                dst_crs=destination_profile["crs"],
                dst_nodata=0,
                resampling=resampling,
            )
    return destination


def robust_rgb_preview(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack((red, green, blue)).astype(np.float32)
    preview = np.zeros_like(rgb, dtype=np.uint8)
    for index in range(3):
        band = rgb[index]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            continue
        low, high = np.quantile(valid, [0.02, 0.98])
        stretched = np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)
        stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
        preview[index] = (stretched * 255).astype(np.uint8)
    return np.transpose(preview, (1, 2, 0))


def class_stats(labels: np.ndarray) -> dict[str, float]:
    valid = labels != 0
    denominator = int(valid.sum())
    if denominator == 0:
        return {}
    return {
        name: round(float((labels == code).sum() / denominator), 6)
        for code, name in WORLDCOVER_CLASSES.items()
        if np.any(labels == code)
    }


def index_stats(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {}
    return {
        "mean": round(float(valid.mean()), 6),
        "median": round(float(np.median(valid)), 6),
        "p05": round(float(np.quantile(valid, 0.05)), 6),
        "p95": round(float(np.quantile(valid, 0.95)), 6),
    }


def build_dataset(
    aoi_key: str,
    output_root: Path,
    lookback_days: int,
    max_cloud: float,
    *,
    max_pixels: int = 1024,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
) -> Path:
    if not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    if not 256 <= max_pixels <= 2048:
        raise ValueError("max_pixels must be between 256 and 2048")
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    aoi = CURATED_AOIS[aoi_key]
    assert_single_worldcover_tile(aoi.bbox)
    item = stac_item(
        aoi.bbox,
        lookback_days,
        max_cloud,
        start=start,
        end=end,
        require_cloud_threshold=require_cloud_threshold,
        quality_max_pixels=min(max_pixels, 512),
    )
    assets = item["assets"]
    if not isinstance(assets, dict):
        raise RuntimeError("selected STAC item has no valid asset map")
    band_hrefs = {name: validated_asset_href(assets[name]["href"]) for name in SENTINEL_BANDS}

    red, profile = read_reference_window(band_hrefs["red"], aoi.bbox, max_pixels=max_pixels)
    profile.update(dtype=str(red.dtype), nodata=0)
    raw_bands: dict[str, np.ndarray] = {"red": red}
    for name in SENTINEL_BANDS:
        if name == "red":
            continue
        raw_bands[name] = reproject_single_band(
            band_hrefs[name],
            profile,
            dtype=red.dtype,
            resampling=Resampling.bilinear,
        )
    reflectance = {
        name: calibrated_reflectance(raw_bands[name], assets[name])
        for name in SENTINEL_BANDS
    }

    scl_href = scl_asset_href(assets)
    scl: np.ndarray | None = None
    scl_valid_fraction: float | None = None
    scl_obscured: float | None = None
    if scl_href is not None:
        scl = reproject_single_band(
            scl_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
        )
        scl_valid_fraction = float((~np.isin(scl, list(SCL_EXCLUDED_CLASSES))).mean())
        scl_obscured = float((1.0 - scl_valid_fraction) * 100.0)
    if require_cloud_threshold:
        if scl is None or scl_obscured is None:
            raise RuntimeError("strict curation requires a readable Sentinel SCL layer")
        if scl_obscured > max_cloud:
            raise RuntimeError(
                "selected Sentinel scene exceeds the AOI SCL obscured-pixel threshold "
                f"({scl_obscured:.3f}% > {max_cloud:.3f}%)"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    final_destination = output_root / aoi.key
    destination = Path(
        tempfile.mkdtemp(prefix=f".{aoi.key}.tmp-", dir=output_root)
    )
    sentinel_stack = np.stack([reflectance[name] for name in SENTINEL_BANDS])
    sentinel_output = np.where(np.isfinite(sentinel_stack), sentinel_stack, -9999.0).astype(
        np.float32
    )
    sentinel_profile = profile.copy()
    sentinel_profile.update(dtype="float32", nodata=-9999.0)
    sentinel_path = destination / "sentinel2_multispectral.tif"
    write_tif(sentinel_path, sentinel_output, sentinel_profile, SENTINEL_BANDS)
    Image.fromarray(
        robust_rgb_preview(reflectance["red"], reflectance["green"], reflectance["blue"])
    ).save(destination / "sentinel2_preview.png")

    ndvi, ndwi, ndbi = spectral_indices(
        reflectance["red"],
        reflectance["green"],
        reflectance["nir"],
        reflectance["swir16"],
    )
    if scl is not None:
        valid_scl = ~np.isin(scl, list(SCL_EXCLUDED_CLASSES))
        ndvi = np.where(valid_scl, ndvi, np.nan).astype(np.float32)
        ndwi = np.where(valid_scl, ndwi, np.nan).astype(np.float32)
        ndbi = np.where(valid_scl, ndbi, np.nan).astype(np.float32)
    indices = np.stack((ndvi, ndwi, ndbi)).astype(np.float32)
    index_output = np.where(np.isfinite(indices), indices, -9999.0).astype(np.float32)
    index_profile = profile.copy()
    index_profile.update(dtype="float32", nodata=-9999.0)
    write_tif(
        destination / "sentinel2_indices.tif",
        index_output,
        index_profile,
        ("ndvi", "ndwi", "ndbi"),
    )
    if scl is not None:
        scl_profile = profile.copy()
        scl_profile.update(dtype="uint8", nodata=0)
        write_tif(
            destination / "sentinel2_scl.tif",
            scl[None, ...],
            scl_profile,
            ("sentinel2_scene_classification",),
        )

    lon, lat = bbox_center(aoi.bbox)
    wc_href = worldcover_map_url(lon, lat)
    worldcover = reproject_single_band(
        wc_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    label_profile = profile.copy()
    label_profile.update(count=1, dtype="uint8", nodata=0)
    write_tif(
        destination / "worldcover_2021_on_sentinel.tif",
        worldcover[None, ...],
        label_profile,
        ("worldcover_class",),
    )

    tree_href = hansen_url("treecover2000", lon, lat)
    loss_href = hansen_url("lossyear", lon, lat)
    treecover = reproject_single_band(
        tree_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    lossyear = reproject_single_band(
        loss_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    write_tif(
        destination / "hansen_treecover2000_on_sentinel.tif",
        treecover[None, ...],
        label_profile,
        ("treecover2000_percent",),
    )
    write_tif(
        destination / "hansen_lossyear_on_sentinel.tif",
        lossyear[None, ...],
        label_profile,
        ("lossyear",),
    )

    output_files = [
        sentinel_path,
        destination / "sentinel2_indices.tif",
        destination / "sentinel2_preview.png",
        destination / "worldcover_2021_on_sentinel.tif",
        destination / "hansen_treecover2000_on_sentinel.tif",
        destination / "hansen_lossyear_on_sentinel.tif",
    ]
    if scl is not None:
        output_files.append(destination / "sentinel2_scl.tif")
    output_integrity = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_files
    }

    item_properties = item.get("properties", {})
    forest_mask = treecover >= 30
    valid_tree = treecover[treecover <= 100]
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "integrity": {
            "builder": "scripts/prepare_geospatial_dataset.py",
            "max_pixels": max_pixels,
            "outputs": output_integrity,
        },
        "aoi": {
            "key": aoi.key,
            "name": aoi.name,
            "bbox": aoi.bbox,
            "purpose": aoi.purpose,
        },
        "sentinel2": {
            "item_id": item.get("id"),
            "requested_start": start.isoformat() if start is not None else None,
            "requested_end": end.isoformat() if end is not None else None,
            "datetime": item_properties.get("datetime"),
            "cloud_cover": item_properties.get("eo:cloud_cover"),
            "cloud_threshold_required": require_cloud_threshold,
            "cloud_quality_basis": (
                "aoi_scl_obscured_percent" if require_cloud_threshold else "scene_metadata_rank"
            ),
            "aoi_scl_valid_fraction": (
                round(scl_valid_fraction, 6) if scl_valid_fraction is not None else None
            ),
            "aoi_scl_obscured_percent": (
                round(scl_obscured, 6) if scl_obscured is not None else None
            ),
            "collection": "sentinel-2-c1-l2a",
            "selection_policy": (
                "rank a bounded set of low-cloud candidates by STAC metadata, then require the "
                "actual AOI Sentinel SCL obscured-pixel fraction to meet the configured ceiling"
                if require_cloud_threshold
                else "prefer the lowest scene-wide STAC cloud estimate in the declared window"
            ),
            "bands": list(SENTINEL_BANDS),
            "band_cogs": band_hrefs,
            "band_calibration": {
                name: asset_calibration(assets[name]) for name in SENTINEL_BANDS
            },
            "scl_cog": scl_href,
            "scl_excluded_classes": sorted(SCL_EXCLUDED_CLASSES),
            "scl_valid_fraction": (
                round(scl_valid_fraction, 6) if scl_valid_fraction is not None else None
            ),
            "indices": {
                "ndvi": index_stats(ndvi),
                "ndwi": index_stats(ndwi),
                "ndbi": index_stats(ndbi),
            },
        },
        "worldcover": {
            "version": "2021 v200",
            "source": wc_href,
            "class_fractions": class_stats(worldcover),
        },
        "hansen": {
            "version": "GFC-2025-v1.13",
            "treecover2000_source": tree_href,
            "lossyear_source": loss_href,
            "mean_treecover2000_percent": (
                round(float(valid_tree.mean()), 4) if valid_tree.size else None
            ),
            "forest_pixels_ge_30pct": int(forest_mask.sum()),
            "loss_pixels_2001_2025": int(((lossyear > 0) & forest_mask).sum()),
        },
        "alignment": {
            "crs": str(profile["crs"]),
            "width": profile["width"],
            "height": profile["height"],
            "transform": list(profile["transform"])[:6],
            "note": (
                "All bands/references were reprojected onto the red-band Sentinel grid. "
                "Continuous Sentinel bands use bilinear resampling; categorical layers "
                "use nearest. "
                "Reflectance uses each STAC asset's raster:bands scale/offset metadata."
            ),
        },
        "scientific_warning": (
            "WorldCover is a reference land-cover product, not perfect ground truth; Hansen v1.13 "
            "has documented temporal-consistency limitations. Use spatial/temporal holdouts."
        ),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    backup = final_destination.with_name(f".{final_destination.name}.previous")
    try:
        if backup.exists():
            shutil.rmtree(backup)
        if final_destination.exists():
            os.replace(final_destination, backup)
        try:
            os.replace(destination, final_destination)
        except BaseException:
            if backup.exists() and not final_destination.exists():
                os.replace(backup, final_destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if destination.exists():
            shutil.rmtree(destination)
    return final_destination / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", choices=sorted(CURATED_AOIS), default="helsinki_metro")
    parser.add_argument("--output", type=Path, default=Path("data/curated"))
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help=(
            "maximum AOI SCL obscured-pixel percentage in strict mode; scene-wide "
            "eo:cloud_cover is used only to rank discovery candidates"
        ),
    )
    parser.add_argument(
        "--require-cloud-threshold",
        action="store_true",
        help="require the AOI SCL obscured-pixel fraction to satisfy --max-cloud",
    )
    parser.add_argument("--max-pixels", type=int, default=1024, choices=range(256, 2049))
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="optional UTC start date (YYYY-MM-DD); use with --end-date",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="optional UTC end date (YYYY-MM-DD); use with --start-date",
    )
    args = parser.parse_args()
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")
    if not 1 <= args.lookback_days <= 3650:
        parser.error("--lookback-days must be between 1 and 3650")
    if not math.isfinite(args.max_cloud) or not 0.0 <= args.max_cloud <= 100.0:
        parser.error("--max-cloud must be a finite percentage between 0 and 100")
    start = (
        datetime.combine(args.start_date, time.min, tzinfo=UTC)
        if args.start_date is not None
        else None
    )
    end = (
        datetime.combine(args.end_date, time.max, tzinfo=UTC)
        if args.end_date is not None
        else None
    )
    path = build_dataset(
        args.aoi,
        args.output,
        args.lookback_days,
        args.max_cloud,
        max_pixels=args.max_pixels,
        start=start,
        end=end,
        require_cloud_threshold=args.require_cloud_threshold,
    )
    print(path)


if __name__ == "__main__":
    main()
