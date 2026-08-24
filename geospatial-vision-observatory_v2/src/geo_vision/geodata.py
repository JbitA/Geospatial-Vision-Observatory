from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

_WORLDCOVER_TILE = re.compile(r"^[NS]\d{2}[EW]\d{3}$")
_HANSEN_TILE = re.compile(r"^\d{2}[NS]_\d{3}[EW]$")

WORLDCOVER_CLASSES: dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


@dataclass(frozen=True)
class CuratedAOI:
    key: str
    name: str
    bbox: tuple[float, float, float, float]
    purpose: str


CURATED_AOIS: dict[str, CuratedAOI] = {
    "helsinki_metro": CuratedAOI(
        key="helsinki_metro",
        name="Helsinki metropolitan urban-forest interface",
        bbox=(24.80, 60.10, 25.20, 60.35),
        purpose="urban land-cover segmentation, green-space mapping, temporal change",
    ),
    "tampere_growth": CuratedAOI(
        key="tampere_growth",
        name="Tampere urban and peri-urban landscape",
        bbox=(23.55, 61.40, 23.95, 61.62),
        purpose="built-up segmentation and urban expansion experiments",
    ),
    "north_karelia_forest": CuratedAOI(
        key="north_karelia_forest",
        name="North Karelia managed forest landscape",
        bbox=(29.10, 62.55, 29.70, 62.95),
        purpose="forest-cover segmentation and disturbance/loss benchmarking",
    ),
    "turku_coast": CuratedAOI(
        key="turku_coast",
        name="Turku coastal urban-agricultural landscape",
        bbox=(22.10, 60.35, 22.46, 60.58),
        purpose="urban, cropland, coastal-water and green-space segmentation",
    ),
    "oulu_mixed": CuratedAOI(
        key="oulu_mixed",
        name="Oulu boreal urban-wetland landscape",
        bbox=(25.25, 64.90, 25.70, 65.15),
        purpose="boreal forest, built-up, wetland and water segmentation",
    ),
    "jyvaskyla_validation": CuratedAOI(
        key="jyvaskyla_validation",
        name="Jyväskylä inland validation landscape",
        bbox=(25.55, 62.12, 25.95, 62.38),
        purpose="spatially isolated validation across forest, urban and inland-water classes",
    ),
    "stockholm_external": CuratedAOI(
        key="stockholm_external",
        name="Stockholm external urban-forest holdout",
        bbox=(18.04, 59.20, 18.36, 59.47),
        purpose="geographically external land-cover generalization and urban-water evaluation",
    ),
    "tallinn_external": CuratedAOI(
        key="tallinn_external",
        name="Tallinn external coastal urban holdout",
        bbox=(24.55, 59.31, 24.96, 59.55),
        purpose="geographically external built-up, vegetation, wetland and coastal evaluation",
    ),
}


def _hemisphere(value: int, positive: str, negative: str, width: int) -> str:
    prefix = positive if value >= 0 else negative
    return f"{prefix}{abs(value):0{width}d}"


def worldcover_tile_origin(lon: float, lat: float) -> tuple[int, int]:
    """Return the 3-degree lower-left tile origin containing lon/lat."""

    lon0 = math.floor(lon / 3.0) * 3
    lat0 = math.floor(lat / 3.0) * 3
    return lon0, lat0


def worldcover_tile_id(lon: float, lat: float) -> str:
    lon0, lat0 = worldcover_tile_origin(lon, lat)
    return f"{_hemisphere(lat0, 'N', 'S', 2)}{_hemisphere(lon0, 'E', 'W', 3)}"




def worldcover_tile_ids_for_bbox(bbox: tuple[float, float, float, float]) -> set[str]:
    """Return every 3-degree WorldCover tile intersecting a WGS84 bbox area."""

    west, south, east, north = bbox
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError("bbox must be an ordered WGS84 extent")
    lon_start = math.floor(west / 3.0) * 3
    lon_end = math.floor(math.nextafter(east, west) / 3.0) * 3
    lat_start = math.floor(south / 3.0) * 3
    lat_end = math.floor(math.nextafter(north, south) / 3.0) * 3
    return {
        f"{_hemisphere(lat0, 'N', 'S', 2)}{_hemisphere(lon0, 'E', 'W', 3)}"
        for lat0 in range(lat_start, lat_end + 1, 3)
        for lon0 in range(lon_start, lon_end + 1, 3)
    }


def hansen_tile_ids_for_bbox(bbox: tuple[float, float, float, float]) -> set[str]:
    """Return every Hansen 10-degree tile intersecting a WGS84 bbox area."""

    west, south, east, north = bbox
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError("bbox must be an ordered WGS84 extent")
    lon_start = math.floor(west / 10.0) * 10
    lon_end = math.floor(math.nextafter(east, west) / 10.0) * 10
    # Hansen tile ids use the top edge. Area immediately north of an exact multiple belongs
    # to the tile whose top edge is +10 degrees.
    first_top = math.floor(south / 10.0) * 10 + 10
    last_top = math.ceil(north / 10.0) * 10
    return {
        f"{abs(lat_top):02d}{'N' if lat_top >= 0 else 'S'}_{abs(lon_left):03d}{'E' if lon_left >= 0 else 'W'}"
        for lat_top in range(first_top, last_top + 1, 10)
        for lon_left in range(lon_start, lon_end + 1, 10)
    }

def worldcover_map_url_for_tile(tile: str) -> str:
    if not _WORLDCOVER_TILE.fullmatch(tile):
        raise ValueError("invalid WorldCover tile id")
    return (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    )


def worldcover_map_url(lon: float, lat: float) -> str:
    return worldcover_map_url_for_tile(worldcover_tile_id(lon, lat))


def hansen_tile_origin(lon: float, lat: float) -> tuple[int, int]:
    """Return Hansen 10-degree tile top-left coordinates for a point."""

    lon_left = math.floor(lon / 10.0) * 10
    lat_top = math.ceil(lat / 10.0) * 10
    # Exact multiples belong to the tile immediately north only when the point is
    # numerically above the edge; this keeps conventional 60.1N -> 70N behavior.
    if math.isclose(lat, float(lat_top), abs_tol=1e-12):
        lat_top = int(lat)
    return lon_left, lat_top


def hansen_tile_id(lon: float, lat: float) -> str:
    lon_left, lat_top = hansen_tile_origin(lon, lat)
    lat_token = f"{abs(lat_top):02d}{'N' if lat_top >= 0 else 'S'}"
    lon_token = f"{abs(lon_left):03d}{'E' if lon_left >= 0 else 'W'}"
    return f"{lat_token}_{lon_token}"


def hansen_url_for_tile(layer: str, tile: str) -> str:
    if layer not in {"treecover2000", "lossyear", "gain", "datamask", "first", "last"}:
        raise ValueError("unsupported Hansen layer")
    if not _HANSEN_TILE.fullmatch(tile):
        raise ValueError("invalid Hansen tile id")
    return (
        "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/"
        f"Hansen_GFC-2025-v1.13_{layer}_{tile}.tif"
    )


def hansen_url(layer: str, lon: float, lat: float) -> str:
    return hansen_url_for_tile(layer, hansen_tile_id(lon, lat))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    west, south, east, north = bbox
    return ((west + east) / 2.0, (south + north) / 2.0)


def assert_single_worldcover_tile(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    ids = {
        worldcover_tile_id(west, south),
        worldcover_tile_id(west, north - 1e-9),
        worldcover_tile_id(east - 1e-9, south),
        worldcover_tile_id(east - 1e-9, north - 1e-9),
    }
    if len(ids) != 1:
        raise ValueError("curated AOI crosses a WorldCover 3-degree tile boundary")


def spectral_indices(
    red: npt.NDArray[np.floating[Any] | np.integer[Any]],
    green: npt.NDArray[np.floating[Any] | np.integer[Any]],
    nir: npt.NDArray[np.floating[Any] | np.integer[Any]],
    swir16: npt.NDArray[np.floating[Any] | np.integer[Any]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Return NDVI, NDWI and NDBI with NaN for invalid/zero-denominator pixels."""

    redf = red.astype(np.float32)
    greenf = green.astype(np.float32)
    nirf = nir.astype(np.float32)
    swirf = swir16.astype(np.float32)

    def normalized(
        left: npt.NDArray[np.float32],
        right: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        denominator = left + right
        out = np.full(left.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-6) & (left > 0) & (right > 0)
        out[valid] = (left[valid] - right[valid]) / denominator[valid]
        return out

    ndvi = normalized(nirf, redf)
    ndwi = normalized(greenf, nirf)
    ndbi = normalized(swirf, nirf)
    return ndvi, ndwi, ndbi
