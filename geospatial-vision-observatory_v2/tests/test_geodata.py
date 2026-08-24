import numpy as np

from geo_vision.geodata import (
    CURATED_AOIS,
    assert_single_worldcover_tile,
    hansen_tile_id,
    hansen_url,
    hansen_url_for_tile,
    spectral_indices,
    worldcover_map_url,
    worldcover_map_url_for_tile,
    worldcover_tile_id,
)


def test_worldcover_helsinki_tile() -> None:
    assert worldcover_tile_id(24.95, 60.20) == "N60E024"
    assert worldcover_map_url(24.95, 60.20).endswith(
        "ESA_WorldCover_10m_2021_v200_N60E024_Map.tif"
    )


def test_hansen_helsinki_tile() -> None:
    assert hansen_tile_id(24.95, 60.20) == "70N_020E"
    assert hansen_url("lossyear", 24.95, 60.20).endswith(
        "Hansen_GFC-2025-v1.13_lossyear_70N_020E.tif"
    )


def test_curated_aois_stay_inside_single_worldcover_tile() -> None:
    for aoi in CURATED_AOIS.values():
        assert_single_worldcover_tile(aoi.bbox)


def test_spectral_indices() -> None:
    red = np.array([[1.0, 0.0]], dtype=np.float32)
    green = np.array([[2.0, 1.0]], dtype=np.float32)
    nir = np.array([[3.0, 1.0]], dtype=np.float32)
    swir16 = np.array([[4.0, 1.0]], dtype=np.float32)

    ndvi, ndwi, ndbi = spectral_indices(red, green, nir, swir16)

    assert np.isclose(ndvi[0, 0], 0.5)
    assert np.isclose(ndwi[0, 0], -0.2)
    assert np.isclose(ndbi[0, 0], 1.0 / 7.0)
    assert np.isnan(ndvi[0, 1])


def test_tile_id_url_builders_validate_canonical_ids() -> None:
    assert worldcover_map_url_for_tile("N60E024").endswith("N60E024_Map.tif")
    assert hansen_url_for_tile("lossyear", "70N_020E").endswith("lossyear_70N_020E.tif")
    import pytest

    with pytest.raises(ValueError, match="WorldCover"):
        worldcover_map_url_for_tile("../bad")
    with pytest.raises(ValueError, match="Hansen"):
        hansen_url_for_tile("lossyear", "bad")
