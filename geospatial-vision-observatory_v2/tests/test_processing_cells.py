from __future__ import annotations

import math

import pytest
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shapely_transform

from geo_vision.aoi import ScientificRole, aoi_from_bbox, aoi_from_feature
from geo_vision.cells import cell_core_geometry, plan_processing_grid, select_working_crs
from geo_vision.planning import build_plan


def _feature(aoi_id: str, polygon: Polygon) -> dict:
    from shapely.geometry import mapping

    return {
        "type": "Feature",
        "id": aoi_id,
        "properties": {"name": aoi_id, "role": ScientificRole.INFERENCE_ONLY.value},
        "geometry": mapping(polygon),
    }


def test_single_zone_aoi_uses_utm_and_multi_zone_aoi_uses_laea() -> None:
    helsinki = aoi_from_bbox("helsinki", (24.8, 60.1, 25.2, 60.35))
    assert select_working_crs(helsinki).definition == "EPSG:32635"
    assert select_working_crs(helsinki).strategy == "utm_single_zone"

    cross_zone = aoi_from_bbox("cross-zone", (23.9, 59.8, 24.1, 60.2))
    selected = select_working_crs(cross_zone)
    assert selected.strategy == "laea_aoi_centered"
    assert "+proj=laea" in selected.definition


def test_processing_grid_is_stable_for_equivalent_ring_order() -> None:
    first = aoi_from_feature(
        _feature(
            "same",
            Polygon([(24.0, 60.0), (25.0, 60.0), (25.0, 60.5), (24.0, 60.5)]),
        )
    )
    second = aoi_from_feature(
        _feature(
            "same",
            Polygon([(25.0, 60.5), (25.0, 60.0), (24.0, 60.0), (24.0, 60.5)]),
        )
    )
    left = plan_processing_grid(first, resolution_m=100.0, cell_pixels=512, halo_pixels=16)
    right = plan_processing_grid(second, resolution_m=100.0, cell_pixels=512, halo_pixels=16)
    assert left.as_dict() == right.as_dict()


def test_processing_cell_cores_cover_projected_aoi_without_positive_area_overlap() -> None:
    aoi = aoi_from_feature(
        _feature(
            "irregular",
            Polygon(
                [
                    (24.0, 60.0),
                    (25.1, 60.0),
                    (24.9, 60.55),
                    (24.35, 60.35),
                    (24.0, 60.0),
                ]
            ),
        )
    )
    grid = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=512, halo_pixels=32)
    transformer = Transformer.from_crs("EPSG:4326", grid.working_crs.definition, always_xy=True)
    projected = shapely_transform(transformer.transform, shape(aoi.geometry))
    cell_area = grid.cell_size_m * grid.cell_size_m
    covered_area = sum(cell.coverage_fraction * cell_area for cell in grid.cells)
    assert math.isclose(covered_area, projected.area, rel_tol=0.0, abs_tol=2.0)

    from shapely.geometry import box

    cores = [box(*cell.core_bounds_m) for cell in grid.cells]
    for index, left in enumerate(cores):
        for right in cores[index + 1 :]:
            assert left.intersection(right).area == 0.0


def test_grid_is_origin_anchored_and_halo_does_not_change_core_identity() -> None:
    aoi = aoi_from_bbox("origin-grid", (24.8, 60.1, 25.2, 60.35))
    without_halo = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=512, halo_pixels=0)
    with_halo = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=512, halo_pixels=16)
    assert [(cell.row, cell.column) for cell in without_halo.cells] == [
        (cell.row, cell.column) for cell in with_halo.cells
    ]
    for cell in with_halo.cells:
        left, bottom, right, top = cell.core_bounds_m
        assert math.isclose(left / with_halo.cell_size_m, cell.column)
        assert math.isclose(bottom / with_halo.cell_size_m, cell.row)
        assert math.isclose(right - left, with_halo.cell_size_m)
        assert math.isclose(top - bottom, with_halo.cell_size_m)
        halo_left, halo_bottom, halo_right, halo_top = cell.halo_bounds_m
        assert math.isclose(left - halo_left, 1600.0)
        assert math.isclose(bottom - halo_bottom, 1600.0)
        assert math.isclose(halo_right - right, 1600.0)
        assert math.isclose(halo_top - top, 1600.0)
    # Halo is part of an artifact/grid recipe, so changing it deliberately changes cell identity.
    assert {cell.cell_id for cell in without_halo.cells}.isdisjoint(
        {cell.cell_id for cell in with_halo.cells}
    )


def test_cell_exact_geometry_is_inside_aoi_and_bbox_matches_contract() -> None:
    polygon = Polygon(
        [(23.9, 59.9), (24.2, 59.9), (24.15, 60.2), (23.95, 60.15), (23.9, 59.9)]
    )
    aoi = aoi_from_feature(_feature("clip", polygon))
    grid = plan_processing_grid(aoi, resolution_m=100.0, cell_pixels=512, halo_pixels=16)
    for cell in grid.cells:
        clipped = shape(cell_core_geometry(aoi, grid, cell))
        assert clipped.area > 0.0
        assert polygon.covers(clipped.representative_point())
        minx, miny, maxx, maxy = clipped.bounds
        west, south, east, north = cell.wgs84_bbox
        assert math.isclose(minx, west, abs_tol=1e-9)
        assert math.isclose(miny, south, abs_tol=1e-9)
        assert math.isclose(maxx, east, abs_tol=1e-9)
        assert math.isclose(maxy, north, abs_tol=1e-9)


def test_cross_reference_boundary_cells_enumerate_multiple_tiles() -> None:
    aoi = aoi_from_bbox("boundaries", (19.8, 59.8, 24.2, 60.2))
    grid = plan_processing_grid(aoi, resolution_m=500.0, cell_pixels=512, halo_pixels=0)
    assert any(len(cell.worldcover_tiles) > 1 for cell in grid.cells)
    assert any(len(cell.hansen_tiles) > 1 for cell in grid.cells)


def test_plan_identity_changes_with_grid_contract_and_is_order_independent() -> None:
    a = aoi_from_bbox("a", (24.0, 60.0, 24.2, 60.2))
    b = aoi_from_bbox("b", (25.0, 60.0, 25.2, 60.2))
    first = build_plan([a, b], cell_pixels=512, halo_pixels=16)
    second = build_plan([b, a], cell_pixels=512, halo_pixels=16)
    changed = build_plan([a, b], cell_pixels=1024, halo_pixels=16)
    assert first["plan_id"] == second["plan_id"]
    assert first["plan_id"] != changed["plan_id"]
    assert first["schema_version"] == "2.0"
    assert first["aois"][0]["processing_grid"]["cells"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resolution_m": 0.0}, "resolution_m"),
        ({"cell_pixels": 63}, "cell_pixels"),
        ({"halo_pixels": -1}, "halo_pixels"),
        ({"cell_pixels": 128, "halo_pixels": 64}, "half"),
        ({"max_cells": 0}, "max_cells"),
    ],
)
def test_processing_grid_rejects_unsafe_parameters(kwargs: dict[str, object], message: str) -> None:
    aoi = aoi_from_bbox("bad-grid", (24.0, 60.0, 24.1, 60.1))
    with pytest.raises(ValueError, match=message):
        plan_processing_grid(aoi, **kwargs)  # type: ignore[arg-type]


def test_iteration_2_json_schemas_are_valid() -> None:
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator

    root = Path(__file__).parents[1]
    for name in (
        "aoi-plan.schema.json",
        "processing-cell.schema.json",
        "processing-cell-collection.schema.json",
    ):
        Draft202012Validator.check_schema(json.loads((root / "schemas" / name).read_text()))
