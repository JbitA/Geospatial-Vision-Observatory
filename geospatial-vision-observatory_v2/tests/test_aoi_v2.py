from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from shapely.geometry import Polygon, mapping

from geo_vision.aoi import (
    ScientificRole,
    aoi_from_bbox,
    aoi_from_feature,
    canonical_geometry,
    load_aoi_document,
    validate_scientific_isolation,
)
from geo_vision.artifacts import LocalArtifactStore, validate_artifact_key
from geo_vision.geodata import hansen_tile_ids_for_bbox, worldcover_tile_ids_for_bbox
from geo_vision.planning import build_plan, plan_aoi


def feature(aoi_id: str, coordinates: list[list[float]], role: str = "inference_only") -> dict:
    return {
        "type": "Feature",
        "id": aoi_id,
        "properties": {"name": aoi_id, "role": role},
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
    }


def test_canonical_geometry_hash_is_ring_order_stable() -> None:
    first = mapping(Polygon([(24.0, 60.0), (25.0, 60.0), (25.0, 61.0), (24.0, 61.0)]))
    second = mapping(Polygon([(25.0, 61.0), (25.0, 60.0), (24.0, 60.0), (24.0, 61.0)]))
    assert canonical_geometry(first)[1] == canonical_geometry(second)[1]


def test_aoi_rejects_invalid_self_intersection() -> None:
    payload = feature("bad", [[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]])
    with pytest.raises(ValueError, match="invalid"):
        aoi_from_feature(payload)


def test_aoi_rejects_antimeridian_spanning_shape() -> None:
    payload = feature("dateline", [[170, 10], [-170, 10], [-170, 12], [170, 12], [170, 10]])
    with pytest.raises(ValueError, match="antimeridian"):
        aoi_from_feature(payload)


def test_feature_id_can_be_supplied_in_properties() -> None:
    payload = feature("ignored", [[24, 60], [25, 60], [25, 61], [24, 61], [24, 60]])
    payload.pop("id")
    payload["properties"]["id"] = "property-id"
    assert aoi_from_feature(payload).aoi_id == "property-id"


def test_scientific_aois_reject_positive_area_overlap_even_same_role() -> None:
    left = aoi_from_bbox("left", (24, 60, 25, 61), role=ScientificRole.TRAIN)
    right = aoi_from_bbox("right", (24.5, 60.5, 25.5, 61.5), role=ScientificRole.TRAIN)
    with pytest.raises(ValueError, match="positive-area disjoint"):
        validate_scientific_isolation([left, right])


def test_unobserved_holdout_cannot_overlap_observed_benchmark() -> None:
    observed = aoi_from_bbox("observed", (24, 60, 25, 61), role=ScientificRole.EXTERNAL_OBSERVED)
    unobserved = aoi_from_bbox(
        "unobserved", (24.8, 60.8, 25.8, 61.8), role=ScientificRole.EXTERNAL_UNOBSERVED
    )
    with pytest.raises(ValueError, match="positive-area disjoint"):
        validate_scientific_isolation([observed, unobserved])


def test_inference_only_overlap_is_allowed() -> None:
    train = aoi_from_bbox("train", (24, 60, 25, 61), role=ScientificRole.TRAIN)
    inference = aoi_from_bbox("inference", (24.2, 60.2, 24.8, 60.8))
    validate_scientific_isolation([train, inference])


def test_feature_collection_rejects_duplicate_ids() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            feature("same", [[24, 60], [25, 60], [25, 61], [24, 61], [24, 60]]),
            feature("same", [[26, 60], [27, 60], [27, 61], [26, 61], [26, 60]]),
        ],
    }
    with pytest.raises(ValueError, match="duplicate AOI id"):
        load_aoi_document(payload)


def test_reference_tile_enumeration_detects_boundaries() -> None:
    assert worldcover_tile_ids_for_bbox((24.1, 60.1, 25.0, 61.0)) == {"N60E024"}
    assert len(worldcover_tile_ids_for_bbox((23.9, 60.1, 24.1, 61.0))) == 2
    assert hansen_tile_ids_for_bbox((24.1, 60.1, 25.0, 61.0)) == {"70N_020E"}
    assert len(hansen_tile_ids_for_bbox((19.9, 60.1, 20.1, 61.0))) == 2


def test_plan_marks_multi_reference_tile_aoi_blocked_for_legacy_engine() -> None:
    aoi = aoi_from_bbox("wide", (23.9, 59.0, 24.1, 61.0))
    planned = plan_aoi(aoi)
    assert not planned.legacy_executable
    assert planned.blockers


def test_plan_identity_is_independent_of_input_order() -> None:
    a = aoi_from_bbox("a", (24, 60, 25, 61))
    b = aoi_from_bbox("b", (26, 60, 27, 61))
    assert build_plan([a, b])["plan_id"] == build_plan([b, a])["plan_id"]


@pytest.mark.parametrize("key", ["../secret", "/absolute", "C:/windows", "a\\b", "a/../b"])
def test_artifact_key_rejects_path_escape(key: str) -> None:
    with pytest.raises(ValueError):
        validate_artifact_key(key)


def test_local_artifact_store_round_trip_and_hash(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    result = store.put_bytes("runs/abc/result.json", b'{"ok":true}\n')
    assert result.bytes == len(b'{"ok":true}\n')
    assert len(result.sha256) == 64
    assert store.exists("runs/abc/result.json")
    assert store.get_bytes("runs/abc/result.json") == b'{"ok":true}\n'
    assert store.stat("runs/abc/result.json") == result


def test_json_schemas_accept_example_and_emitted_plan() -> None:
    root = Path(__file__).parents[1]
    aoi_schema = json.loads((root / "schemas/aoi.schema.json").read_text())
    plan_schema = json.loads((root / "schemas/aoi-plan.schema.json").read_text())
    Draft202012Validator.check_schema(aoi_schema)
    Draft202012Validator.check_schema(plan_schema)
    example = json.loads((root / "examples/aoi-irregular.geojson").read_text())
    Draft202012Validator(aoi_schema).validate(example)
    plan = build_plan(load_aoi_document(example))
    Draft202012Validator(plan_schema).validate(plan)


def test_baseline_external_aois_are_observed_not_unobserved() -> None:
    from geo_vision.aoi import baseline_aoi
    from geo_vision.geodata import CURATED_AOIS

    assert baseline_aoi("stockholm_external", CURATED_AOIS["stockholm_external"]).role is ScientificRole.EXTERNAL_OBSERVED
    assert baseline_aoi("tallinn_external", CURATED_AOIS["tallinn_external"]).role is ScientificRole.EXTERNAL_OBSERVED


def test_exact_aoi_mask_excludes_bbox_pixels_outside_irregular_polygon() -> None:
    from affine import Affine
    from scripts.prepare_geospatial_dataset import aoi_mask_on_profile

    aoi = aoi_from_feature(
        feature(
            "triangle",
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        )
    )
    profile = {
        "crs": "EPSG:4326",
        "height": 10,
        "width": 10,
        "transform": Affine(0.1, 0.0, 0.0, 0.0, -0.1, 1.0),
    }
    mask = aoi_mask_on_profile(aoi, profile)
    assert mask.shape == (10, 10)
    assert 0 < int(mask.sum()) < 100


def test_schema_4_curated_manifests_remain_supported_by_ml_loader_contract(tmp_path: Path) -> None:
    # This checks the compatibility guard without constructing large GeoTIFF fixtures.
    import inspect
    from geo_vision.ml import data as ml_data

    source = inspect.getsource(ml_data.load_scene)
    assert "{3, 4}" in source


def test_baseline_aoi_geojson_is_present_valid_and_complete() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads((root / "config/aois-baseline.geojson").read_text())
    schema = json.loads((root / "schemas/aoi.schema.json").read_text())
    Draft202012Validator(schema).validate(payload)
    aois = load_aoi_document(payload)
    assert {aoi.aoi_id for aoi in aois} == {
        "helsinki_metro",
        "north_karelia_forest",
        "turku_coast",
        "oulu_mixed",
        "tampere_growth",
        "jyvaskyla_validation",
        "stockholm_external",
        "tallinn_external",
    }
    assert {aoi.aoi_id for aoi in aois if aoi.role is ScientificRole.EXTERNAL_OBSERVED} == {
        "stockholm_external",
        "tallinn_external",
    }


def test_local_artifact_store_is_immutable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes("runs/immutable/result.bin", b"stable")
    assert store.put_bytes("runs/immutable/result.bin", b"stable") == first
    with pytest.raises(FileExistsError, match="immutable artifact"):
        store.put_bytes("runs/immutable/result.bin", b"changed")
    assert store.get_bytes("runs/immutable/result.bin") == b"stable"
