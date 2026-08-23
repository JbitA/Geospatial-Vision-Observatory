import asyncio
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from geo_vision.config import Settings
from geo_vision.domain import ObservationMetadata
from geo_vision.source import Sentinel2StacClient, UpstreamInvalid
from geo_vision.storage import Store


def config(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "test.db",
        min_request_spacing_seconds=1,
        retry_attempts=0,
    )


def stac_payload() -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "S2B_35VLG_20260820_0_L2A",
                "bbox": [23.9, 59.4, 25.9, 60.4],
                "properties": {
                    "datetime": "2026-08-20T09:38:11Z",
                    "eo:cloud_cover": 8.5,
                    "proj:epsg": 32635,
                },
                "assets": {
                    "thumbnail": {
                        "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/thumb.jpg"
                    },
                    "visual": {
                        "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/visual.tif"
                    },
                },
            }
        ],
    }


def test_stac_metadata_is_strictly_parsed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/geo+json"},
            json=stac_payload(),
        )

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    record = asyncio.run(client.latest_metadata())
    assert record is not None
    assert record.identifier == "S2B_35VLG_20260820_0_L2A"
    assert record.collection == "sentinel-2-l2a"
    assert record.cloud_cover == 8.5
    assert record.epsg == 32635
    assert record.visual_asset_href and record.visual_asset_href.endswith("visual.tif")


def test_preview_signature_geometry_and_type(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (512, 512), (30, 90, 40)).save(buffer, format="JPEG")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "image/jpeg"}, content=buffer.getvalue()
        )

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    metadata = ObservationMetadata.model_validate(
        {
            "identifier": "S2_TEST_20260820",
            "caption": "test",
            "image": "S2_TEST_20260820",
            "date": "2026-08-20T09:38:11Z",
            "centroid_coordinates": {"lat": 60.2, "lon": 24.95},
            "source": "test",
            "asset_href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/thumb.jpg",
        }
    )
    payload, media_type = asyncio.run(client.image_bytes(metadata))
    assert payload.startswith(b"\xff\xd8\xff")
    assert media_type == "image/jpeg"


def test_semantic_failure_updates_circuit_state(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{")

    settings = config(tmp_path)
    store = Store(settings)
    client = Sentinel2StacClient(settings, store, httpx.MockTransport(handler))
    with pytest.raises(UpstreamInvalid, match="decoded"):
        asyncio.run(client.latest_metadata())
    assert store.get_state("consecutive_failures") == "1"


def test_cloud_threshold_prefers_recent_acceptable_item(tmp_path: Path) -> None:
    payload = stac_payload()
    features = payload["features"]
    assert isinstance(features, list)
    features.append(
        {
            "type": "Feature",
            "id": "S2B_35VLG_20260821_CLOUDY",
            "bbox": [23.9, 59.4, 25.9, 60.4],
            "properties": {"datetime": "2026-08-21T09:38:11Z", "eo:cloud_cover": 90.0},
            "assets": {
                "thumbnail": {
                    "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/cloudy.jpg"
                },
                "visual": {
                    "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/cloudy.tif"
                },
            },
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json=payload)

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    record = asyncio.run(client.latest_metadata())
    assert record is not None and record.identifier == "S2B_35VLG_20260820_0_L2A"


def test_stac_thumbnail_link_is_supported(tmp_path: Path) -> None:
    payload = stac_payload()
    features = payload["features"]
    assert isinstance(features, list)
    item = features[0]
    assert isinstance(item, dict)
    assets = item["assets"]
    assert isinstance(assets, dict)
    assets.pop("thumbnail")
    item["links"] = [
        {
            "rel": "thumbnail",
            "href": (
                "https://earth-search.aws.element84.com/v1/collections/"
                "sentinel-2-l2a/items/test/thumbnail"
            ),
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json=payload)

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    record = asyncio.run(client.latest_metadata())
    assert record is not None
    assert record.asset_href is not None and record.asset_href.endswith("/thumbnail")


def test_retry_after_is_bounded_and_malformed_values_are_ignored() -> None:
    numeric = httpx.Response(429, headers={"retry-after": "999"})
    malformed = httpx.Response(429, headers={"retry-after": "not-a-date"})
    negative = httpx.Response(429, headers={"retry-after": "-5"})
    assert Sentinel2StacClient._retry_after(numeric) == 300.0
    assert Sentinel2StacClient._retry_after(malformed) is None
    assert Sentinel2StacClient._retry_after(negative) == 0.0


def test_expired_circuit_resets_failure_state(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    settings = config(tmp_path)
    store = Store(settings)
    store.set_state("circuit_opened_at", (datetime.now(UTC) - timedelta(days=1)).isoformat())
    store.set_state("consecutive_failures", "9")
    client = Sentinel2StacClient(settings, store, httpx.MockTransport(lambda request: httpx.Response(200)))
    client._check_circuit()
    assert store.get_state("circuit_opened_at") == ""
    assert store.get_state("consecutive_failures") == "0"


def test_open_circuit_blocks_request_before_transport(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from geo_vision.source import CircuitOpen

    settings = config(tmp_path)
    store = Store(settings)
    store.set_state("circuit_opened_at", datetime.now(UTC).isoformat())
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    client = Sentinel2StacClient(settings, store, httpx.MockTransport(handler))
    with pytest.raises(CircuitOpen):
        asyncio.run(client.latest_metadata())
    assert called is False


@pytest.mark.parametrize(
    ("headers", "content", "message"),
    [
        ({"content-type": "application/json", "content-length": "invalid"}, b"{}", "content-length"),
        ({"content-type": "application/json", "content-length": "99999999"}, b"{}", "size ceiling"),
    ],
)
def test_metadata_response_rejects_malformed_or_oversized_length(
    tmp_path: Path, headers: dict[str, str], content: bytes, message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, content=content)

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    with pytest.raises(UpstreamInvalid, match=message):
        asyncio.run(client.latest_metadata())


def test_redirect_is_rejected_and_recorded_as_failure(tmp_path: Path) -> None:
    from geo_vision.security import SecurityViolation

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/"})

    settings = config(tmp_path)
    store = Store(settings)
    client = Sentinel2StacClient(settings, store, httpx.MockTransport(handler))
    with pytest.raises(SecurityViolation, match="redirects"):
        asyncio.run(client.latest_metadata())
    assert store.get_state("consecutive_failures") == "1"


def test_all_cloudy_items_are_rejected_by_declared_threshold(tmp_path: Path) -> None:
    payload = stac_payload()
    features = payload["features"]
    assert isinstance(features, list)
    first = features[0]
    assert isinstance(first, dict)
    properties = first["properties"]
    assert isinstance(properties, dict)
    properties["eo:cloud_cover"] = 70.0

    settings = config(tmp_path).model_copy(update={"sentinel_max_cloud_cover": 10.0})
    client = Sentinel2StacClient(
        settings,
        Store(settings),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "application/json"}, json=payload
            )
        ),
    )
    with pytest.raises(UpstreamInvalid, match="cloud threshold"):
        asyncio.run(client.latest_metadata())


@pytest.mark.parametrize(
    ("content_type", "payload", "message"),
    [
        ("text/plain", b"not-an-image", "unsupported media type"),
        ("image/jpeg", b"", "empty"),
        ("image/jpeg", b"not-a-jpeg", "decoder rejected"),
    ],
)
def test_preview_rejects_invalid_payloads(
    tmp_path: Path, content_type: str, payload: bytes, message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=payload)

    settings = config(tmp_path)
    client = Sentinel2StacClient(settings, Store(settings), httpx.MockTransport(handler))
    metadata = ObservationMetadata.model_validate(
        {
            "identifier": "S2_TEST",
            "caption": "test",
            "image": "S2_TEST",
            "date": "2026-08-20T09:38:11Z",
            "centroid_coordinates": {"lat": 60.2, "lon": 24.95},
            "source": "test",
            "asset_href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test/thumb.jpg",
        }
    )
    with pytest.raises(UpstreamInvalid, match=message):
        asyncio.run(client.image_bytes(metadata))


def test_stac_rejects_naive_nonfinite_and_out_of_window_candidates(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    settings = config(tmp_path)
    now = datetime.now(UTC)
    base = stac_payload()["features"][0]
    assert isinstance(base, dict)

    invalid_properties = (
        {"datetime": now.replace(tzinfo=None).isoformat(), "eo:cloud_cover": 5.0},
        {"datetime": (now - timedelta(days=1)).isoformat(), "eo:cloud_cover": "NaN"},
        {
            "datetime": (now - timedelta(days=settings.sentinel_lookback_days + 1)).isoformat(),
            "eo:cloud_cover": 5.0,
        },
    )
    features: list[dict[str, object]] = []
    for index, properties in enumerate(invalid_properties):
        item = dict(base)
        item["id"] = f"INVALID_{index:04d}"
        item["properties"] = properties
        features.append(item)

    client = Sentinel2StacClient(
        settings,
        Store(settings),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"type": "FeatureCollection", "features": features},
            )
        ),
    )
    with pytest.raises(UpstreamInvalid, match="preview and visual assets"):
        asyncio.run(client.latest_metadata())


def test_stac_invalid_bbox_and_epsg_fall_back_safely(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    payload = stac_payload()
    feature = payload["features"][0]
    assert isinstance(feature, dict)
    feature["bbox"] = ["NaN", 59.4, 25.9, 60.4]
    properties = feature["properties"]
    assert isinstance(properties, dict)
    properties["datetime"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    properties["proj:epsg"] = "NaN"

    settings = config(tmp_path)
    client = Sentinel2StacClient(
        settings,
        Store(settings),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "application/json"}, json=payload
            )
        ),
    )
    record = asyncio.run(client.latest_metadata())
    assert record is not None
    assert record.bbox == settings.aoi_bbox
    assert record.epsg is None
