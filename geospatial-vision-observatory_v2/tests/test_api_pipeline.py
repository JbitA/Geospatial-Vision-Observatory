import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from geo_vision.api import create_app
from geo_vision.config import Settings
from geo_vision.domain import ObservationMetadata
from geo_vision.pipeline import Pipeline
from geo_vision.storage import Store


def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, db_path=tmp_path / "test.db", environment="test")


def payload() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (512, 512), (25, 105, 45)).save(buffer, format="PNG")
    return buffer.getvalue()


def metadata() -> ObservationMetadata:
    identifier = "S2_TEST_" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return ObservationMetadata.model_validate(
        {
            "identifier": identifier,
            "caption": "Sentinel-2 test fixture",
            "image": identifier,
            "date": datetime.now(UTC).isoformat(),
            "centroid_coordinates": {"lat": 60.2, "lon": 24.95},
            "source": "Copernicus Sentinel-2 via Earth Search",
            "collection": "sentinel-2-l2a",
            "bbox": [24.8, 60.1, 25.2, 60.35],
            "cloud_cover": 5.0,
        }
    )


class FakeClient:
    def __init__(self) -> None:
        self.record = metadata()

    async def latest_metadata(self) -> ObservationMetadata:
        return self.record

    async def image_bytes(self, item: ObservationMetadata) -> tuple[bytes, str]:
        return payload(), "image/png"


def test_pipeline_and_api(tmp_path: Path) -> None:
    config = settings(tmp_path)
    store = Store(config)
    pipeline = Pipeline(config, store, FakeClient())  # type: ignore[arg-type]
    frame = asyncio.run(pipeline.ingest_once())
    assert frame is not None
    assert asyncio.run(pipeline.ingest_once()) is None
    assert pipeline.processors() is pipeline.processors()
    records = store.latest_frames(1)
    assert "geospatial_rgb_baseline" in str(records[0]["analyses"])

    client = TestClient(create_app(config))
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["audit_chain_valid"] is True
    frames = client.get("/api/v1/frames?limit=1").json()["data"]
    assert len(frames) == 1 and len(frames[0]["analyses"]) == 3
    assert frames[0]["metadata"]["collection"] == "sentinel-2-l2a"
    assert client.get(frames[0]["image_url"]).headers["content-type"] == "image/png"
    status = client.get("/api/v1/system/status").json()
    assert status["source"].startswith("Copernicus Sentinel-2")
    assert status["aoi_bbox"] == [24.8, 60.1, 25.2, 60.35]
    assert client.get("/assets/dashboard.css").status_code == 200
    assert client.get("/assets/dashboard.js").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/api/v1/frames/999/image").status_code == 404


def test_model_summary_requires_matching_bundle_and_artifact(tmp_path: Path) -> None:
    import hashlib
    import json

    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    bundle_dir = model_dir / "landcover"
    report_dir = reports_dir / "landcover"
    bundle_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    artifact = bundle_dir / "landcover_multispectral.pt2"
    artifact.write_bytes(b"verified-model-fixture")
    for name, body in {
        "normalization.json": "{}",
        "classes.json": "{}",
        "training-config.json": "{}",
        "MODEL_CARD.md": "# Fixture\n",
    }.items():
        (bundle_dir / name).write_text(body)
    files = {}
    for path in bundle_dir.iterdir():
        if path.is_file():
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    digest = files[artifact.name]["sha256"]
    dataset_signature = "a" * 64
    experiment_signature = "b" * 64
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task": "worldcover_landcover_segmentation",
                "dataset_signature": dataset_signature,
                "experiment_signature": experiment_signature,
                "model_file": artifact.name,
                "files": files,
            }
        )
    )
    (report_dir / "showcase-summary.json").write_text(
        json.dumps(
            {
                "dataset_signature": dataset_signature,
                "experiment_signature": experiment_signature,
                "model_bundle_sha256": digest,
                "external_test": {"macro_iou": 0.51},
                "showcase_scene": "stockholm_external",
            }
        )
    )
    config = Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        environment="test",
        model_dir=model_dir,
        reports_dir=reports_dir,
    )
    client = TestClient(create_app(config))
    response = client.get("/api/v1/model/summary")
    assert response.status_code == 200
    assert response.json()["model_sha256"] == digest
    artifact.write_bytes(b"tampered")
    assert client.get("/api/v1/model/summary").status_code == 503
