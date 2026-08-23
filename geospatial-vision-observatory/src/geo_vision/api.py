from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from .config import Settings, get_settings
from .ml.integrity import verify_bundle
from .storage import Store

REQUESTS = Counter("geo_http_requests_total", "HTTP requests", ["method", "path", "status"])


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = Store(settings)
    app = FastAPI(
        title="Geospatial Vision Observatory",
        version="1.0.0",
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None,
    )
    app.state.store = store
    app.state.settings = settings

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared_length = request.headers.get("content-length")
        if declared_length:
            try:
                if int(declared_length) > 1_000_000:
                    return JSONResponse({"detail": "request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        response.headers.update(
            {
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
                    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                ),
                "Cross-Origin-Opener-Policy": "same-origin",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
                "Cache-Control": "no-store",
            }
        )
        REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
        return response

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        index = Path(__file__).with_name("dashboard.html")
        return HTMLResponse(index.read_text())

    @app.get("/assets/dashboard.css", include_in_schema=False)
    def dashboard_css() -> Response:
        path = Path(__file__).with_name("dashboard.css")
        return Response(path.read_text(), media_type="text/css")

    @app.get("/assets/dashboard.js", include_in_schema=False)
    def dashboard_js() -> Response:
        path = Path(__file__).with_name("dashboard.js")
        return Response(path.read_text(), media_type="text/javascript")

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        with store.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        audit_ok = store.verify_audit_chain()
        if not audit_ok:
            raise HTTPException(status_code=503, detail="audit chain verification failed")
        return {"status": "ready", "audit_chain_valid": True}

    @app.get("/api/v1/frames")
    def frames(limit: int = Query(20, ge=1, le=100)) -> dict[str, object]:
        records = store.latest_frames(limit)
        for record in records:
            record["metadata"] = json.loads(str(record.pop("metadata_json")))
            record["analyses"] = json.loads(str(record["analyses"] or "[]"))
            record["image_url"] = f"/api/v1/frames/{record['id']}/image"
            record.pop("relative_path", None)
        return {"data": records, "count": len(records)}

    @app.get("/api/v1/frames/{frame_id}/image")
    def frame_image(frame_id: int) -> FileResponse:
        path = store.frame_path(frame_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(
            path,
            media_type=store.frame_media_type(frame_id) or "application/octet-stream",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/v1/model/summary")
    def model_summary() -> dict[str, object]:
        summary_path = settings.reports_dir / "landcover" / "showcase-summary.json"
        bundle_path = settings.model_dir / "landcover" / "bundle.json"
        if not summary_path.is_file() or not bundle_path.is_file():
            raise HTTPException(status_code=404, detail="trained showcase summary not available")
        if summary_path.stat().st_size > 1_000_000 or bundle_path.stat().st_size > 1_000_000:
            raise HTTPException(status_code=503, detail="model evidence file exceeds safety limit")
        try:
            summary = json.loads(summary_path.read_text())
            bundle = verify_bundle(settings.model_dir / "landcover")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=503, detail="model evidence is unreadable or invalid") from error
        if not isinstance(summary, dict):
            raise HTTPException(status_code=503, detail="model evidence has an invalid schema")
        model_file = str(bundle["model_file"])
        expected = bundle["files"][model_file]["sha256"]
        if (
            summary.get("model_bundle_sha256") != expected
            or summary.get("dataset_signature") != bundle.get("dataset_signature")
            or summary.get("experiment_signature") != bundle.get("experiment_signature")
        ):
            raise HTTPException(status_code=503, detail="model evidence integrity mismatch")
        external = summary.get("external_test")
        if not isinstance(external, dict):
            raise HTTPException(status_code=503, detail="model evidence metrics are invalid")
        return {
            "task": bundle.get("task"),
            "dataset_signature": bundle.get("dataset_signature"),
            "experiment_signature": bundle.get("experiment_signature"),
            "model_sha256": expected,
            "external_test": external,
            "showcase_scene": summary.get("showcase_scene"),
        }

    @app.get("/api/v1/system/status")
    def status() -> dict[str, object]:
        return {
            "source": "Copernicus Sentinel-2 L2A via Element 84 Earth Search STAC",
            "aoi_bbox": settings.aoi_bbox,
            "lookback_days": settings.sentinel_lookback_days,
            "max_cloud_cover": settings.sentinel_max_cloud_cover,
            "poll_interval_seconds": settings.poll_interval_seconds,
            "hourly_request_budget": settings.hourly_request_budget,
            "daily_request_budget": settings.daily_request_budget,
            "retry_attempts": settings.retry_attempts,
            "trained_landcover_bundle_present": (settings.model_dir / "landcover" / "bundle.json").is_file(),
            "showcase_evidence_present": (settings.reports_dir / "landcover" / "showcase-summary.json").is_file(),
            "circuit_opened_at": store.get_state("circuit_opened_at") or None,
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "geo_vision.api:app",
        host=settings.bind_host,
        port=settings.bind_port,
        proxy_headers=False,
    )
