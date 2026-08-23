from __future__ import annotations

import asyncio
import email.utils
import hashlib
import math
import os
import secrets
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Never, Protocol
from urllib.parse import urlencode, urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .domain import ObservationMetadata
from .security import SecurityViolation, reject_private_resolution, validate_outbound_url
from .storage import Store


class BudgetExhausted(RuntimeError):
    pass


class CircuitOpen(RuntimeError):
    pass


class UpstreamInvalid(RuntimeError):
    pass


class SourceClient(Protocol):
    async def latest_metadata(self) -> ObservationMetadata | None: ...

    async def image_bytes(self, metadata: ObservationMetadata) -> tuple[bytes, str]: ...


class SecureUpstreamClient:
    def __init__(
        self, settings: Settings, store: Store, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.settings = settings
        self.store = store
        self.transport = transport

    def _check_circuit(self) -> None:
        opened = self.store.get_state("circuit_opened_at")
        if not opened:
            return
        opened_at = datetime.fromisoformat(opened)
        if datetime.now(UTC) - opened_at < timedelta(seconds=self.settings.circuit_open_seconds):
            raise CircuitOpen("upstream circuit is open")
        self.store.set_state("circuit_opened_at", "")
        self.store.set_state("consecutive_failures", "0")

    def _note_success(self) -> None:
        self.store.set_state("consecutive_failures", "0")

    def _note_failure(self) -> None:
        failures = int(self.store.get_state("consecutive_failures") or "0") + 1
        self.store.set_state("consecutive_failures", str(failures))
        if failures >= self.settings.circuit_failure_threshold:
            self.store.set_state("circuit_opened_at", datetime.now(UTC).isoformat())

    def _reject_invalid(self, message: str) -> Never:
        self._note_failure()
        raise UpstreamInvalid(message)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """Parse bounded Retry-After seconds or HTTP-date; ignore malformed upstream values."""

        value = response.headers.get("retry-after")
        if not value:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            seconds = (parsed - datetime.now(UTC)).total_seconds()
        if not (seconds >= 0.0):  # also rejects NaN
            return 0.0
        return float(min(seconds, 300.0))

    async def _get(
        self, url: str, purpose: str, params: dict[str, str] | None = None
    ) -> httpx.Response:
        self._check_circuit()
        validate_outbound_url(url, self.settings)
        host = urlsplit(url).hostname or ""
        proxy_configured = any(
            os.getenv(name) for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
        )
        if self.transport is None and not (
            self.settings.use_environment_proxy and proxy_configured
        ):
            await asyncio.to_thread(reject_private_resolution, host)
        headers = {
            "Accept": "application/json" if purpose == "stac" else "image/png,image/jpeg",
            "User-Agent": "geospatial-vision-observatory/1.0 (+open geospatial research)",
        }
        identity = url if not params else f"{url}?{urlencode(sorted(params.items()))}"
        etag_key = f"etag:{hashlib.sha256(identity.encode()).hexdigest()}"
        etag = self.store.get_state(etag_key)
        if etag:
            headers["If-None-Match"] = etag
        attempts = self.settings.retry_attempts + 1
        async with httpx.AsyncClient(
            transport=self.transport,
            timeout=httpx.Timeout(self.settings.request_timeout_seconds),
            follow_redirects=False,
            verify=True,
        ) as client:
            for attempt in range(attempts):
                spacing_wait = self.store.seconds_until_next_request(datetime.now(UTC))
                if spacing_wait:
                    await asyncio.sleep(spacing_wait)
                now = datetime.now(UTC)
                request_id = self.store.acquire_request_budget(host, purpose, now)
                if request_id is None:
                    raise BudgetExhausted("global request budget or spacing guard denied request")
                try:
                    request = client.build_request("GET", url, params=params, headers=headers)
                    streamed = await client.send(request, stream=True)
                    response_limit = (
                        self.settings.max_metadata_bytes
                        if purpose == "stac"
                        else self.settings.max_image_bytes
                    )
                    declared = streamed.headers.get("content-length")
                    if declared:
                        try:
                            declared_bytes = int(declared)
                        except ValueError as error:
                            await streamed.aclose()
                            raise UpstreamInvalid("upstream content-length is invalid") from error
                        if declared_bytes < 0 or declared_bytes > response_limit:
                            await streamed.aclose()
                            raise UpstreamInvalid("upstream response exceeds size ceiling")
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in streamed.aiter_bytes():
                        received += len(chunk)
                        if received > response_limit:
                            await streamed.aclose()
                            raise UpstreamInvalid("streamed response exceeds size ceiling")
                        chunks.append(chunk)
                    await streamed.aclose()
                    response = httpx.Response(
                        streamed.status_code,
                        headers=streamed.headers,
                        content=b"".join(chunks),
                        request=request,
                    )
                    self.store.finish_request(request_id, str(response.status_code))
                    if 300 <= response.status_code < 400:
                        raise SecurityViolation("redirects are forbidden for upstream fetches")
                    if response.status_code == 304:
                        self._note_success()
                        return response
                    if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError:
                            self._note_failure()
                            raise
                        if new_etag := response.headers.get("etag"):
                            self.store.set_state(etag_key, new_etag[:256])
                        return response
                    if attempt == attempts - 1:
                        self._note_failure()
                        response.raise_for_status()
                    delay = self._retry_after(response)
                except (httpx.TimeoutException, httpx.NetworkError):
                    self.store.finish_request(request_id, "network_error")
                    if attempt == attempts - 1:
                        self._note_failure()
                        raise
                    delay = None
                except (UpstreamInvalid, SecurityViolation):
                    self.store.finish_request(request_id, "invalid")
                    self._note_failure()
                    raise
                base = max(
                    self.settings.min_request_spacing_seconds,
                    self.settings.retry_base_seconds * (2**attempt),
                )
                jitter = secrets.SystemRandom().uniform(0.8, 1.2)
                await asyncio.sleep(delay if delay is not None else base * jitter)
        raise RuntimeError("unreachable")


class Sentinel2StacClient(SecureUpstreamClient):
    """Operational Sentinel-2 L2A client backed by Element 84 Earth Search STAC.

    The service stores a small preview image for dashboard monitoring. The analysis-grade
    COG URL is retained in provenance and consumed by the dataset preparation command.
    """

    ITEMS_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"

    @staticmethod
    def _parse_datetime(value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("missing Sentinel-2 datetime")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Sentinel-2 datetime must include an explicit timezone")
        return parsed.astimezone(UTC)

    @staticmethod
    def _preview_href(item: dict[str, object], assets: dict[str, object]) -> str | None:
        """Resolve a small browser preview across current and legacy Earth Search items."""

        links = item.get("links")
        if isinstance(links, list):
            for link in links:
                if (
                    isinstance(link, dict)
                    and link.get("rel") == "thumbnail"
                    and isinstance(link.get("href"), str)
                ):
                    return str(link["href"])
        for key in ("rendered_preview", "preview", "thumbnail"):
            asset = assets.get(key)
            if isinstance(asset, dict) and isinstance(asset.get("href"), str):
                return str(asset["href"])
        return None

    async def latest_metadata(self) -> ObservationMetadata | None:
        now = datetime.now(UTC)
        start = now - timedelta(days=self.settings.sentinel_lookback_days)
        west, south, east, north = self.settings.aoi_bbox
        response = await self._get(
            self.ITEMS_URL,
            "stac",
            {
                "bbox": f"{west},{south},{east},{north}",
                "datetime": (
                    f"{start.isoformat().replace('+00:00', 'Z')}/"
                    f"{now.isoformat().replace('+00:00', 'Z')}"
                ),
                "limit": str(self.settings.sentinel_search_limit),
            },
        )
        if response.status_code == 304:
            return None
        if "json" not in response.headers.get("content-type", "").lower():
            self._reject_invalid("Earth Search STAC response is not JSON")
        try:
            payload = response.json()
        except ValueError:
            self._reject_invalid("Earth Search STAC JSON could not be decoded")
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list) or not features:
            self._reject_invalid("Earth Search returned no Sentinel-2 items for the AOI")

        parsed: list[tuple[datetime, float, dict[str, object]]] = []
        for item in features:
            if not isinstance(item, dict):
                continue
            properties = item.get("properties")
            assets = item.get("assets")
            if not isinstance(properties, dict) or not isinstance(assets, dict):
                continue
            try:
                acquired = self._parse_datetime(properties.get("datetime"))
                cloud = float(properties.get("eo:cloud_cover", 100.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(cloud) or not 0.0 <= cloud <= 100.0:
                continue
            if acquired < start or acquired > now + timedelta(minutes=5):
                continue
            preview_href = self._preview_href(item, assets)
            visual = assets.get("visual")
            if preview_href is None:
                continue
            if not isinstance(visual, dict) or not isinstance(visual.get("href"), str):
                continue
            parsed.append((acquired, cloud, item))
        if not parsed:
            self._reject_invalid("Sentinel-2 items did not expose preview and visual assets")

        acceptable = [row for row in parsed if row[1] <= self.settings.sentinel_max_cloud_cover]
        if not acceptable:
            self._reject_invalid(
                "Earth Search returned no Sentinel-2 item within the configured cloud threshold"
            )
        acquired, cloud, selected = max(acceptable, key=lambda row: row[0])

        identifier = selected.get("id")
        properties = selected.get("properties")
        assets = selected.get("assets")
        if (
            not isinstance(identifier, str)
            or not isinstance(properties, dict)
            or not isinstance(assets, dict)
        ):
            self._reject_invalid("Sentinel-2 item structure is incomplete")
        preview_href = self._preview_href(selected, assets)
        visual = assets.get("visual")
        if preview_href is None or not isinstance(visual, dict):
            self._reject_invalid("Sentinel-2 item assets are incomplete")
        visual_href = visual.get("href")
        if not isinstance(visual_href, str):
            self._reject_invalid("Sentinel-2 asset URLs are missing")
        validate_outbound_url(preview_href, self.settings)
        validate_outbound_url(visual_href, self.settings)

        item_bbox = selected.get("bbox")
        use_item_bbox = False
        if isinstance(item_bbox, list) and len(item_bbox) >= 4:
            values = item_bbox[:4]
            if all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in values
            ):
                iw, isouth, ie, inorth = (float(value) for value in values)
                use_item_bbox = (
                    all(math.isfinite(value) for value in (iw, isouth, ie, inorth))
                    and -180.0 <= iw < ie <= 180.0
                    and -90.0 <= isouth < inorth <= 90.0
                )
        if not use_item_bbox:
            iw, isouth, ie, inorth = west, south, east, north

        epsg_value = properties.get("proj:epsg")
        epsg: int | None = None
        if isinstance(epsg_value, int | float) and not isinstance(epsg_value, bool):
            numeric_epsg = float(epsg_value)
            if math.isfinite(numeric_epsg) and numeric_epsg.is_integer():
                candidate_epsg = int(numeric_epsg)
                if 1000 <= candidate_epsg <= 999999:
                    epsg = candidate_epsg
        self._note_success()
        return ObservationMetadata.model_validate(
            {
                "identifier": identifier,
                "caption": "Sentinel-2 L2A observation selected through Earth Search STAC",
                "image": identifier,
                "date": acquired,
                "centroid_coordinates": {
                    "lat": (isouth + inorth) / 2.0,
                    "lon": (iw + ie) / 2.0,
                },
                "source": "Copernicus Sentinel-2 via Earth Search",
                "collection": "sentinel-2-l2a",
                "bbox": [iw, isouth, ie, inorth],
                "asset_href": preview_href,
                "visual_asset_href": visual_href,
                "cloud_cover": cloud,
                "epsg": epsg,
            }
        )

    async def image_bytes(self, metadata: ObservationMetadata) -> tuple[bytes, str]:
        if not metadata.asset_href:
            raise UpstreamInvalid("Sentinel-2 metadata did not include a preview asset")
        response = await self._get(metadata.asset_href, "image")
        if response.status_code == 304:
            self._reject_invalid("selected Sentinel-2 preview unexpectedly returned 304")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in {"image/jpeg", "image/png"}:
            self._reject_invalid("Sentinel-2 preview returned an unsupported media type")
        payload = response.content
        if not payload or len(payload) > self.settings.max_image_bytes:
            self._reject_invalid("Sentinel-2 preview is empty or exceeds the size ceiling")
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            self._note_failure()
            raise UpstreamInvalid("Sentinel-2 preview decoder rejected payload") from error
        if not (128 <= width <= 4096 and 128 <= height <= 4096):
            self._reject_invalid("Sentinel-2 preview returned implausible geometry")
        self._note_success()
        return payload, content_type
