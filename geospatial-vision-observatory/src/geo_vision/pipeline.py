from __future__ import annotations

import logging
from pathlib import Path

from prometheus_client import Counter, Histogram

from .config import Settings
from .domain import AnalysisRecord, FrameRecord
from .source import SourceClient
from .storage import Store
from .vision import GeospatialRgbBaseline, QualityControl, TemporalChangeBaseline
from .vision.processors import Processor, timed_process

LOGGER = logging.getLogger(__name__)
PROCESSOR_RUNS = Counter(
    "geo_processor_runs_total", "Processor executions", ["processor", "status"]
)
PROCESSOR_SECONDS = Histogram(
    "geo_processor_duration_seconds", "Processor execution time", ["processor"]
)


class Pipeline:
    """Integrity-first preview ingestion; scientific multispectral inference is a separate path."""

    def __init__(self, settings: Settings, store: Store, client: SourceClient):
        self.settings = settings
        self.store = store
        self.client = client
        self._processor_cache: list[Processor] | None = None

    async def ingest_once(self) -> FrameRecord | None:
        metadata = await self.client.latest_metadata()
        if metadata is None or self.store.contains_source(metadata.identifier):
            return None
        payload, media_type = await self.client.image_bytes(metadata)
        frame = self.store.save_frame(metadata, payload, media_type)
        self.run_processors(frame)
        return frame

    def processors(self) -> list[Processor]:
        if self._processor_cache is None:
            self._processor_cache = [
                QualityControl(),
                GeospatialRgbBaseline(),
                TemporalChangeBaseline(),
            ]
        return self._processor_cache

    def run_processors(self, frame: FrameRecord) -> None:
        path = self.settings.data_dir / frame.relative_path
        previous = self._previous_path(frame.id)
        for processor in self.processors():
            try:
                result, duration = timed_process(processor, path, previous)
                status = "ok"
            except Exception as error:  # isolate analysis failures from ingestion durability
                LOGGER.exception("processor failed", extra={"processor": processor.name})
                result, duration, status = {"error_type": type(error).__name__}, 0.0, "failed"
            PROCESSOR_RUNS.labels(processor.name, status).inc()
            PROCESSOR_SECONDS.labels(processor.name).observe(duration / 1_000)
            self.store.record_analysis(
                AnalysisRecord(
                    frame_id=frame.id,
                    processor=processor.name,
                    processor_version=processor.version,
                    status=status,
                    result=result,
                    duration_ms=duration,
                )
            )

    def _previous_path(self, frame_id: int) -> Path | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT relative_path FROM frames WHERE id < ? ORDER BY captured_at DESC LIMIT 1",
                (frame_id,),
            ).fetchone()
        return None if row is None else self.settings.data_dir / row[0]
