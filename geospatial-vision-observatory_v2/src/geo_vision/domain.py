from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Coordinates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ObservationMetadata(BaseModel):
    """Strict source-neutral metadata persisted with every visual observation."""

    model_config = ConfigDict(extra="ignore")
    identifier: str = Field(pattern=r"^[A-Za-z0-9_.:-]{4,160}$")
    caption: str = Field(max_length=500)
    image: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    date: datetime
    centroid_coordinates: Coordinates
    source: str = Field(default="unknown", max_length=80)
    collection: str | None = Field(default=None, max_length=120)
    bbox: tuple[float, float, float, float] | None = None
    asset_href: str | None = Field(default=None, max_length=2048)
    visual_asset_href: str | None = Field(default=None, max_length=2048)
    cloud_cover: float | None = Field(default=None, ge=0, le=100)
    epsg: int | None = Field(default=None, ge=1000, le=999999)

    @field_validator("date")
    @classmethod
    def require_naive_or_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is not None and offset.total_seconds() != 0:
            raise ValueError("observation timestamp must be UTC")
        return value.replace(tzinfo=UTC) if offset is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_bbox(self) -> ObservationMetadata:
        if self.bbox is None:
            return self
        west, south, east, north = self.bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be valid WGS84 west,south,east,north")
        return self


class FrameRecord(BaseModel):
    id: int
    source_id: str
    captured_at: str
    fetched_at: str
    sha256: str
    byte_size: int
    media_type: str
    relative_path: str
    metadata_json: str


class AnalysisRecord(BaseModel):
    frame_id: int
    processor: str
    processor_version: str
    status: str
    result: dict[str, object]
    duration_ms: float
