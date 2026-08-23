from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration; secrets are never serialized."""

    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env", extra="ignore")

    environment: str = "development"
    source: str = Field(default="sentinel2_stac", pattern=r"^sentinel2_stac$")
    poll_interval_seconds: int = Field(default=21_600, ge=300, le=86_400)
    min_request_spacing_seconds: int = Field(default=15, ge=1, le=3_600)
    request_timeout_seconds: float = Field(default=30.0, ge=2.0, le=90.0)
    retry_attempts: int = Field(default=2, ge=0, le=3)
    retry_base_seconds: float = Field(default=2.0, ge=0.25, le=30.0)
    hourly_request_budget: int = Field(default=12, ge=1, le=60)
    daily_request_budget: int = Field(default=120, ge=1, le=1_000)
    circuit_failure_threshold: int = Field(default=3, ge=1, le=10)
    circuit_open_seconds: int = Field(default=21_600, ge=60, le=86_400)
    max_metadata_bytes: int = Field(default=3_000_000, ge=100_000, le=10_000_000)
    max_image_bytes: int = Field(default=15_000_000, ge=100_000, le=50_000_000)
    bind_host: str = Field(default="127.0.0.1", pattern=r"^(127\.0\.0\.1|0\.0\.0\.0)$")
    bind_port: int = Field(default=8080, ge=1024, le=65535)
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/geospatial.db")

    # Default operational AOI: Helsinki metropolitan urban/forest interface.
    aoi_west: float = Field(default=24.80, ge=-180, le=180)
    aoi_south: float = Field(default=60.10, ge=-90, le=90)
    aoi_east: float = Field(default=25.20, ge=-180, le=180)
    aoi_north: float = Field(default=60.35, ge=-90, le=90)
    sentinel_lookback_days: int = Field(default=45, ge=1, le=365)
    sentinel_max_cloud_cover: float = Field(default=30.0, ge=0, le=100)
    sentinel_search_limit: int = Field(default=20, ge=1, le=100)

    host_allowlist: tuple[str, ...] = (
        "earth-search.aws.element84.com",
        "sentinel-cogs.s3.us-west-2.amazonaws.com",
        "sentinel-cogs.s3.amazonaws.com",
    )
    use_environment_proxy: bool = True
    cors_origins: tuple[str, ...] = ()
    audit_hmac_key: SecretStr | None = None
    model_dir: Path = Path("./models")
    reports_dir: Path = Path("./reports")

    @field_validator("host_allowlist")
    @classmethod
    def hosts_are_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any("/" in host or ":" in host or "*" in host for host in value):
            raise ValueError("host_allowlist must contain exact DNS hostnames")
        return tuple(host.lower().rstrip(".") for host in value)

    @model_validator(mode="after")
    def aoi_is_valid(self) -> Settings:
        if self.aoi_west >= self.aoi_east or self.aoi_south >= self.aoi_north:
            raise ValueError("AOI bounds must satisfy west < east and south < north")
        if (self.aoi_east - self.aoi_west) > 2.0 or (self.aoi_north - self.aoi_south) > 2.0:
            raise ValueError("operational AOI is limited to 2 degrees per side")
        return self

    @property
    def aoi_bbox(self) -> tuple[float, float, float, float]:
        return (self.aoi_west, self.aoi_south, self.aoi_east, self.aoi_north)


@lru_cache
def get_settings() -> Settings:
    return Settings()
