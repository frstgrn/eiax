"""Runtime configuration: env vars for pipelines, optional .env for local dev."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EIASettings(BaseSettings):
    """Env vars for pipelines; optional `.env` for local dev."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str | None = Field(default=None, validation_alias="EIA_API_KEY")
    cache_dir: Path | None = Field(default=None, validation_alias="EIA_CACHE_DIR")
    rate_limit: float = Field(default=5.0, validation_alias="EIA_RATE_LIMIT")
    cache_enabled: bool = Field(default=True, validation_alias="EIA_CACHE_ENABLED")
    cache_ttl_hours: int = Field(default=48, validation_alias="EIA_CACHE_TTL_HOURS")

    def require_api_key(self, explicit: str | None = None) -> str:
        key = explicit or self.api_key
        if not key:
            msg = "EIA API key required: set EIA_API_KEY or pass api_key="
            raise ValueError(msg)
        return key


@lru_cache(maxsize=1)
def get_settings() -> EIASettings:
    return EIASettings()
