"""Pydantic models for EIA API envelopes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseMeta(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    total: int
    frequency: str = ""
    date_format: str = Field(default="YYYY-MM-DD", alias="dateFormat")
    data: list[dict[str, Any]] = Field(default_factory=list)
    description: str | None = None

    @field_validator("total", mode="before")
    @classmethod
    def _coerce_total(cls, v: Any) -> int:
        return int(v)


class EIAResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: ResponseMeta

    @property
    def data(self) -> list[dict[str, Any]]:
        return self.response.data
