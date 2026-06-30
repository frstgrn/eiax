"""Live EIA API integration tests."""

from __future__ import annotations

import polars as pl
import pytest

import eiax


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ciso_demand_live(skip_without_key: str) -> None:
    df = await eiax.fetch_async(
        "electricity/rto/region-data/data",
        facets={"respondent": "CISO", "type": "D"},
        frequency="hourly",
        start="2024-01-01",
        end="2024-01-07",
    )
    assert df.height > 0
    assert df["value"].dtype == pl.Float64
    assert "period" in df.columns
