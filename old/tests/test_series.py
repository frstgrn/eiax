"""Unit tests for period resolution and to_wide."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import polars as pl
import pytest

import eiax
from eiax.series import resolve_range, to_wide, value_column


def test_search_finds_routes() -> None:
    hits = eiax.search("demand")
    assert hits.height > 0
    assert (hits["kind"] == "route").all()


def test_resolve_range_period_overrides_when_dates_missing() -> None:
    with patch("eiax.series.date") as mock_date:
        mock_date.today.return_value = datetime(2024, 6, 15).date()
        start, end = resolve_range(
            start=None,
            end=None,
            period="1mo",
            route="electricity/rto/region-data/data",
        )
    assert end == "2024-06-15"
    assert start == "2024-05-16"


def test_resolve_range_start_end_win_over_period() -> None:
    start, end = resolve_range(
        start="2024-01-01",
        end="2024-01-07",
        period="1y",
        route="electricity/rto/region-data/data",
    )
    assert start == "2024-01-01"
    assert end == "2024-01-07"


def test_to_wide_single_facet() -> None:
    df = pl.from_dicts(
        [
            {"period": datetime(2024, 1, 1, tzinfo=UTC), "type": "D", "value": 1.0},
            {"period": datetime(2024, 1, 1, tzinfo=UTC), "type": "NG", "value": 2.0},
        ]
    )
    wide = to_wide(df)
    assert "D" in wide.columns
    assert "NG" in wide.columns


def test_to_wide_explicit_by() -> None:
    df = pl.from_dicts(
        [
            {
                "period": datetime(2024, 1, 1, tzinfo=UTC),
                "respondent": "CISO",
                "value": 1.0,
            },
            {
                "period": datetime(2024, 1, 1, tzinfo=UTC),
                "respondent": "PJM",
                "value": 2.0,
            },
        ]
    )
    wide = to_wide(df, by="respondent")
    assert "CISO" in wide.columns
    assert "PJM" in wide.columns


def test_to_wide_retail_measure() -> None:
    df = pl.from_dicts(
        [
            {
                "period": datetime(2024, 1, 1, tzinfo=UTC),
                "stateid": "US",
                "sales": 100.0,
                "price": 12.0,
            }
        ]
    )
    wide = to_wide(df, measure="sales")
    assert wide.columns == ["period", "sales"]


def test_value_column_ambiguous_raises() -> None:
    df = pl.from_dicts(
        [
            {
                "period": datetime(2024, 1, 1, tzinfo=UTC),
                "sales": 1.0,
                "price": 2.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="ambiguous"):
        value_column(df)
