"""Unit tests for parse.py."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from eiax.errors import EmptyResultError
from eiax.parse import records_to_frame
from eiax.schema import ResponseMeta


def test_records_to_frame_types() -> None:
    meta = ResponseMeta(total=2, frequency="hourly", dateFormat="YYYY-MM-DDTHH")
    df = records_to_frame(
        [
            {"period": "2024-01-01T00", "value": "100.5", "respondent": "CISO"},
            {"period": "2024-01-01T01", "value": None, "respondent": "CISO"},
        ],
        meta,
    )
    assert df.height == 2
    assert df["value"].dtype == pl.Float64
    assert df["respondent"].dtype == pl.Utf8
    assert df["period"][0] == datetime(2024, 1, 1, 0, tzinfo=UTC)


def test_empty_records_raises() -> None:
    meta = ResponseMeta(total=0, frequency="hourly", dateFormat="YYYY-MM-DD")
    with pytest.raises(EmptyResultError):
        records_to_frame([], meta)


def test_monthly_period() -> None:
    meta = ResponseMeta(total=1, frequency="monthly", dateFormat="YYYY-MM")
    df = records_to_frame([{"period": "2024-01", "value": 1.0}], meta)
    assert df["period"][0] == datetime(2024, 1, 1, tzinfo=UTC)


def test_multi_measure_columns_cast_to_float() -> None:
    # Routes like retail-sales return several measures (no lone "value").
    meta = ResponseMeta(total=1, frequency="monthly", dateFormat="YYYY-MM")
    df = records_to_frame(
        [
            {
                "period": "2024-01",
                "stateid": "CA",
                "revenue": "123.4",
                "sales": "5678.0",
            }
        ],
        meta,
        measures=["revenue", "sales"],
    )
    assert df["revenue"].dtype == pl.Float64
    assert df["sales"].dtype == pl.Float64
    assert df["stateid"].dtype == pl.Utf8
