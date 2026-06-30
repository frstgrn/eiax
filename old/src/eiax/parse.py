"""JSON records → Polars DataFrame with route-aware typing."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import polars as pl

from eiax.errors import EmptyResultError
from eiax.schema import ResponseMeta

# EIA dateFormat strings → strptime patterns (ponytail: covers common frequencies)
_DATE_FORMATS: dict[str, str] = {
    "YYYY-MM-DDTHH": "%Y-%m-%dT%H",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY-MM": "%Y-%m",
    "YYYY": "%Y",
}


def records_to_frame(
    records: list[dict],
    meta: ResponseMeta,
    measures: list[str] | None = None,
) -> pl.DataFrame:
    """Convert EIA data rows to a typed Polars frame.

    ``measures`` names the numeric value columns to cast to Float64 (e.g.
    ``["value"]`` or ``["revenue", "sales"]``). When omitted, we fall back to a
    lone ``value`` column so direct/low-level calls still type correctly.
    """
    if not records:
        raise EmptyResultError("EIA returned zero rows for this query")

    df = pl.from_dicts(records, infer_schema_length=None)

    if "period" not in df.columns:
        raise EmptyResultError("EIA response has no period column")

    df = df.with_columns(_parse_period(meta.date_format))

    measure_cols = [m for m in (measures or ["value"]) if m in df.columns]
    if measure_cols:
        df = df.with_columns(
            [pl.col(m).cast(pl.Float64, strict=False) for m in measure_cols]
        )
    elif "value" not in df.columns and measures is None:
        warnings.warn("EIA response has no value column", stacklevel=2)

    string_cols = [c for c in df.columns if c != "period" and c not in measure_cols]
    if string_cols:
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in string_cols])

    return df.sort("period")


def _parse_period(date_format: str) -> pl.Expr:
    normalized = date_format.replace('"', "").replace("HH24", "HH")
    if normalized == "YYYY-MM-DDTHH":
        padded = (
            pl.when(pl.col("period").str.len_chars() == 13)
            .then(pl.col("period") + ":00")
            .otherwise(pl.col("period"))
        )
        return padded.str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC")
    fmt = _DATE_FORMATS.get(normalized)
    if fmt:
        return pl.col("period").str.to_datetime(fmt, time_zone="UTC")
    warnings.warn(
        f"Unknown EIA dateFormat {date_format!r}; parsing as string",
        stacklevel=2,
    )
    return pl.col("period").cast(pl.Utf8)


if __name__ == "__main__":
    meta = ResponseMeta(total=2, frequency="hourly", dateFormat="YYYY-MM-DDTHH")
    frame = records_to_frame(
        [
            {"period": "2024-01-01T00", "value": "100.5", "respondent": "CISO"},
            {"period": "2024-01-01T01", "value": "99.0", "respondent": "CISO"},
        ],
        meta,
    )
    assert frame.height == 2
    assert frame["value"].dtype == pl.Float64
    assert frame["respondent"].dtype == pl.Utf8
    assert frame["period"][0] == datetime(2024, 1, 1, 0, tzinfo=UTC)
    print("parse self-check ok")
