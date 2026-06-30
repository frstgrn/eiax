"""Period helpers and long→wide pivot for fetch results."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from eiax.catalog import get_route

# ponytail: fixed period vocabulary; add a timedelta here to extend.
PERIOD_DELTAS: dict[str, timedelta] = {
    "1d": timedelta(days=1),
    "5d": timedelta(days=5),
    "1mo": timedelta(days=30),
    "3mo": timedelta(days=90),
    "6mo": timedelta(days=180),
    "1y": timedelta(days=365),
    "5y": timedelta(days=365 * 5),
}


def resolve_range(
    *,
    start: str | None,
    end: str | None,
    period: str | None,
    route: str,
) -> tuple[str, str]:
    """Resolve a date range from explicit bounds or a period shorthand."""
    if start is not None and end is not None:
        return start, end

    if period is None:
        msg = "provide start and end, or period="
        raise ValueError(msg)

    today = date.today()
    if period == "max":
        info = get_route(route)
        return str(info["start_period"]), today.isoformat()

    if period not in PERIOD_DELTAS:
        raise ValueError(f"Unknown period: {period!r}")

    delta = PERIOD_DELTAS[period]
    return (today - delta).isoformat(), today.isoformat()


def _numeric_measures(df: pl.DataFrame) -> list[str]:
    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    return [
        c
        for c in df.columns
        if c != "period" and not c.endswith("-name") and df.schema[c] in numeric_types
    ]


def value_column(df: pl.DataFrame, measure: str | None = None) -> str:
    if measure is not None:
        if measure not in df.columns:
            msg = f"measure column {measure!r} not in frame; columns: {df.columns}"
            raise ValueError(msg)
        return measure
    if "value" in df.columns:
        return "value"
    numeric = _numeric_measures(df)
    if len(numeric) == 1:
        return numeric[0]
    if len(numeric) > 1:
        msg = (
            f"ambiguous measure columns: {numeric}; pass measure= "
            "(e.g. measure='sales' for retail sales)"
        )
        raise ValueError(msg)
    msg = "no measure column found"
    raise ValueError(msg)


def facet_columns(df: pl.DataFrame, *, measure: str | None = None) -> list[str]:
    col = value_column(df, measure=measure)
    skip = {"period", col, "series"}
    return [c for c in df.columns if c not in skip and not c.endswith("-name")]


def to_wide(
    df: pl.DataFrame,
    *,
    by: str | None = None,
    measure: str | None = None,
) -> pl.DataFrame:
    """Long → wide: period index, one column per facet value.

    ``by`` names the facet column to pivot on (defaults to the lone non-measure
    facet column, or ``"series"`` if present).
    """
    measure_col = value_column(df, measure=measure)
    if measure is not None and by is None:
        return df.select("period", measure_col)
    if by == "series" or (by is None and "series" in df.columns):
        return df.pivot(on="series", index="period", values=measure_col).sort("period")

    if by is not None:
        if by not in df.columns:
            msg = f"by column {by!r} not in frame; columns: {df.columns}"
            raise ValueError(msg)
        return df.pivot(on=by, index="period", values=measure_col).sort("period")

    facets = facet_columns(df, measure=measure_col)
    if not facets:
        return df.select("period", measure_col)
    if len(facets) == 1:
        return df.pivot(on=facets[0], index="period", values=measure_col).sort("period")
    key = pl.concat_str([pl.col(c) for c in facets], separator="_")
    return (
        df.with_columns(key.alias("_key"))
        .pivot(on="_key", index="period", values=measure_col)
        .sort("period")
    )
