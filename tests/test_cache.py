"""Unit tests for cache gap detection and read/write."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from eiax.cache import (
    CacheConfig,
    CacheStore,
    DateRange,
    find_gaps,
)


def test_find_gaps_no_partition() -> None:
    gaps = find_gaps(None, start="2024-01-01", end="2024-03-01", recent_ttl_hours=48)
    assert gaps == [DateRange("2024-01-01", "2024-03-01")]


def test_find_gaps_before_and_after() -> None:
    row = ("2024-02-01", "2024-02-28", datetime.now(UTC).isoformat())
    gaps = find_gaps(row, start="2024-01-01", end="2024-03-01", recent_ttl_hours=48)
    assert DateRange("2024-01-01", "2024-02-01") in gaps
    assert DateRange("2024-02-28", "2024-03-01") in gaps


def test_find_gaps_stale_recent_window() -> None:
    row = ("2024-01-01", "2024-03-01", "2020-01-01T00:00:00+00:00")
    gaps = find_gaps(
        row,
        start="2024-01-01",
        end="2024-03-15",
        recent_ttl_hours=48,
        now=datetime(2024, 3, 15, tzinfo=UTC),
    )
    assert gaps


def test_find_gaps_full_hit() -> None:
    row = ("2024-01-01", "2024-03-01", datetime.now(UTC).isoformat())
    gaps = find_gaps(row, start="2024-02-01", end="2024-02-15", recent_ttl_hours=48)
    assert gaps == []


def test_find_gaps_normalizes_hourly_bounds() -> None:
    row = ("2024-01-01T00", "2024-01-01T23", datetime.now(UTC).isoformat())
    gaps = find_gaps(row, start="2024-01-01", end="2024-01-01T23", recent_ttl_hours=48)
    assert gaps == []


def _sample_frame(start_hour: int, count: int) -> pl.DataFrame:
    rows = [
        {
            "period": datetime(2024, 1, 1, h, tzinfo=UTC),
            "value": float(h),
            "respondent": "CISO",
        }
        for h in range(start_hour, start_hour + count)
    ]
    return pl.from_dicts(rows)


def test_merge_write_and_read_slice(tmp_path: Path) -> None:
    store = CacheStore(CacheConfig(cache_dir=tmp_path / "cache"))
    frame = _sample_frame(0, 3)
    store.merge_write(
        "electricity/rto/region-data/data",
        "hourly",
        {"respondent": "CISO", "type": "D"},
        frame,
    )
    out = store.read_slice(
        "electricity/rto/region-data/data",
        "hourly",
        {"respondent": "CISO", "type": "D"},
        "2024-01-01T00",
        "2024-01-01T01",
    )
    assert out is not None
    assert out.height == 2


def test_merge_dedup_new_rows_win(tmp_path: Path) -> None:
    store = CacheStore(CacheConfig(cache_dir=tmp_path / "cache"))
    route = "electricity/rto/region-data/data"
    facets = {"respondent": "CISO", "type": "D"}
    store.merge_write(route, "hourly", facets, _sample_frame(0, 2))
    updated = pl.from_dicts(
        [
            {
                "period": datetime(2024, 1, 1, 0, tzinfo=UTC),
                "value": 999.0,
                "respondent": "CISO",
            },
        ]
    )
    store.merge_write(route, "hourly", facets, updated)
    out = store.read_slice(route, "hourly", facets, "2024-01-01T00", "2024-01-01T00")
    assert out is not None
    assert out["value"][0] == 999.0


def test_merge_dedup_non_value_measure(tmp_path: Path) -> None:
    # Dedup must work when the measure isn't called "value" (e.g. retail "sales").
    store = CacheStore(CacheConfig(cache_dir=tmp_path / "cache"))
    route = "electricity/retail-sales/data"
    facets = {"stateid": "CA", "sectorid": "ALL"}

    def frame(sales: float) -> pl.DataFrame:
        return pl.from_dicts(
            [
                {
                    "period": datetime(2024, 1, 1, tzinfo=UTC),
                    "stateid": "CA",
                    "sales": sales,
                }
            ]
        )

    store.merge_write(route, "monthly", facets, frame(100.0))
    store.merge_write(route, "monthly", facets, frame(250.0))
    out = store.read_slice(route, "monthly", facets, "2024-01-01", "2024-01-01")
    assert out is not None
    assert out.height == 1
    assert out["sales"][0] == 250.0


def test_gaps_partial_coverage(tmp_path: Path) -> None:
    store = CacheStore(CacheConfig(cache_dir=tmp_path / "cache", recent_ttl_hours=9999))
    route = "electricity/rto/region-data/data"
    facets = {"respondent": "CISO", "type": "D"}
    store.merge_write(route, "hourly", facets, _sample_frame(0, 24))
    gaps = store.gaps(
        route,
        "hourly",
        facets,
        "2024-01-01T00",
        "2024-01-02T02",
    )
    assert gaps
    assert any(g.start >= "2024-01-01T23" for g in gaps)


def test_cache_config_coerces_str_cache_dir(tmp_path: Path) -> None:
    # Public API: users pass string paths; CacheConfig must coerce to Path.
    store = CacheStore(CacheConfig(cache_dir=str(tmp_path / "cache")))
    store.merge_write(
        "electricity/rto/region-data/data",
        "hourly",
        {"respondent": "CISO", "type": "D"},
        _sample_frame(0, 2),
    )
    out = store.read_slice(
        "electricity/rto/region-data/data",
        "hourly",
        {"respondent": "CISO", "type": "D"},
        "2024-01-01T00",
        "2024-01-01T01",
    )
    assert out is not None and out.height == 2
