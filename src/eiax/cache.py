"""Parquet cache + sqlite manifest, gap detection, TTL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import polars as pl

from eiax.settings import get_settings

_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS partitions (
  route         TEXT NOT NULL,
  frequency     TEXT NOT NULL,
  facets_key    TEXT NOT NULL,
  covered_start TEXT NOT NULL,
  covered_end   TEXT NOT NULL,
  row_count     INTEGER NOT NULL,
  written_at    TEXT NOT NULL,
  PRIMARY KEY (route, frequency, facets_key)
);
"""


@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: Path | None = None
    recent_ttl_hours: int = 48

    @classmethod
    def from_settings(cls) -> CacheConfig:
        cfg = get_settings()
        return cls(
            enabled=cfg.cache_enabled,
            cache_dir=cfg.cache_dir,
            recent_ttl_hours=cfg.cache_ttl_hours,
        )

    def resolve_dir(self) -> Path:
        if self.cache_dir is not None:
            return Path(self.cache_dir)
        if cache_home := os.environ.get("XDG_CACHE_HOME"):
            return Path(cache_home) / "eiax"
        return Path.home() / ".cache" / "eiax"


class DateRange(NamedTuple):
    start: str
    end: str


def facets_key(facets: dict[str, str | list[str]]) -> str:
    normalized = json.dumps(facets, sort_keys=True, default=list)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_MANIFEST_SCHEMA)
    return conn


def _partition_dir(cache_dir: Path, route: str, frequency: str, fkey: str) -> Path:
    return cache_dir.joinpath(*route.strip("/").split("/"), frequency or "_", fkey)


def _parse_bound(value: str) -> datetime:
    for fmt in (
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    msg = f"unsupported period bound: {value!r}"
    raise ValueError(msg)


def _period_bounds(df: pl.DataFrame) -> tuple[str, str]:
    periods = df.sort("period")["period"]
    start = periods[0]
    end = periods[-1]
    return _period_to_cache_str(start), _period_to_cache_str(end)


def _period_to_cache_str(value: object) -> str:
    if isinstance(value, datetime):
        dt = value.astimezone(UTC)
        if dt.hour or dt.minute:
            if dt.minute:
                return dt.strftime("%Y-%m-%dT%H:%M")
            return dt.strftime("%Y-%m-%dT%H")
        if dt.day == 1 and dt.month == 1 and dt.hour == 0:
            return dt.strftime("%Y")
        if dt.day == 1 and dt.hour == 0:
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")
    return str(value)


def _merge_ranges(ranges: list[DateRange]) -> list[DateRange]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: _parse_bound(r.start))
    merged: list[DateRange] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        if _parse_bound(current.start) <= _parse_bound(prev.end):
            later = (
                prev.end
                if _parse_bound(prev.end) >= _parse_bound(current.end)
                else current.end
            )
            merged[-1] = DateRange(prev.start, later)
        else:
            merged.append(current)
    return merged


def find_gaps(
    row: tuple[str, str, str] | None,
    *,
    start: str,
    end: str,
    recent_ttl_hours: int,
    now: datetime | None = None,
) -> list[DateRange]:
    """Return sub-ranges that still need a network fetch."""
    if row is None:
        return [DateRange(start, end)]

    covered_start, covered_end, written_at_raw = row
    start_dt = _parse_bound(start)
    end_dt = _parse_bound(end)
    cov_start_dt = _parse_bound(covered_start)
    cov_end_dt = _parse_bound(covered_end)

    gaps: list[DateRange] = []
    if start_dt < cov_start_dt:
        gaps.append(DateRange(start, covered_start))
    if end_dt > cov_end_dt:
        gaps.append(DateRange(covered_end, end))

    now = now or datetime.now(UTC)
    written_at = datetime.fromisoformat(written_at_raw)
    if written_at.tzinfo is None:
        written_at = written_at.replace(tzinfo=UTC)
    if now - written_at > timedelta(hours=recent_ttl_hours):
        cutoff_dt = now - timedelta(hours=recent_ttl_hours)
        if cutoff_dt <= end_dt:
            recent_start = (
                start if start_dt >= cutoff_dt else cutoff_dt.strftime("%Y-%m-%dT%H")
            )
            gaps.append(DateRange(recent_start, end))

    return _merge_ranges(gaps)


def _key_columns(df: pl.DataFrame) -> list[str]:
    """Dedup key = period + facet-code columns.

    Measures are Float64 after parsing, so excluding numeric columns drops them
    regardless of their name; ``-name`` descriptive columns are functionally
    dependent on their code column and are excluded too. This keeps "new rows
    win" correct even for routes whose measure isn't literally called ``value``.
    """
    keys = ["period"]
    keys += [
        c
        for c in df.columns
        if c != "period" and df.schema[c] == pl.Utf8 and not c.endswith("-name")
    ]
    return keys


class CacheStore:
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.cache_dir = config.resolve_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.cache_dir / "manifest.db"

    def _manifest_row(
        self, route: str, frequency: str, fkey: str
    ) -> tuple[str, str, str, int] | None:
        with _connect(self._manifest) as conn:
            row = conn.execute(
                """
                SELECT covered_start, covered_end, written_at, row_count
                FROM partitions
                WHERE route = ? AND frequency = ? AND facets_key = ?
                """,
                (route, frequency or "", fkey),
            ).fetchone()
        if row is None:
            return None
        return row[0], row[1], row[2], row[3]

    def gaps(
        self,
        route: str,
        frequency: str | None,
        facets: dict[str, str | list[str]],
        start: str,
        end: str,
    ) -> list[DateRange]:
        row = self._manifest_row(route, frequency or "", facets_key(facets))
        if row is None:
            return find_gaps(None, start=start, end=end, recent_ttl_hours=0)
        return find_gaps(
            (row[0], row[1], row[2]),
            start=start,
            end=end,
            recent_ttl_hours=self.config.recent_ttl_hours,
        )

    def partition_path(
        self, route: str, frequency: str | None, facets: dict[str, str | list[str]]
    ) -> Path:
        return _partition_dir(
            self.cache_dir, route, frequency or "_", facets_key(facets)
        )

    def read_slice(
        self,
        route: str,
        frequency: str | None,
        facets: dict[str, str | list[str]],
        start: str,
        end: str,
    ) -> pl.DataFrame | None:
        path = self.partition_path(route, frequency, facets) / "data.parquet"
        if not path.is_file():
            return None
        start_dt = _parse_bound(start)
        end_dt = _parse_bound(end)
        return (
            pl.scan_parquet(path)
            .filter(
                pl.col("period") >= pl.lit(start_dt, dtype=pl.Datetime("us", "UTC")),
                pl.col("period") <= pl.lit(end_dt, dtype=pl.Datetime("us", "UTC")),
            )
            .collect()
        )

    def merge_write(
        self,
        route: str,
        frequency: str | None,
        facets: dict[str, str | list[str]],
        frame: pl.DataFrame,
    ) -> None:
        if frame.is_empty():
            return
        part_dir = self.partition_path(route, frequency, facets)
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / "data.parquet"
        tmp = part_dir / "data.parquet.tmp"

        existing: pl.DataFrame | None = None
        if path.is_file():
            existing = pl.read_parquet(path)

        keys = _key_columns(frame)
        if existing is not None and not existing.is_empty():
            merged = (
                pl.concat([existing, frame], how="diagonal_relaxed")
                .unique(subset=keys, keep="last")
                .sort("period")
            )
        else:
            merged = frame.unique(subset=keys, keep="last").sort("period")

        merged.write_parquet(tmp)
        os.replace(tmp, path)

        covered_start, covered_end = _period_bounds(merged)
        written_at = datetime.now(UTC).isoformat()
        with _connect(self._manifest) as conn:
            conn.execute(
                """
                INSERT INTO partitions (
                  route, frequency, facets_key,
                  covered_start, covered_end, row_count, written_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(route, frequency, facets_key) DO UPDATE SET
                  covered_start = excluded.covered_start,
                  covered_end = excluded.covered_end,
                  row_count = excluded.row_count,
                  written_at = excluded.written_at
                """,
                (
                    route,
                    frequency or "",
                    facets_key(facets),
                    covered_start,
                    covered_end,
                    merged.height,
                    written_at,
                ),
            )
            conn.commit()


if __name__ == "__main__":
    full = find_gaps(None, start="2024-01-01", end="2024-03-01", recent_ttl_hours=48)
    assert full == [DateRange("2024-01-01", "2024-03-01")]
    row = ("2024-02-01", "2024-02-28", datetime.now(UTC).isoformat())
    gaps = find_gaps(row, start="2024-01-01", end="2024-03-01", recent_ttl_hours=48)
    assert DateRange("2024-01-01", "2024-02-01") in gaps
    assert DateRange("2024-02-28", "2024-03-01") in gaps
    stale = ("2024-01-01", "2024-03-01", "2020-01-01T00:00:00+00:00")
    now = datetime(2024, 3, 15, tzinfo=UTC)
    stale_gaps = find_gaps(
        stale,
        start="2024-01-01",
        end="2024-03-15",
        recent_ttl_hours=48,
        now=now,
    )
    assert stale_gaps
    print("cache self-check ok")
