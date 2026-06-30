"""Auto-pagination, date chunking, cache, and offline preflight."""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from eiax.cache import CacheConfig, CacheStore, DateRange
from eiax.catalog import facet_values, get_route
from eiax.client import PAGE_SIZE, EIAClient
from eiax.errors import EmptyResultError, UnknownSeriesError
from eiax.parse import records_to_frame
from eiax.schema import ResponseMeta

ProgressCallback = Callable[[int, int], None]

# ponytail: skip value validation when a facet has more than this many codes
_MAX_FACET_VALUES_TO_VALIDATE = 500


@dataclass
class _FetchParams:
    route: str
    start: str
    end: str
    facets: dict[str, str | list[str]]
    frequency: str | None


def _measure_columns(route: str, columns: list[str] | None) -> list[str]:
    if columns is not None:
        return columns
    try:
        info = get_route(route)
        raw = info["columns"]
        assert isinstance(raw, list)
        names = [c["name"] for c in raw if c.get("role") == "measure"]
        return names or ["value"]
    except Exception:
        return ["value"]


def _default_frequency(route: str) -> str | None:
    try:
        info = get_route(route)
        freq = str(info.get("default_frequency") or "")
        return freq or None
    except UnknownSeriesError:
        return None


def _preflight(
    route: str,
    frequency: str | None,
    facets: dict[str, str | list[str]],
) -> str | None:
    """Offline validation before HTTP. Returns resolved frequency."""
    info = get_route(route)
    resolved = frequency or str(info.get("default_frequency") or "") or None

    if frequency:
        raw_freqs = info["frequencies"]
        assert isinstance(raw_freqs, list)
        valid_freqs = [f["id"] for f in raw_freqs]
        if frequency not in valid_freqs:
            msg = (
                f"Invalid frequency {frequency!r} for route {route!r}. "
                f"Valid options: {', '.join(valid_freqs)}"
            )
            raise ValueError(msg)

    raw_facets = info["facets"]
    assert isinstance(raw_facets, list)
    valid_facet_ids = {f["id"] for f in raw_facets}
    for key in facets:
        if key not in valid_facet_ids:
            msg = (
                f"Unknown facet {key!r} for route {route!r}. "
                f"Valid facet ids: {', '.join(sorted(valid_facet_ids))}"
            )
            raise ValueError(msg)

    for key, raw_val in facets.items():
        values = raw_val if isinstance(raw_val, list) else [raw_val]
        try:
            table = facet_values(route, key)
        except UnknownSeriesError:
            continue
        if table.height > _MAX_FACET_VALUES_TO_VALIDATE:
            continue
        valid = set(table["value_id"].to_list())
        for val in values:
            if val not in valid:
                matches = difflib.get_close_matches(val, valid, n=3, cutoff=0.5)
                hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
                msg = (
                    f"Invalid facet value {val!r} for {key!r} on route {route!r}."
                    f"{hint} Run facet_values({route!r}, {key!r}) for valid codes."
                )
                raise ValueError(msg)

    return resolved


def _resolve_dates(
    route: str,
    *,
    start: str | None,
    end: str | None,
    period: str | None,
) -> tuple[str, str]:
    if start is not None and end is not None:
        return start, end
    if period is None:
        msg = "pass start/end or period= (e.g. period='1mo')"
        raise ValueError(msg)
    from eiax.series import resolve_range

    return resolve_range(start=start, end=end, period=period, route=route)


def _build_query(
    params: _FetchParams,
    *,
    columns: list[str] | None,
    offset: int = 0,
) -> dict:
    query: dict = {
        "facets": params.facets,
        "offset": offset,
        "length": PAGE_SIZE,
    }
    if params.frequency:
        query["frequency"] = params.frequency
    query["start"] = params.start
    query["end"] = params.end
    query["columns"] = _measure_columns(params.route, columns)
    return query


def _page_offsets(total: int) -> list[int]:
    return list(range(PAGE_SIZE, total, PAGE_SIZE))


async def _fetch_pages_async(
    client: EIAClient,
    route: str,
    base_query: dict,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[dict], ResponseMeta]:
    first = await client.arequest(route, {**base_query, "offset": 0})
    meta = first.response
    records = list(first.data)
    if on_progress:
        on_progress(len(records), meta.total)

    offsets = _page_offsets(meta.total)
    if not offsets:
        return records, meta

    async def _page(offset: int) -> list[dict]:
        resp = await client.arequest(route, {**base_query, "offset": offset})
        return resp.data

    for chunk in _offset_chunks(offsets):
        pages = await asyncio.gather(*[_page(offset) for offset in chunk])
        for page in pages:
            records.extend(page)
        if on_progress:
            on_progress(min(len(records), meta.total), meta.total)

    return records, meta


def _offset_chunks(offsets: list[int], *, size: int = 8) -> list[list[int]]:
    """Batch parallel page offsets (ponytail: cap in-flight requests)."""
    return [offsets[i : i + size] for i in range(0, len(offsets), size)]


def _cacheable(start: str | None, end: str | None, cache: CacheConfig | None) -> bool:
    cfg = cache or CacheConfig.from_settings()
    return cfg.enabled and bool(start and end)


async def _fetch_gap_async(
    client: EIAClient,
    params: _FetchParams,
    gap: DateRange,
    *,
    measures: list[str],
    on_progress: ProgressCallback | None,
) -> pl.DataFrame | None:
    gap_params = _FetchParams(
        route=params.route,
        start=gap.start,
        end=gap.end,
        facets=params.facets,
        frequency=params.frequency,
    )
    base_query = _build_query(gap_params, columns=measures)
    try:
        records, meta = await _fetch_pages_async(
            client, params.route, base_query, on_progress=on_progress
        )
    except EmptyResultError:
        return None
    return records_to_frame(records, meta, measures)


async def fetch_async(
    route: str,
    *,
    facets: dict[str, str | list[str]] | None = None,
    frequency: str | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    columns: list[str] | None = None,
    client: EIAClient | None = None,
    cache: CacheConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Fetch one EIA route; pages >5000 rows in parallel."""
    start, end = _resolve_dates(route, start=start, end=end, period=period)
    facet_map = facets or {}
    frequency = _preflight(route, frequency, facet_map)

    params = _FetchParams(
        route=route,
        start=start,
        end=end,
        facets=facet_map,
        frequency=frequency,
    )
    own_client = client is None
    c = client or EIAClient()
    cfg = cache or CacheConfig.from_settings()
    measures = _measure_columns(params.route, columns)
    try:
        if _cacheable(start, end, cfg):
            store = CacheStore(cfg)
            gaps = store.gaps(params.route, frequency, params.facets, start, end)
            if not gaps:
                hit = store.read_slice(
                    params.route, frequency, params.facets, start, end
                )
                if hit is not None and not hit.is_empty():
                    return hit
            fetched: list[pl.DataFrame] = []
            for gap in gaps:
                frame = await _fetch_gap_async(
                    c,
                    params,
                    gap,
                    measures=measures,
                    on_progress=on_progress,
                )
                if frame is not None:
                    fetched.append(frame)
            if fetched:
                store.merge_write(
                    params.route,
                    frequency,
                    params.facets,
                    pl.concat(fetched, how="diagonal_relaxed"),
                )
            result = store.read_slice(
                params.route, frequency, params.facets, start, end
            )
            if result is not None and not result.is_empty():
                return result
            if fetched:
                return pl.concat(fetched, how="diagonal_relaxed").sort("period")
            raise EmptyResultError("EIA returned zero rows for this query")

        base_query = _build_query(params, columns=measures)
        records, meta = await _fetch_pages_async(
            c, params.route, base_query, on_progress=on_progress
        )
        return records_to_frame(records, meta, measures)
    finally:
        if own_client:
            await c.aclose()


def fetch_sync(
    route: str,
    *,
    facets: dict[str, str | list[str]] | None = None,
    frequency: str | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    columns: list[str] | None = None,
    client: EIAClient | None = None,
    cache: CacheConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Sync fetch — delegates to the async implementation."""
    from eiax._sync import run_sync

    return run_sync(
        fetch_async(
            route,
            facets=facets,
            frequency=frequency,
            start=start,
            end=end,
            period=period,
            columns=columns,
            client=client,
            cache=cache,
            on_progress=on_progress,
        )
    )
