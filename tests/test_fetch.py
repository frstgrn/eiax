"""Unit tests for fetch pagination."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import polars as pl
import pytest

from eiax.cache import CacheConfig
from eiax.client import PAGE_SIZE, EIAClient
from eiax.fetch import _offset_chunks, _page_offsets, fetch_async, fetch_sync


def _page_payload(offset: int, total: int) -> dict:
    remaining = total - offset
    count = min(PAGE_SIZE, remaining)
    data = [
        {
            "period": f"2024-01-01T{((offset + i) % 24):02d}",
            "value": float(offset + i),
        }
        for i in range(count)
    ]
    return {
        "response": {
            "total": str(total),
            "frequency": "hourly",
            "dateFormat": "YYYY-MM-DDTHH",
            "data": data,
        }
    }


def _mock_transport(total: int) -> httpx.MockTransport:
    seen_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlparse(str(request.url)).query)
        offset = int(qs.get("offset", ["0"])[0])
        seen_offsets.append(offset)
        return httpx.Response(200, json=_page_payload(offset, total))

    transport = httpx.MockTransport(handler)
    transport.seen_offsets = seen_offsets  # type: ignore[attr-defined]
    return transport


@pytest.mark.parametrize(
    ("total", "expected_pages"),
    [
        (100, 1),
        (5000, 1),
        (5001, 2),
        (12_000, 3),
    ],
)
def test_page_offsets(total: int, expected_pages: int) -> None:
    offsets = _page_offsets(total)
    assert len(offsets) == expected_pages - 1
    if offsets:
        assert offsets[0] == PAGE_SIZE


def test_offset_chunks() -> None:
    offsets = list(range(5000, 50_000, 5000))
    chunks = _offset_chunks(offsets, size=3)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == len(offsets)


@pytest.mark.asyncio
async def test_fetch_async_multi_page() -> None:
    total = 6000
    transport = _mock_transport(total)
    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._async = httpx.AsyncClient(
        base_url="https://api.eia.gov/v2/", transport=transport
    )
    progress: list[tuple[int, int]] = []

    df = await fetch_async(
        "electricity/rto/region-data/data",
        facets={"respondent": "CISO", "type": "D"},
        frequency="hourly",
        start="2024-01-01",
        end="2024-12-31",
        columns=["value"],
        client=client,
        cache=CacheConfig(enabled=False),
        on_progress=lambda f, t: progress.append((f, t)),
    )

    assert df.height == total
    assert df["value"].dtype == pl.Float64
    assert sorted(transport.seen_offsets) == [0, PAGE_SIZE]  # type: ignore[attr-defined]
    assert progress[-1] == (total, total)
    await client.aclose()


def test_fetch_sync_multi_page() -> None:
    total = 6000
    transport = _mock_transport(total)
    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._async = httpx.AsyncClient(
        base_url="https://api.eia.gov/v2/", transport=transport
    )

    df = fetch_sync(
        "electricity/rto/region-data/data",
        facets={"respondent": "CISO", "type": "D"},
        frequency="hourly",
        start="2024-01-01",
        end="2024-12-31",
        columns=["value"],
        client=client,
        cache=CacheConfig(enabled=False),
    )

    assert df.height == total
    assert sorted(transport.seen_offsets) == [0, PAGE_SIZE]  # type: ignore[attr-defined]
    from eiax._sync import run_sync

    run_sync(client.aclose())


@pytest.mark.asyncio
async def test_fetch_cache_hit_skips_second_request(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        data = [
            {
                "period": f"2024-01-01T{h:02d}",
                "value": float(h),
                "respondent": "CISO",
            }
            for h in range(3)
        ]
        return httpx.Response(
            200,
            json={
                "response": {
                    "total": "3",
                    "frequency": "hourly",
                    "dateFormat": "YYYY-MM-DDTHH",
                    "data": data,
                }
            },
        )

    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._async = httpx.AsyncClient(
        base_url="https://api.eia.gov/v2/", transport=httpx.MockTransport(handler)
    )
    cache = CacheConfig(cache_dir=tmp_path / "cache")
    kwargs = {
        "route": "electricity/rto/region-data/data",
        "facets": {"respondent": "CISO", "type": "D"},
        "frequency": "hourly",
        "start": "2024-01-01T00",
        "end": "2024-01-01T02",
        "columns": ["value"],
        "client": client,
        "cache": cache,
    }
    df1 = await fetch_async(**kwargs)
    first_calls = calls
    df2 = await fetch_async(**kwargs)
    assert df1.height == 3
    assert df2.height == 3
    assert calls == first_calls
    await client.aclose()


def test_fetch_requires_dates_or_period() -> None:
    import pytest

    with pytest.raises(ValueError, match="start/end or period"):
        fetch_sync("electricity/rto/region-data/data")


@pytest.mark.asyncio
async def test_fetch_period_resolves() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": {
                    "total": "1",
                    "frequency": "hourly",
                    "dateFormat": "YYYY-MM-DDTHH",
                    "data": [{"period": "2024-01-01T00", "value": 1.0}],
                }
            },
        )

    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._async = httpx.AsyncClient(
        base_url="https://api.eia.gov/v2/", transport=httpx.MockTransport(handler)
    )
    df = await fetch_async(
        "electricity/rto/region-data/data",
        facets={"respondent": "CISO", "type": "D"},
        frequency="hourly",
        period="1mo",
        columns=["value"],
        client=client,
        cache=CacheConfig(enabled=False),
    )
    assert df.height == 1
    await client.aclose()


def test_fetch_preflight_bad_facet() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown facet"):
        fetch_sync(
            "electricity/rto/region-data/data",
            facets={"not_a_facet": "x"},
            start="2024-01-01",
            end="2024-01-02",
        )


def test_fetch_default_frequency_from_catalog() -> None:
    seen_freq: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlparse(str(request.url)).query)
        if "frequency" in qs:
            seen_freq.append(qs["frequency"][0])
        return httpx.Response(
            200,
            json={
                "response": {
                    "total": "1",
                    "frequency": "monthly",
                    "dateFormat": "YYYY-MM",
                    "data": [{"period": "2024-01", "sales": "1"}],
                }
            },
        )

    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._async = httpx.AsyncClient(
        base_url="https://api.eia.gov/v2/", transport=httpx.MockTransport(handler)
    )
    from eiax._sync import run_sync

    run_sync(
        fetch_async(
            "electricity/retail-sales/data",
            facets={"stateid": "US", "sectorid": "ALL"},
            start="2024-01-01",
            end="2024-01-31",
            client=client,
            cache=CacheConfig(enabled=False),
        )
    )
    assert seen_freq == ["monthly"]
    run_sync(client.aclose())
