"""EIA API Client: Connects to the API, for now handles all API calls"""

from __future__ import annotations

import sys

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_URL = "https://api.eia.gov/v2/"
API_MAX_PAGE_SIZE = 5000

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    eia_api_key:str

class EIAClient:
    def __init__(self):
        settings = AppSettings()
        self._api_key = settings.eia_api_key

    def fetch(
        self, 
        route:str,
        facets:dict[str,str],
        start:str,
        end:str,
        length:int = API_MAX_PAGE_SIZE,
        **extras,
    ):
        if not self._api_key:
            print("Logging: API Key does not exist or is empty, failed to fetch!")
            sys.exit(1)
        elif length <1 or length > API_MAX_PAGE_SIZE:
            raise ValueError(f"length must be between 1 and {API_MAX_PAGE_SIZE}, got {length} instead.")
        else:
            all_rows: list[dict] = []
            offset = 0
            meta = None

            while True:
                params = self._build_params(facets, start, end, length, offset, **extras)
                page = self._get_page(route, params)

                if meta is None:
                    meta = page

                batch = page.get("data") or []   # rename: batch, not rows — avoids confusion
                all_rows.extend(batch)

                total = int(page.get("total", 0))
                if len(all_rows) >= total or not batch:
                    break

                offset += len(batch)

            return all_rows, meta


    def _build_params(
        self, 
        facets:dict[str,str|list[str]], # add anything else needed here 
        start:str, 
        end:str, 
        length:int=API_MAX_PAGE_SIZE,
        offset:int=0,
        **extra, # kwargs for extra arguments, if needed later
    ) -> dict[str,str|list[str]|int]:
        """Build `params` object for the API call"""
        params = {
            "api_key":self._api_key,
            "start": start,
            "end": end,
            "length": length, # what is this doing? -> number of rows to return
            "offset": offset, # filters what pages of the response you want. Needed for future pagination work
        }

        params.update(self._format_facets(facets))
        params.update(extra) # any extra params if they are used

        return params


    def _format_facets(self, facets:dict[str, str|list[str]]):
        """Wrap facet names (keys) with 'facets[key][]' to facets arg to simplify calls to fetch() and improve readability in facet definitions"""
        facets_as_params: dict[str, str|list[str]] = {}
        for name, value in facets.items():
            facets_as_params[f"facets[{name}][]"] = value
        return facets_as_params

    
    def _get_page(self, route:str, params:dict) -> dict:
        response = httpx.get(BASE_URL + route, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()["response"] # return just the response dict

if __name__ == "__main__":
    client = EIAClient()
    route = "electricity/rto/region-data/data"
    facets = {"respondent": "CISO", "type": "D"}

    def check(label: str, rows: list, meta: dict, *, expect_len: int | None = None) -> None:
        total = int(meta.get("total", 0))
        print(f"\n{label}")
        print(f"  len(rows)={len(rows)}  total={total}")
        if rows:
            print(f"  first period={rows[0].get('period')}")
            print(f"  last period={rows[-1].get('period')}")
        if expect_len is not None:
            assert len(rows) == expect_len, f"expected {expect_len}, got {len(rows)}"
        print("  ok")

    # Case 1: single-page sample (5 rows) — no pagination loop, fast smoke check
    page = client._get_page(
        route,
        client._build_params(
            facets, "2024-01-01", "2024-01-02",
            length=5, offset=0, frequency="hourly",
        ),
    )
    rows = page.get("data") or []
    meta = page
    check("Case 1 — single request, length=5", rows, meta, expect_len=5)
    assert int(meta["total"]) > 5
    # Case 2: full month, one page (default length=5000)
    rows, meta = client.fetch(
        route, facets, "2024-01-01", "2024-01-31",
        frequency="hourly",
    )
    check("Case 2 — full month, one page", rows, meta)
    assert len(rows) == int(meta["total"])
    assert int(meta["total"]) < API_MAX_PAGE_SIZE
    # Case 3: full year, multi-page (>5000 hourly rows → 2 requests)
    rows, meta = client.fetch(
        route, facets, "2024-01-01", "2024-12-31",
        frequency="hourly",
    )
    check("Case 3 — full year, multi-page", rows, meta)
    assert int(meta["total"]) > API_MAX_PAGE_SIZE
    assert len(rows) == int(meta["total"])
