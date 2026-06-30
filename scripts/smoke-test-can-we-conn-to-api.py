"""Smoke test to confirm API functionality"""

import sys
import httpx

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_URL = "https://api.eia.gov/v2/"
ROUTE = "electricity/rto/region-data/data"

class TestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    eia_api_key: str



def test_api_connection() -> None:
    settings = TestSettings()
    api_key = settings.eia_api_key

    if not api_key: # using if not instead of key == None catches None + empty str
        print("Failed to fetch API key")
        sys.exit(1)

    params = {
        "api_key":api_key,
        "facets[respondent][]": "CISO",
        "facets[type][]": "D",
        "start": "2024-01-01",
        "end": "2024-01-31",
        "length": 5, # what is this doing? -> number of rows to return
        "offset": 0,
    }

    test_response = httpx.get(BASE_URL+ROUTE, params=params, timeout=30.0)
    print (f"HTTP {test_response.status_code}")

    if test_response.status_code in (401, 403):
        print("auth failed, check API key or params", file=sys.stderr)
        sys.exit(1)

    body = test_response.json()
    meta = body.get("response", {})
    rows = meta.get("data", [])

    print(body)
    print(meta)
    print(rows)

test_api_connection()