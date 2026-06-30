"""EIA API Client: Connects to the API, for now handles all API calls"""

from __future__ import annotations

import sys

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_URL = "https://api.eia.gov/v2/"

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
        length:int,
    ):
        if not self._api_key:
            print("Logging: API Key does not exist or is empty, failed to fetch!")
            sys.exit(1)

        params = self._build_params(facets, start, end, length)

        response = httpx.get(BASE_URL+route, params=params, timeout=30.0)
        return response

    def _build_params(
        self, 
        facets:dict[str,str|list[str]], # add anything else needed here 
        start:str, 
        end:str, 
        length:int,
        **extra, # kwargs for extra arguments, if needed later
    ) -> dict[str,str|list[str]|int]:
        params = {
            "api_key":self._api_key,
            "start": start,
            "end": end,
            "length": length, # what is this doing? -> number of rows to return
            "offset": 0, # filters what pages of the response you want. Needed for future pagination work
        }

        params.update(self._format_facets(facets))
        params.update(extra) # any extra params if they are used
        print(params)
        return params


    def _format_facets(self, facets:dict[str, str|list[str]]):
        facets_as_params: dict[str, str|list[str]] = {}
        for name, value in facets.items():
            facets_as_params[f"facets[{name}][]"] = value
        return facets_as_params

t_route = "electricity/rto/region-data/data"
t_facets = {
    "respondent":"CISO",
    "type":"D",
}
t_start = "2024-01-01"
t_end = "2024-01-02"
t_length = 5


test_client = EIAClient()
test_response = test_client.fetch(
    t_route,
    t_facets,
    t_start,
    t_end,
    t_length
)

print(f"HTTP Response Code: {test_response.status_code}")
print(f"API Response Body: {test_response.json()}")
print(f"API Response Meta: {test_response.json().get("response",{})}")
print(f"API Response Meta: {test_response.json().get("response",{}).get("data",[])}")
