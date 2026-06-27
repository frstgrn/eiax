"""httpx sync + async client: auth, retries, rate limiting."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from eiax.errors import AuthenticationError, EIAError, RateLimitError
from eiax.schema import EIAResponse
from eiax.settings import EIASettings, get_settings

BASE_URL = "https://api.eia.gov/v2/"
MAX_RETRIES = 3
PAGE_SIZE = 5000
RETRY_STATUS = {429, 500, 502, 503, 504}


def _load_json(raw: bytes) -> dict:
    try:
        import orjson

        return orjson.loads(raw)
    except ImportError:
        import json

        return json.loads(raw.decode())


def _format_facets(facets: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Expand facets into EIA query params: facets[name][]=value."""
    params: dict[str, str | list[str]] = {}
    for name, values in facets.items():
        params[f"facets[{name}][]"] = values
    return params


class _RateLimiter:
    """ponytail: min-interval gate; upgrade to token bucket if burst matters."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._last = 0.0
        self._async_lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def acquire(self) -> None:
        now = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    async def acquire_async(self) -> None:
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._lock_loop = loop
        async with self._async_lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class EIAClient:
    """Low-level HTTP client for EIA API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        rate_limit: float | None = None,
        timeout: float = 30.0,
        settings: EIASettings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        try:
            key = cfg.require_api_key(api_key)
        except ValueError as exc:
            raise AuthenticationError(str(exc)) from exc
        self.api_key = key
        limit = rate_limit if rate_limit is not None else cfg.rate_limit
        self._limiter = _RateLimiter(limit)
        self._sync = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        self._async = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        await self._async.aclose()

    def __enter__(self) -> EIAClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> EIAClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def request(
        self,
        route: str,
        params: dict[str, Any] | None = None,
    ) -> EIAResponse:
        return self._request_sync(route, params or {})

    async def arequest(
        self,
        route: str,
        params: dict[str, Any] | None = None,
    ) -> EIAResponse:
        return await self._request_async(route, params or {})

    def _build_params(self, params: dict[str, Any]) -> dict[str, str | list[str]]:
        out: dict[str, str | list[str]] = {"api_key": self.api_key}
        facets = params.pop("facets", None)
        if facets:
            out.update(_format_facets(facets))
        for key, value in params.items():
            if value is None:
                continue
            if key == "columns" and isinstance(value, list):
                for i, col in enumerate(value):
                    out[f"data[{i}]"] = col
            else:
                out[key] = str(value)
        return out

    def _request_sync(self, route: str, params: dict[str, Any]) -> EIAResponse:
        url = urljoin(BASE_URL, route.lstrip("/"))
        query = self._build_params(params)
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._limiter.acquire()
            try:
                resp = self._sync.get(url, params=query)
            except httpx.TransportError as exc:
                last_exc = exc
                time.sleep(2**attempt)
                continue
            try:
                return self._parse_response(resp)
            except EIAError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                last_exc = exc
                time.sleep(2**attempt)
        raise last_exc or EIAError("Request failed after retries")

    async def _request_async(self, route: str, params: dict[str, Any]) -> EIAResponse:
        url = urljoin(BASE_URL, route.lstrip("/"))
        query = self._build_params(params)
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            await self._limiter.acquire_async()
            try:
                resp = await self._async.get(url, params=query)
            except httpx.TransportError as exc:
                last_exc = exc
                await asyncio.sleep(2**attempt)
                continue
            try:
                return self._parse_response(resp)
            except EIAError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                last_exc = exc
                await asyncio.sleep(2**attempt)
        raise last_exc or EIAError("Request failed after retries")

    @staticmethod
    def _should_retry(exc: EIAError, attempt: int) -> bool:
        return attempt < MAX_RETRIES - 1 and exc.status_code in RETRY_STATUS

    def _parse_response(self, resp: httpx.Response) -> EIAResponse:
        if resp.status_code in (401, 403):
            raise AuthenticationError(
                "Invalid or missing EIA API key",
                status_code=resp.status_code,
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "EIA rate limit exceeded",
                status_code=429,
            )
        if resp.status_code >= 500:
            raise EIAError(
                f"EIA server error: {resp.status_code}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            body = resp.text[:500]
            raise EIAError(
                f"EIA request failed ({resp.status_code}): {body}",
                status_code=resp.status_code,
            )
        payload = _load_json(resp.content)
        if "error" in payload:
            err = payload["error"]
            msg = err if isinstance(err, str) else str(err)
            raise EIAError(msg, api_code=payload.get("code"))
        return EIAResponse.model_validate(payload)


Client = EIAClient
