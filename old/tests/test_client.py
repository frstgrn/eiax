"""Unit tests for client param formatting."""

from __future__ import annotations

import httpx
import pytest

from eiax.client import EIAClient, _format_facets
from eiax.errors import AuthenticationError


def test_format_facets_single_and_list() -> None:
    assert _format_facets({"respondent": "CISO"}) == {"facets[respondent][]": "CISO"}
    assert _format_facets({"type": ["D", "NG"]}) == {"facets[type][]": ["D", "NG"]}


def test_format_facets_multiple_names() -> None:
    params = _format_facets({"respondent": "CISO", "type": "D"})
    assert params == {
        "facets[respondent][]": "CISO",
        "facets[type][]": "D",
    }


def test_missing_api_key_raises() -> None:
    from eiax.settings import EIASettings

    cfg = EIASettings.model_construct(
        api_key=None,
        cache_dir=None,
        rate_limit=5.0,
        cache_enabled=True,
    )
    with pytest.raises(AuthenticationError, match="API key"):
        EIAClient(settings=cfg)


def test_request_builds_url_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = {
        "response": {
            "total": "2",
            "frequency": "hourly",
            "dateFormat": "YYYY-MM-DDTHH",
            "data": [
                {"period": "2024-01-01T00", "value": 100.0},
                {"period": "2024-01-01T01", "value": 99.0},
            ],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api_key=test-key" in str(request.url)
        assert "facets%5Brespondent%5D%5B%5D=CISO" in str(request.url)
        return httpx.Response(200, json=sample)

    transport = httpx.MockTransport(handler)
    client = EIAClient(api_key="test-key")
    client._sync = httpx.Client(base_url="https://api.eia.gov/v2/", transport=transport)
    resp = client.request(
        "electricity/rto/region-data/data",
        {"facets": {"respondent": "CISO"}, "frequency": "hourly"},
    )
    assert resp.response.total == 2
    assert len(resp.data) == 2
    client.close()


def test_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import eiax.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    attempts = 0
    ok = {
        "response": {
            "total": "1",
            "frequency": "hourly",
            "dateFormat": "YYYY-MM-DDTHH",
            "data": [{"period": "2024-01-01T00", "value": 1.0}],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json=ok)

    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._sync = httpx.Client(
        base_url="https://api.eia.gov/v2/", transport=httpx.MockTransport(handler)
    )
    resp = client.request("electricity/rto/region-data/data")
    assert attempts == 2
    assert resp.response.total == 1
    client.close()


def test_does_not_retry_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    import eiax.client as client_mod
    from eiax.errors import EIAError

    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._sync = httpx.Client(
        base_url="https://api.eia.gov/v2/", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(EIAError):
        client.request("electricity/rto/region-data/data")
    assert attempts == 1
    client.close()


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_on_invalid_key_status(status: int) -> None:
    # EIA returns 403 for an invalid key (not 401); both must map to
    # AuthenticationError so users can catch the documented exception.
    client = EIAClient(api_key="test-key", rate_limit=1000.0)
    client._sync = httpx.Client(
        base_url="https://api.eia.gov/v2/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(status, text="invalid key")
        ),
    )
    with pytest.raises(AuthenticationError) as exc:
        client.request("electricity/rto/region-data/data")
    assert exc.value.status_code == status
    client.close()
