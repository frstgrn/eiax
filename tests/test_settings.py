"""Settings loading from env and optional .env file."""

from __future__ import annotations

from eiax.settings import get_settings


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "test-key-from-env")
    monkeypatch.setenv("EIA_RATE_LIMIT", "2.5")
    get_settings.cache_clear()
    cfg = get_settings()
    assert cfg.api_key == "test-key-from-env"
    assert cfg.rate_limit == 2.5
    get_settings.cache_clear()


def test_explicit_api_key_overrides_settings() -> None:
    from eiax.client import EIAClient

    client = EIAClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"
    client.close()
