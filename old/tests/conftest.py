"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def api_key() -> str | None:
    return os.environ.get("EIA_API_KEY")


@pytest.fixture
def skip_without_key(api_key: str | None) -> str:
    if not api_key:
        pytest.skip("no API key")
    return api_key
