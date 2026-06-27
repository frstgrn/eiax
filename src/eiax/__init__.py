"""Public exports."""

from eiax.__version__ import __version__
from eiax.cache import CacheConfig
from eiax.catalog import facet_values, help_route, search
from eiax.client import EIAClient
from eiax.errors import (
    AuthenticationError,
    EIAError,
    EmptyResultError,
    RateLimitError,
    UnknownSeriesError,
)
from eiax.fetch import fetch_async, fetch_sync
from eiax.series import to_wide

fetch = fetch_sync

__all__ = [
    "__version__",
    "AuthenticationError",
    "CacheConfig",
    "EIAClient",
    "EIAError",
    "EmptyResultError",
    "RateLimitError",
    "UnknownSeriesError",
    "facet_values",
    "fetch",
    "fetch_async",
    "help_route",
    "search",
    "to_wide",
]
