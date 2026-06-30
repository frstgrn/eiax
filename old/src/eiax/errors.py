"""Typed exceptions for eiax."""

from __future__ import annotations


class EIAError(Exception):
    """Base exception; carries HTTP/API context when available."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        api_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.api_code = api_code


class AuthenticationError(EIAError):
    """Missing or invalid API key."""


class RateLimitError(EIAError):
    """429 after retries exhausted."""


class EmptyResultError(EIAError):
    """Valid request returned zero rows."""


class UnknownSeriesError(EIAError):
    """Unknown route or tree path (offline catalog lookup)."""
