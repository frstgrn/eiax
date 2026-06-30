# Changelog

All notable changes are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-26

Initial release — Polars-native EIA Open Data API v2 client.

### Added

- `fetch()` / `fetch_async()` with auto-pagination and parallel page fetches.
- Offline route catalog (~230 routes): `search()`, `help_route()`, `facet_values()`.
- Catalog helpers in `eiax.catalog`: `list_routes()`, `list_children()`, `describe_node()`, `get_route()`.
- Parquet + SQLite cache with gap detection and configurable TTL.
- `EIAClient` with sync/async HTTP, rate limiting, and retries on 429/5xx.
- `to_wide()` for reshaping long frames to wide period-indexed tables.
- Settings via `EIA_API_KEY`, `EIA_CACHE_DIR`, `EIA_CACHE_ENABLED`, `EIA_CACHE_TTL_HOURS`, `EIA_RATE_LIMIT`.
- Typed exceptions: `AuthenticationError`, `RateLimitError`, `EmptyResultError`, `UnknownSeriesError`, `EIAError`.
- Optional `[fastjson]` extra for `orjson`-backed JSON decode.
