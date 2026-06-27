# eiax

Unofficial Python client for the EIA Open Data API v2. polars-native, auto-paginating, and parquet cached.

## Features

- **Polars-native** — every fetch returns a typed `pl.DataFrame` (`df.to_pandas()` if you prefer).
- **Invisible pagination** — EIA caps responses at 5,000 rows; eiax fetches as much as you want in parallel and returns one frame.
- **Offline catalog** — EIA's route tree ships inside the package. `search`, `help_route`, and `facet_values` work with no API key.
- **Parquet cache** — repeat queries with the same route/facets/range read from disk; partial coverage fetches only the missing date ranges.
- **Typed errors** — `AuthenticationError`, `RateLimitError`, `EmptyResultError`, `UnknownSeriesError`, `EIAError`.

## Installation

```bash
pip install eiax # install eiax[fastjson] to use orjson for faster loads
```

eiax requires Python 3.12+.

## API key

Get a key at [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php). Browsing the offline catalog (`search`, `help_route`, `facet_values`) does not need a key.

Provide the key any of these ways (highest precedence first):

```python
import eiax

# 1. Explicit argument
client = eiax.EIAClient(api_key="your_key")
# 2. Environment variable (best for pipelines / CI)
#   export EIA_API_KEY=your_key
# 3. A local .env file (best for development)
#   EIA_API_KEY=your_key
```

## License & attribution

eiax is released under the [MIT License](LICENSE). It is not affiliated with the U.S. EIA.
