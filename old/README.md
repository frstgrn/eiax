# eiax v0.1 (archived)

This directory holds the original package (v0.1.0). It is not published from here.

The repo root is reserved for the v2 rewrite. To work on or run v1:

```bash
cd old
uv sync
export EIA_API_KEY=your_key   # or use ../.env from repo root
uv run pytest -m "not integration"
```

Original package README: [PACKAGE_README.md](PACKAGE_README.md)
