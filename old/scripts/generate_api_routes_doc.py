"""Regenerate docs/api-routes.md from src/eiax/catalog/routes.parquet."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

ROUTES = (
    Path(__file__).resolve().parent.parent / "src" / "eiax" / "catalog" / "routes.parquet"
)
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "api-routes.md"


def main() -> None:
    df = pl.read_parquet(ROUTES)
    lines = [
        "# EIA API v2 — Route Index",
        "",
        f"Skimmable index of every data route in the shipped eiax catalog (**{df.height} routes**).",
        "You do not need to install eiax to read this file; it is generated from",
        "`src/eiax/catalog/routes.parquet`.",
        "",
        "Regenerate after a catalog refresh:",
        "",
        "```bash",
        "uv run python scripts/generate_api_routes_doc.py",
        "```",
        "",
        "## How every call is shaped",
        "",
        "All routes end with `/data`. A typical request supplies:",
        "",
        "| Parameter | Required | Example |",
        "|---|---|---|",
        '| `route` | yes | `"petroleum/pri/spt/data"` |',
        '| `facets` | usually | `{"series": "RWTC"}` |',
        '| `frequency` | often | `"weekly"`, `"monthly"`, `"hourly"` |',
        '| `start` / `end` | for time series | `"2024-01-01"`, `"2024-12"` |',
        "",
        "With eiax installed (API key required for data):",
        "",
        "```python",
        "import eiax",
        "",
        'df = eiax.series("wti_spot").history(period="1y")',
        "",
        "df = eiax.fetch(",
        '    "petroleum/pri/spt/data",',
        '    facets={"series": "RWTC"},',
        '    frequency="weekly",',
        '    start="2024-01-01",',
        '    end="2024-12-31",',
        ")",
        "",
        'eiax.get_route("petroleum/pri/spt/data")  # offline, no key',
        "```",
        "",
        "---",
        "",
    ]

    tree: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in df.sort("route").iter_rows(named=True):
        path = row["route"].removesuffix("/data")
        parts = path.split("/")
        root = parts[0]
        branch = parts[1] if len(parts) > 1 else "(root)"
        leaf = "/".join(parts[2:]) if len(parts) > 2 else "(index)"
        facets = [f["id"] for f in json.loads(row["facets"])]
        tree[root][branch].append(
            {
                "leaf": leaf,
                "route": row["route"],
                "name": (row["name"] or "").strip(),
                "desc": (row["description"] or "").strip()[:100],
                "freq": row["default_frequency"] or "?",
                "facets": facets,
                "start": row["start_period"] or "",
                "end": row["end_period"] or "",
            }
        )

    for root in sorted(tree):
        routes_n = sum(len(v) for v in tree[root].values())
        lines.append(f"## {root} ({routes_n} routes)")
        lines.append("")
        for branch in sorted(tree[root]):
            items = tree[root][branch]
            lines.append(f"### {root}/{branch} ({len(items)})")
            lines.append("")
            for item in sorted(items, key=lambda x: x["leaf"]):
                title = item["name"] or item["leaf"]
                facet_str = ", ".join(f"`{f}`" for f in item["facets"]) or "—"
                lines.append(f"- **`{item['route']}`** — {title}")
                if item["desc"]:
                    lines.append(f"  - {item['desc']}")
                lines.append(
                    f"  - default frequency: `{item['freq']}` | "
                    f"coverage: `{item['start']}` → `{item['end']}`"
                )
                lines.append(f"  - facets: {facet_str}")
            lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
