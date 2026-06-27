"""Walk the EIA v2 metadata tree and write catalog parquet files.

Build one top-level category at a time (merge into existing parquet):

    uv run python scripts/build_catalog.py coal
    uv run python scripts/build_catalog.py --list
    uv run python scripts/build_catalog.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from eiax.settings import get_settings

BASE_URL = "https://api.eia.gov/v2/"
CATALOG_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "eiax" / "catalog"
)
ROUTES_OUTPUT = CATALOG_DIR / "routes.parquet"
NODES_OUTPUT = CATALOG_DIR / "nodes.parquet"
FACET_VALUES_OUTPUT = CATALOG_DIR / "facet_values.parquet"

ROUTES_SCHEMA = {
    "route": pl.Utf8,
    "metadata_path": pl.Utf8,
    "id": pl.Utf8,
    "name": pl.Utf8,
    "description": pl.Utf8,
    "default_frequency": pl.Utf8,
    "default_date_format": pl.Utf8,
    "start_period": pl.Utf8,
    "end_period": pl.Utf8,
    "facets": pl.Utf8,
    "frequencies": pl.Utf8,
    "columns": pl.Utf8,
    "sample_columns": pl.Utf8,
}
NODES_SCHEMA = {
    "path": pl.Utf8,
    "parent": pl.Utf8,
    "name": pl.Utf8,
    "description": pl.Utf8,
    "is_leaf": pl.Boolean,
}
FACET_VALUES_SCHEMA = {
    "metadata_path": pl.Utf8,
    "route": pl.Utf8,
    "facet_id": pl.Utf8,
    "value_id": pl.Utf8,
    "value_name": pl.Utf8,
}


def _empty_routes() -> pl.DataFrame:
    return pl.DataFrame(schema=ROUTES_SCHEMA)


def _empty_nodes() -> pl.DataFrame:
    return pl.DataFrame(schema=NODES_SCHEMA)


def _empty_facet_values() -> pl.DataFrame:
    return pl.DataFrame(schema=FACET_VALUES_SCHEMA)


def _load_existing(path: Path, empty: pl.DataFrame) -> pl.DataFrame:
    if path.is_file() and path.stat().st_size > 100:
        return pl.read_parquet(path)
    return empty


def _under_root(path_col: pl.Expr, root: str) -> pl.Expr:
    return path_col.eq(root) | path_col.str.starts_with(f"{root}/")


def _merge_root(
    root: str,
    existing_routes: pl.DataFrame,
    existing_nodes: pl.DataFrame,
    existing_facets: pl.DataFrame,
    new_routes: pl.DataFrame,
    new_nodes: pl.DataFrame,
    new_facets: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    routes = pl.concat(
        [
            existing_routes.filter(~_under_root(pl.col("metadata_path"), root)),
            new_routes,
        ],
        how="vertical_relaxed",
    ).sort("route")
    nodes = pl.concat(
        [
            existing_nodes.filter(~_under_root(pl.col("path"), root)),
            new_nodes,
        ],
        how="vertical_relaxed",
    ).sort("path")
    facets = pl.concat(
        [
            existing_facets.filter(~_under_root(pl.col("metadata_path"), root)),
            new_facets,
        ],
        how="vertical_relaxed",
    ).sort("metadata_path", "facet_id", "value_id")
    return routes, nodes, facets


def _write_catalog(
    routes: pl.DataFrame,
    nodes: pl.DataFrame,
    facets: pl.DataFrame,
    *,
    routes_output: Path,
    nodes_output: Path,
    facet_values_output: Path,
) -> None:
    routes_output.parent.mkdir(parents=True, exist_ok=True)
    routes.write_parquet(routes_output, compression="zstd")
    nodes.write_parquet(nodes_output, compression="zstd")
    facets.write_parquet(facet_values_output, compression="zstd")


def list_roots(*, api_key: str) -> list[str]:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        root = client.get("", params={"api_key": api_key}, timeout=30.0).json()[
            "response"
        ]
    return [str(c["id"]) for c in root.get("routes") or []]


def _columns_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    cols: list[dict[str, Any]] = [
        {"name": "period", "role": "period", "alias": "Period", "units": None},
    ]
    for facet in meta.get("facets") or []:
        cols.append(
            {
                "name": facet["id"],
                "role": "facet",
                "alias": facet.get("description"),
                "units": None,
            }
        )
    data = meta.get("data") or {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "name" in item:
                cols.append(
                    {
                        "name": item["name"],
                        "role": "measure",
                        "alias": item.get("alias"),
                        "units": item.get("units"),
                        "aggregation_method": item.get("aggregation-method"),
                    }
                )
    else:
        for name, info in data.items():
            if not isinstance(info, dict):
                continue
            cols.append(
                {
                    "name": name,
                    "role": "measure",
                    "alias": info.get("alias"),
                    "units": info.get("units"),
                    "aggregation_method": info.get("aggregation-method"),
                }
            )
    return cols


def _row_from_meta(path: str, meta: dict[str, Any]) -> dict[str, Any]:
    route = f"{path.rstrip('/')}/data"
    return {
        "route": route,
        "metadata_path": path.rstrip("/"),
        "id": meta.get("id", ""),
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "default_frequency": meta.get("defaultFrequency", ""),
        "default_date_format": meta.get("defaultDateFormat", ""),
        "start_period": meta.get("startPeriod", ""),
        "end_period": meta.get("endPeriod", ""),
        "facets": json.dumps(meta.get("facets") or []),
        "frequencies": json.dumps(meta.get("frequency") or []),
        "columns": json.dumps(_columns_from_meta(meta)),
        "sample_columns": "[]",
    }


def _node_row(path: str, parent: str, meta: dict[str, Any], *, is_leaf: bool) -> dict[str, Any]:
    return {
        "path": path.rstrip("/"),
        "parent": parent,
        "name": str(meta.get("name") or ""),
        "description": str(meta.get("description") or ""),
        "is_leaf": is_leaf,
    }


def _fetch_facet_values(
    client: httpx.Client,
    api_key: str,
    metadata_path: str,
    route: str,
    facet_id: str,
    *,
    rate: float,
    errors: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    offset = 0
    page_size = 5000
    while True:
        time.sleep(1.0 / rate)
        try:
            resp = client.get(
                f"{metadata_path}/facet/{facet_id}",
                params={"api_key": api_key, "length": page_size, "offset": offset},
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            errors.append(f"{metadata_path}/facet/{facet_id}: {exc}")
            break
        if resp.status_code != 200:
            errors.append(f"{metadata_path}/facet/{facet_id}: HTTP {resp.status_code}")
            break
        payload = resp.json()
        block = payload.get("response") or payload
        facets = block.get("facets") or []
        added = 0
        for item in facets:
            value_id = str(item.get("id", ""))
            if value_id in seen:
                continue
            seen.add(value_id)
            added += 1
            rows.append(
                {
                    "metadata_path": metadata_path,
                    "route": route,
                    "facet_id": facet_id,
                    "value_id": value_id,
                    "value_name": str(item.get("name", "")),
                }
            )
        total = block.get("totalFacets")
        if not facets or added == 0:
            break
        if total is not None and len(rows) >= int(total):
            break
        if len(facets) < page_size:
            break
        offset += len(facets)
    return rows


def _walk(
    client: httpx.Client,
    api_key: str,
    path: str,
    route_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    facet_rows: list[dict[str, str]],
    *,
    parent: str,
    rate: float,
    errors: list[str],
) -> None:
    time.sleep(1.0 / rate)
    try:
        resp = client.get(path, params={"api_key": api_key}, timeout=30.0)
    except httpx.HTTPError as exc:
        errors.append(f"{path}: {exc}")
        return
    if resp.status_code != 200:
        errors.append(f"{path}: HTTP {resp.status_code}")
        return
    payload = resp.json()
    if "response" not in payload:
        errors.append(f"{path}: no response key ({list(payload.keys())})")
        return
    meta = payload["response"]
    child_routes = meta.get("routes")
    if child_routes:
        node_rows.append(_node_row(path, parent, meta, is_leaf=False))
        for child in child_routes:
            child_path = f"{path.rstrip('/')}/{child['id']}"
            _walk(
                client,
                api_key,
                child_path,
                route_rows,
                node_rows,
                facet_rows,
                parent=path.rstrip("/"),
                rate=rate,
                errors=errors,
            )
        return
    if "frequency" not in meta:
        return

    metadata_path = path.rstrip("/")
    node_rows.append(_node_row(path, parent, meta, is_leaf=True))
    row = _row_from_meta(path, meta)
    data_path = f"{metadata_path}/data"
    time.sleep(1.0 / rate)
    try:
        sample = client.get(
            data_path,
            params={"api_key": api_key, "length": 1, "offset": 0},
            timeout=30.0,
        )
        if sample.status_code == 200:
            sample_json = sample.json()
            data = sample_json.get("response", {}).get("data") or sample_json.get(
                "data"
            )
            if data:
                row["sample_columns"] = json.dumps(list(data[0].keys()))
    except httpx.HTTPError as exc:
        errors.append(f"{data_path}: sample fetch failed ({exc})")

    route_rows.append(row)

    for facet in meta.get("facets") or []:
        facet_id = facet.get("id")
        if not facet_id:
            continue
        facet_rows.extend(
            _fetch_facet_values(
                client,
                api_key,
                metadata_path,
                row["route"],
                str(facet_id),
                rate=rate,
                errors=errors,
            )
        )


def build_root(
    root: str,
    *,
    api_key: str,
    client: httpx.Client,
    rate_limit: float = 5.0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    route_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    facet_rows: list[dict[str, str]] = []
    errors: list[str] = []

    _walk(
        client,
        api_key,
        root,
        route_rows,
        node_rows,
        facet_rows,
        parent="",
        rate=rate_limit,
        errors=errors,
    )

    if not route_rows:
        msg = f"catalog walk for {root!r} produced zero routes"
        raise RuntimeError(msg)

    return (
        pl.from_dicts(route_rows),
        pl.from_dicts(node_rows),
        pl.from_dicts(facet_rows) if facet_rows else _empty_facet_values(),
        errors,
    )


def build_catalog(
    root: str | None = None,
    *,
    routes_output: Path = ROUTES_OUTPUT,
    nodes_output: Path = NODES_OUTPUT,
    facet_values_output: Path = FACET_VALUES_OUTPUT,
    rate_limit: float = 5.0,
) -> tuple[Path, Path, Path]:
    settings = get_settings()
    api_key = settings.require_api_key()

    existing_routes = _load_existing(routes_output, _empty_routes())
    existing_nodes = _load_existing(nodes_output, _empty_nodes())
    existing_facets = _load_existing(facet_values_output, _empty_facet_values())

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        new_routes, new_nodes, new_facets, errors = build_root(
            root,
            api_key=api_key,
            client=client,
            rate_limit=rate_limit,
        )

    routes, nodes, facets = _merge_root(
        root,
        existing_routes,
        existing_nodes,
        existing_facets,
        new_routes,
        new_nodes,
        new_facets,
    )
    _write_catalog(
        routes,
        nodes,
        facets,
        routes_output=routes_output,
        nodes_output=nodes_output,
        facet_values_output=facet_values_output,
    )

    print(
        f"{root}: +{new_routes.height} routes, +{new_nodes.height} nodes, "
        f"+{new_facets.height} facet values "
        f"(catalog totals: {routes.height} routes, {nodes.height} nodes, "
        f"{facets.height} facet values)",
        flush=True,
    )
    if errors:
        print(f"  {len(errors)} errors (first 3):", file=sys.stderr)
        for err in errors[:3]:
            print(f"    {err}", file=sys.stderr)
    return routes_output, nodes_output, facet_values_output


def build_all(
    *,
    routes_output: Path = ROUTES_OUTPUT,
    nodes_output: Path = NODES_OUTPUT,
    facet_values_output: Path = FACET_VALUES_OUTPUT,
    rate_limit: float = 5.0,
) -> None:
    settings = get_settings()
    api_key = settings.require_api_key()
    roots = list_roots(api_key=api_key)
    print(f"Building {len(roots)} top-level categories: {', '.join(roots)}", flush=True)
    for i, root in enumerate(roots, 1):
        print(f"[{i}/{len(roots)}] {root}", flush=True)
        build_catalog(
            root,
            routes_output=routes_output,
            nodes_output=nodes_output,
            facet_values_output=facet_values_output,
            rate_limit=rate_limit,
        )
    print("Done.", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build eiax offline catalog parquet files (batched by top-level route).",
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Top-level EIA route to build (e.g. coal, petroleum, natural-gas)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build every top-level category sequentially (merge after each)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List top-level route ids from the live API and exit",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=5.0,
        help="Max API requests per second (default: 5)",
    )
    args = parser.parse_args(argv)

    if args.list:
        settings = get_settings()
        api_key = settings.require_api_key()
        for root in list_roots(api_key=api_key):
            print(root)
        return

    if args.all:
        build_all(rate_limit=args.rate_limit)
        return

    if not args.root:
        parser.error("provide a top-level route id, or use --all / --list")

    build_catalog(args.root, rate_limit=args.rate_limit)


if __name__ == "__main__":
    main()
