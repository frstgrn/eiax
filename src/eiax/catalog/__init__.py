"""Offline route tree + curated recipes."""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import polars as pl

from eiax.errors import UnknownSeriesError

ROUTES_FILENAME = "routes.parquet"
NODES_FILENAME = "nodes.parquet"
FACET_VALUES_FILENAME = "facet_values.parquet"
BRANCH_NOTES_FILENAME = "branch_notes.json"


def _catalog_path(name: str) -> Path:
    return Path(str(resources.files("eiax.catalog") / name))


@lru_cache(maxsize=1)
def _routes_path() -> Path:
    return _catalog_path(ROUTES_FILENAME)


@lru_cache(maxsize=1)
def routes_table() -> pl.DataFrame:
    path = _routes_path()
    if not path.is_file():
        msg = (
            f"missing {ROUTES_FILENAME}; run "
            "`uv run python scripts/build_catalog.py` to generate the catalog"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


@lru_cache(maxsize=1)
def nodes_table() -> pl.DataFrame:
    path = _catalog_path(NODES_FILENAME)
    if not path.is_file():
        msg = (
            f"missing {NODES_FILENAME}; run "
            "`uv run python scripts/build_catalog.py` to generate the catalog"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


@lru_cache(maxsize=1)
def facet_values_table() -> pl.DataFrame:
    path = _catalog_path(FACET_VALUES_FILENAME)
    if not path.is_file():
        msg = (
            f"missing {FACET_VALUES_FILENAME}; run "
            "`uv run python scripts/build_catalog.py` to generate facet values"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


@lru_cache(maxsize=1)
def _branch_notes() -> dict[str, str]:
    path = _catalog_path(BRANCH_NOTES_FILENAME)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tree_path(path: str) -> str:
    cleaned = path.strip().strip("/")
    if cleaned.endswith("/data"):
        cleaned = cleaned[: -len("/data")]
    return cleaned


def _node_or_error(path: str) -> dict[str, object]:
    normalized = _normalize_tree_path(path)
    df = nodes_table().filter(pl.col("path") == normalized)
    if df.is_empty():
        raise UnknownSeriesError(f"Unknown tree path: {path!r}")
    return df.row(0, named=True)


@lru_cache(maxsize=1)
def _leaf_route_counts() -> pl.DataFrame:
    meta_paths = routes_table()["metadata_path"].to_list()
    rows: list[dict[str, object]] = []
    for node_path in nodes_table()["path"].to_list():
        prefix = f"{node_path}/"
        count = sum(
            1 for mp in meta_paths if mp == node_path or str(mp).startswith(prefix)
        )
        rows.append({"path": node_path, "leaf_routes": count})
    return pl.from_dicts(rows)


_CHILDREN_SCHEMA = {
    "path": pl.Utf8,
    "name": pl.Utf8,
    "description": pl.Utf8,
    "is_leaf": pl.Boolean,
    "leaf_routes": pl.Int64,
    "note": pl.Utf8,
}


def _format_children(df: pl.DataFrame) -> pl.DataFrame:
    notes = _branch_notes()
    out = df.join(_leaf_route_counts(), on="path", how="left").with_columns(
        pl.col("leaf_routes").fill_null(0)
    )
    note_col = [notes.get(str(row["path"]), "") for row in out.iter_rows(named=True)]
    return out.with_columns(pl.Series("note", note_col)).select(
        "path",
        "name",
        "description",
        "is_leaf",
        "leaf_routes",
        "note",
    )


def list_children(path: str = "") -> pl.DataFrame:
    """List direct children of a route-tree node (offline).

    ``path=""`` returns the 14 root categories (``aeo``, ``petroleum``, …).
    """
    parent = _normalize_tree_path(path)
    df = nodes_table().filter(pl.col("parent") == parent).sort("path")
    if df.is_empty() and parent:
        if nodes_table().filter(pl.col("path") == parent).is_empty():
            raise UnknownSeriesError(f"No children (or unknown path): {path!r}")
        return pl.DataFrame(schema=_CHILDREN_SCHEMA)
    return _format_children(df)


def describe_node(path: str) -> dict[str, object]:
    """Metadata for one tree node plus maintainer notes when available."""
    row = _node_or_error(path)
    normalized = str(row["path"])
    children = list_children(normalized)
    return {
        **row,
        "note": _branch_notes().get(normalized, ""),
        "child_count": children.height,
        "children": children.select("path", "name", "is_leaf").to_dicts(),
    }


def _route_suggestions(route: str, *, limit: int = 3) -> str:
    routes = routes_table()["route"].to_list()
    matches = difflib.get_close_matches(route, routes, n=limit, cutoff=0.4)
    if not matches:
        filtered = list_routes(route.split("/")[-1].replace("data", ""))
        matches = filtered["route"].to_list()[:limit]
    if matches:
        return f" Did you mean: {', '.join(matches)}?"
    return " Try eiax.list_routes() to browse available routes."


def facet_values(route: str, facet: str | None = None) -> pl.DataFrame:
    """Valid facet codes for a data route (offline, from catalog build)."""
    info = get_route(route)
    metadata_path = str(info["metadata_path"])
    df = facet_values_table().filter(pl.col("metadata_path") == metadata_path)
    if facet is not None:
        df = df.filter(pl.col("facet_id") == facet)
    if df.is_empty():
        msg = f"No facet values for route {route!r}"
        if facet:
            msg += f" facet {facet!r}"
        msg += "; rebuild catalog with scripts/build_catalog.py"
        raise UnknownSeriesError(msg)
    return df.select("facet_id", "value_id", "value_name").sort("facet_id", "value_id")


def list_routes(query: str | None = None) -> pl.DataFrame:
    df = routes_table()
    if query:
        q = query.lower()
        df = df.filter(
            pl.col("route").str.to_lowercase().str.contains(q, literal=True)
            | pl.col("name").str.to_lowercase().str.contains(q, literal=True)
            | pl.col("description").str.to_lowercase().str.contains(q, literal=True)
        )
    return df.select(
        "route",
        "name",
        "description",
        "default_frequency",
        "start_period",
        "end_period",
        "columns",
    )


def get_route(route: str) -> dict[str, object]:
    normalized = route.rstrip("/")
    if not normalized.endswith("/data"):
        normalized = f"{normalized}/data"
    df = routes_table().filter(pl.col("route") == normalized)
    if df.is_empty():
        raise UnknownSeriesError(
            f"Unknown route: {route!r}.{_route_suggestions(normalized)}"
        )
    row = df.row(0, named=True)
    return {
        **row,
        "facets": json.loads(row["facets"]),
        "frequencies": json.loads(row["frequencies"]),
        "columns": json.loads(row["columns"]),
        "sample_columns": json.loads(row["sample_columns"]),
    }


def column_dictionary(route: str) -> pl.DataFrame:
    """Data dictionary for a route (one row per column)."""
    info = get_route(route)
    cols = info["columns"]
    assert isinstance(cols, list)
    return pl.from_dicts(cols)


def help_route(route: str) -> str:
    """Human-readable route summary with a copy-paste fetch example."""
    info = get_route(route)
    route_id = str(info["route"])
    cols = info["columns"]
    assert isinstance(cols, list)
    measures = [c for c in cols if c.get("role") == "measure"]
    facets = info["facets"]
    assert isinstance(facets, list)
    meta_path = str(info.get("metadata_path", ""))
    note = _branch_notes().get(meta_path, "")

    measure_lines = [
        f"  - {c['name']}: {c.get('units', '?')} ({c.get('description', '')})"
        for c in measures
    ] or ["  - (no measure columns listed)"]
    facet_lines = [
        f"  - {f['id']}: run facet_values({route_id!r}, {f['id']!r}) for valid codes"
        for f in facets
    ] or ["  - (no facets)"]

    example_facets = (
        "{" + ", ".join(f'"{f["id"]}": "..."' for f in facets[:2]) + "}"
        if facets
        else "{}"
    )
    freq = info.get("default_frequency") or "monthly"

    lines = [
        f"Route: {route_id}",
        f"Name: {info.get('name') or '(unnamed)'}",
        f"Description: {info.get('description') or ''}",
        f"Coverage: {info.get('start_period')} → {info.get('end_period')}",
        f"Default frequency: {freq}",
        "",
        "Measure columns:",
        *measure_lines,
        "",
        "Facets:",
        *facet_lines,
    ]
    if note:
        lines.extend(["", f"Note: {note}"])
    lines.extend(
        [
            "",
            "Example:",
            "  import eiax",
            "",
            "  df = eiax.fetch(",
            f"      {route_id!r},",
            f"      facets={example_facets},",
            f"      frequency={freq!r},",
            '      start="2024-01-01",',
            '      end="2024-12-31",',
            "  )",
        ]
    )
    return "\n".join(lines)


def _score(text: str, tokens: list[str]) -> int:
    lower = text.lower()
    return sum(1 for t in tokens if t in lower)


_SEARCH_SCHEMA = {
    "kind": pl.Utf8,
    "id": pl.Utf8,
    "route": pl.Utf8,
    "name": pl.Utf8,
    "description": pl.Utf8,
    "frequency": pl.Utf8,
    "score": pl.Int64,
}


def search(query: str) -> pl.DataFrame:
    """Ranked offline search over route names and descriptions."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return pl.DataFrame(schema=_SEARCH_SCHEMA)

    route_rows: list[dict[str, object]] = []
    for row in routes_table().iter_rows(named=True):
        haystack = " ".join(
            str(row.get(k, "")) for k in ("route", "name", "description")
        )
        score = _score(haystack, tokens)
        if score:
            route_rows.append(
                {
                    "kind": "route",
                    "id": row["route"],
                    "route": row["route"],
                    "name": row["name"],
                    "description": row["description"],
                    "frequency": row["default_frequency"],
                    "score": score,
                }
            )

    if not route_rows:
        return pl.DataFrame(schema=_SEARCH_SCHEMA)
    return (
        pl.from_dicts(route_rows)
        .sort("score", descending=True)
        .select("kind", "id", "route", "name", "description", "frequency", "score")
    )
