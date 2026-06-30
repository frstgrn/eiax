"""Unit tests for offline route catalog."""

from __future__ import annotations

import eiax
from eiax.catalog import (
    column_dictionary,
    describe_node,
    get_route,
    list_children,
    list_routes,
)


def test_routes_table_loaded() -> None:
    df = list_routes()
    assert df.height > 200


def test_get_route_columns() -> None:
    info = get_route("electricity/retail-sales/data")
    cols = info["columns"]
    assert isinstance(cols, list)
    names = {c["name"] for c in cols}
    assert "sales" in names
    assert "stateid" in names


def test_column_dictionary() -> None:
    dd = column_dictionary("electricity/retail-sales/data")
    assert dd.filter(dd["role"] == "measure").height >= 4


def test_search_finds_routes() -> None:
    hits = eiax.search("demand")
    assert hits.height > 0
    assert "route" in hits["kind"].to_list() or "recipe" in hits["kind"].to_list()


def test_list_children_roots() -> None:
    roots = list_children()
    assert roots.height >= 10
    assert "petroleum" in roots["path"].to_list()
    assert "natural-gas" in roots["path"].to_list()


def test_list_children_natural_gas_branch() -> None:
    pri = list_children("natural-gas")
    branches = pri["path"].to_list()
    assert "natural-gas/pri" in branches
    assert "natural-gas/stor" in branches


def test_describe_node() -> None:
    info = describe_node("natural-gas/pri")
    assert info["child_count"] >= 1
    assert any(c["path"].startswith("natural-gas/pri/") for c in info["children"])


def test_describe_node_leaf_path() -> None:
    info = describe_node("petroleum/pri/spt")
    assert info["child_count"] == 0
    assert info["children"] == []


def test_list_children_leaf_path_returns_empty() -> None:
    children = list_children("natural-gas/stor/wkly")
    assert children.height == 0


def test_help_route_nonempty() -> None:
    text = eiax.help_route("petroleum/pri/spt/data")
    assert "petroleum/pri/spt/data" in text
    assert "fetch(" in text


def test_facet_values_wti() -> None:
    fv = eiax.facet_values("petroleum/pri/spt/data", "series")
    ids = set(fv["value_id"].to_list())
    assert "RWTC" in ids
    assert "RBRTE" in ids
