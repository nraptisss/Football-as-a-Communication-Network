"""Tests for tactical_topology.network (Phase 2).

Reads the cached Phase 1 ground-truth parquet for one match and validates the
graph-construction contract.
"""

import os

import networkx as nx
import pandas as pd
import pytest

from tactical_topology import config
from tactical_topology.network import (
    build_dynamic_network,
    build_passing_network,
    compute_node_positions,
    network_summary,
)

MATCH_ID = config.GROUND_TRUTH_MATCH_IDS[0]
REQUIRED_SUMMARY_KEYS = {
    "density", "avg_clustering", "n_nodes", "n_edges",
    "top_betweenness_node", "top_eigenvector_node", "top_pagerank_node",
}


@pytest.fixture(scope="module")
def pass_df() -> pd.DataFrame:
    path = os.path.join(config.PROCESSED_DIR, f"passes_{MATCH_ID}.parquet")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def team(pass_df: pd.DataFrame) -> str:
    return pass_df["team"].unique()[0]


@pytest.fixture(scope="module")
def G(pass_df: pd.DataFrame, team: str) -> nx.DiGraph:
    return build_passing_network(pass_df, team)


def test_graph_is_directed(G: nx.DiGraph) -> None:
    assert isinstance(G, nx.DiGraph)


def test_no_self_loops(G: nx.DiGraph) -> None:
    assert nx.number_of_selfloops(G) == 0


def test_edge_weights_positive_integers(G: nx.DiGraph) -> None:
    for _, _, w in G.edges(data="weight"):
        assert isinstance(w, int) and w > 0


def test_node_count_in_range(G: nx.DiGraph) -> None:
    assert 5 <= G.number_of_nodes() <= 14


def test_node_positions_within_pitch_bounds(pass_df: pd.DataFrame, team: str) -> None:
    positions = compute_node_positions(pass_df, team)
    assert positions
    for _, (x, y) in positions.items():
        assert 0 <= x <= config.PITCH_LENGTH
        assert 0 <= y <= config.PITCH_WIDTH


def test_network_summary_has_all_keys(G: nx.DiGraph) -> None:
    summary = network_summary(G)
    assert set(summary.keys()) == REQUIRED_SUMMARY_KEYS


def test_min_passes_filters_weak_edges(pass_df: pd.DataFrame, team: str) -> None:
    g1 = build_passing_network(pass_df, team, min_passes=1)
    g5 = build_passing_network(pass_df, team, min_passes=5)
    assert g5.number_of_edges() <= g1.number_of_edges()
    assert all(w >= 5 for _, _, w in g5.edges(data="weight"))


def test_dynamic_network_has_enough_steps(pass_df: pd.DataFrame, team: str) -> None:
    series = build_dynamic_network(pass_df, team)
    assert len(series) >= 10
    ts, graph = series[0]
    assert isinstance(ts, float) and isinstance(graph, nx.DiGraph)


def test_missing_column_raises_keyerror(team: str) -> None:
    bad = pd.DataFrame({"team": [team], "player_name": ["X"]})
    with pytest.raises(KeyError):
        build_passing_network(bad, team)


def test_absent_team_raises_valueerror(pass_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_passing_network(pass_df, "Nonexistent FC")
