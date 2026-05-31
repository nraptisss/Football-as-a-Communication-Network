"""Tests for tactical_topology.telecom (Phase 3).

Reads the cached Phase 1 ground-truth parquet and validates the telecom-metric
contracts (mechanical correctness; football-sense checks are human-approval
exit criteria handled in notebook 03).
"""

import os

import pandas as pd
import pytest

from tactical_topology import config
from tactical_topology.network import build_passing_network
from tactical_topology import telecom as T

MATCH_ID = config.GROUND_TRUTH_MATCH_IDS[0]


@pytest.fixture(scope="module")
def pass_df() -> pd.DataFrame:
    path = os.path.join(config.PROCESSED_DIR, f"passes_{MATCH_ID}.parquet")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def team(pass_df: pd.DataFrame) -> str:
    return pass_df["team"].unique()[0]


@pytest.fixture(scope="module")
def G(pass_df: pd.DataFrame, team: str):
    return build_passing_network(pass_df, team)


# --- resilience ---


def test_resilience_one_row_per_node(G, pass_df, team) -> None:
    res = T.compute_resilience(G)
    assert len(res) == G.number_of_nodes()
    assert set(res["player"]) == set(G.nodes())


def test_resilience_score_in_unit_interval(G) -> None:
    res = T.compute_resilience(G)
    assert res["resilience_score"].between(0, 1).all()


def test_resilience_has_required_columns(G) -> None:
    res = T.compute_resilience(G)
    assert set(res.columns) == {
        "player", "density_drop", "clustering_drop",
        "fragmentation_increase", "resilience_score",
    }


def test_resilience_rejects_bad_metric(G) -> None:
    with pytest.raises(ValueError):
        T.compute_resilience(G, metric="not_a_metric")


# --- flow ---


def test_max_flow_positive(pass_df, team) -> None:
    zg = T.build_zone_graph(pass_df, team)
    result = T.compute_max_flow(zg)
    assert result["max_flow_value"] > 0
    assert isinstance(result["min_cut_zones"], list)


def test_flow_efficiency_non_negative(pass_df, team) -> None:
    zg = T.build_zone_graph(pass_df, team)
    assert T.compute_attacking_flow_efficiency(zg) >= 0


def test_zone_graph_node_count(pass_df, team) -> None:
    zg = T.build_zone_graph(pass_df, team, n_zones_x=5, n_zones_y=3)
    assert zg.number_of_nodes() == 15


# --- tempo ---


def test_player_tempo_no_negative_durations(pass_df, team) -> None:
    tempo = T.compute_player_tempo(pass_df, team)
    assert (tempo["avg_touch_seconds"] >= 0).all()
    assert (tempo["median_touch_seconds"] >= 0).all()
    assert (tempo["n_touches"] > 0).all()
    assert tempo["low_latency_ratio"].between(0, 1).all()


def test_team_tempo_is_positive_float(pass_df, team) -> None:
    tempo = T.compute_team_tempo(pass_df, team)
    assert isinstance(tempo, float) and tempo > 0


# --- pressure degradation ---


def test_pressure_degradation_returns_both_subnetworks(pass_df, team) -> None:
    deg = T.compute_pressure_degradation(pass_df, team)
    assert "free_network" in deg and "pressed_network" in deg
    assert "completion_rate_drop" in deg
    assert "centrality_shift" in deg
    # density_degradation was removed (sample-size confounded); ensure it is gone.
    assert "density_degradation" not in deg
