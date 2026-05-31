"""Tests for tactical_topology.ml (Phase 4).

Feature matrices are built from the 5 ground-truth matches only (fast); the
full-season matrix is exercised by the Phase 4 build script / notebook.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tactical_topology import config
from tactical_topology.ml import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_matrix,
    cluster_teams_by_network,
    detect_tactical_shifts,
    train_outcome_model,
)
from tactical_topology.network import build_dynamic_network

GT = config.GROUND_TRUTH_MATCH_IDS


@pytest.fixture(scope="module")
def full_matrix() -> pd.DataFrame:
    return build_feature_matrix(match_ids=GT, half=None)


@pytest.fixture(scope="module")
def firsthalf_matrix() -> pd.DataFrame:
    return build_feature_matrix(match_ids=GT, half=1)


def test_feature_matrix_has_no_nan(full_matrix: pd.DataFrame) -> None:
    assert not full_matrix[FEATURE_COLUMNS + TARGET_COLUMNS].isna().any().any()


def test_feature_matrix_shape(full_matrix: pd.DataFrame) -> None:
    # 5 matches x 2 teams = 10 rows.
    assert len(full_matrix) == 2 * len(GT)
    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(full_matrix[col])


def test_centrality_features_are_values_not_names(full_matrix: pd.DataFrame) -> None:
    for col in ["top_betweenness_value", "top_eigenvector_value",
                "top_pagerank_value"]:
        assert pd.api.types.is_numeric_dtype(full_matrix[col])
        assert full_matrix[col].between(0, 1).all()


def test_team_level_cluster_labels_valid(full_matrix: pd.DataFrame) -> None:
    n_clusters = 3  # 6 unique ground-truth teams -> keep < n_teams
    result = cluster_teams_by_network(full_matrix, n_clusters=n_clusters)
    assert list(result.columns) == ["team", "cluster_label"]
    assert result["cluster_label"].dtype.kind in "iu"
    assert result["cluster_label"].between(0, n_clusters - 1).all()
    assert len(result) == full_matrix["team"].nunique()


def test_match_level_cluster_labels_valid(full_matrix: pd.DataFrame) -> None:
    n_clusters = 4
    result = cluster_teams_by_network(full_matrix, n_clusters=n_clusters, level="match")
    assert result["cluster_label"].between(0, n_clusters - 1).all()
    assert len(result) == len(full_matrix)


def test_cv_scores_length_matches_splits(firsthalf_matrix: pd.DataFrame) -> None:
    result = train_outcome_model(firsthalf_matrix, n_splits=2)
    assert len(result["cv_scores"]) == result["n_splits"]
    assert "feature_importances" in result
    assert len(result["feature_importances"]) == len(FEATURE_COLUMNS)


def test_detect_shifts_within_match_time(full_matrix: pd.DataFrame) -> None:
    df = pd.read_parquet(
        os.path.join(config.PROCESSED_DIR, f"passes_{GT[0]}.parquet")
    )
    team = df["team"].unique()[0]
    dyn = build_dynamic_network(df, team)
    input_ts = {ts for ts, _ in dyn}
    shifts = detect_tactical_shifts(dyn)
    assert isinstance(shifts, list)
    for t in shifts:
        assert 0 <= t <= 6000          # continuous match-seconds bound
        assert t in input_ts            # must be an actual window start
