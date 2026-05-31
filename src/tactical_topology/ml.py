"""Phase 4: ML models.

Three interpretable models built on Phase 2/3 network + telecom features:

  4.1  build_feature_matrix          - per match x team feature engineering
  4.2  cluster_teams_by_network      - KMeans tactical clustering (unsupervised)
  4.3  train_outcome_model           - RandomForest W/D/L prediction (supervised)
  4.4  detect_tactical_shifts        - change-point detection on dynamic networks

The feature matrix is built on the FULL season (Phase 4 is where we first use
all matches); the 5 ground-truth matches are reserved for unit tests.

Centrality FEATURES are numeric values (the score of the top node), never the
player-name strings. The name columns are kept only for display.
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import config, telecom
from .loader import load_pass_events, load_season_matches
from .network import build_passing_network, build_dynamic_network, network_summary

# Numeric feature columns used by every ML model (no names, no targets).
FEATURE_COLUMNS = [
    "density",
    "avg_clustering",
    "top_betweenness_value",
    "top_eigenvector_value",
    "top_pagerank_value",
    "n_nodes",
    "n_edges",
    "attacking_flow_efficiency",
    "team_tempo",
    "low_latency_ratio",
    "density_degradation",
    "completion_rate_drop",
]
TARGET_COLUMNS = ["goals_scored", "match_result"]
_DISPLAY_COLUMNS = [
    "top_betweenness_node", "top_eigenvector_node", "top_pagerank_node",
]


# --- 4.1 Feature engineering ----------------------------------------------


def _top_centrality_values(G: nx.DiGraph) -> dict:
    """Max centrality VALUE for each of the three measures (0.0 if no edges)."""
    if G.number_of_edges() == 0:
        return {"betweenness": 0.0, "eigenvector": 0.0, "pagerank": 0.0}
    betw = nx.betweenness_centrality(G)
    pr = nx.pagerank(G, weight="weight")
    try:
        ev = nx.eigenvector_centrality(G, weight="weight", max_iter=2000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        ev = pr
    return {
        "betweenness": max(betw.values()),
        "eigenvector": max(ev.values()),
        "pagerank": max(pr.values()),
    }


def compute_feature_row(pass_df: pd.DataFrame, team: str) -> dict:
    """Compute the numeric feature dict for one team from a pass DataFrame.

    ``pass_df`` may be full-match or a single-period subset; the caller decides.
    Telecom values that are undefined for a subset (e.g. completion drop with no
    under-pressure passes) come back as NaN and are imputed at matrix level.
    """
    G = build_passing_network(pass_df, team)
    summary = network_summary(G)
    centrality = _top_centrality_values(G)

    row = {
        "density": summary["density"],
        "avg_clustering": summary["avg_clustering"],
        "top_betweenness_value": centrality["betweenness"],
        "top_eigenvector_value": centrality["eigenvector"],
        "top_pagerank_value": centrality["pagerank"],
        "n_nodes": summary["n_nodes"],
        "n_edges": summary["n_edges"],
        "top_betweenness_node": summary["top_betweenness_node"],
        "top_eigenvector_node": summary["top_eigenvector_node"],
        "top_pagerank_node": summary["top_pagerank_node"],
    }

    # Telecom features; guard the ones that can be undefined on a subset.
    zone_graph = telecom.build_zone_graph(pass_df, team)
    row["attacking_flow_efficiency"] = telecom.compute_attacking_flow_efficiency(
        zone_graph
    )
    try:
        row["team_tempo"] = telecom.compute_team_tempo(pass_df, team)
        tempo = telecom.compute_player_tempo(pass_df, team)
        row["low_latency_ratio"] = float(
            np.average(tempo["low_latency_ratio"], weights=tempo["n_touches"])
        )
    except ValueError:
        row["team_tempo"] = np.nan
        row["low_latency_ratio"] = np.nan

    degradation = telecom.compute_pressure_degradation(pass_df, team)
    row["density_degradation"] = degradation["density_degradation"]
    row["completion_rate_drop"] = degradation["completion_rate_drop"]
    return row


def _match_targets(meta_row: pd.Series, team: str) -> dict:
    """goals_scored and W/D/L for ``team`` from match metadata."""
    home, away = meta_row["home_team"], meta_row["away_team"]
    hs, as_ = int(meta_row["home_score"]), int(meta_row["away_score"])
    if team == home:
        scored, conceded = hs, as_
    elif team == away:
        scored, conceded = as_, hs
    else:
        raise ValueError(f"Team '{team}' not in match {meta_row['match_id']}.")
    result = "W" if scored > conceded else ("L" if scored < conceded else "D")
    return {"goals_scored": scored, "match_result": result,
            "is_home": team == home}


def build_feature_matrix(
    competition_id: int = config.COMPETITION_ID,
    season_id: int = config.SEASON_ID,
    match_ids: list[int] | None = None,
    half: int | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Build the per match x team feature matrix for a season (or a subset).

    ``half=1`` restricts features to first-half events (used for the outcome
    model); ``half=None`` uses the full match (used for clustering). Targets are
    always full-match. NaN feature values are median-imputed so the result is
    ML-ready. If ``cache_path`` exists it is loaded instead of recomputed.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    matches = load_season_matches(competition_id, season_id)
    meta = matches.set_index("match_id")
    if match_ids is None:
        match_ids = [int(m) for m in matches["match_id"].tolist()]

    rows, skipped = [], []
    for match_id in tqdm(match_ids, desc=f"[Phase 4] features (half={half})"):
        try:
            passes = load_pass_events(match_id)
        except Exception as exc:  # noqa: BLE001 - skip unusable matches, keep going
            skipped.append((match_id, repr(exc)))
            continue
        if half is not None:
            passes = passes[passes["period"] == half]
        meta_row = meta.loc[match_id].copy()
        meta_row["match_id"] = match_id
        for team in passes["team"].unique():
            try:
                row = compute_feature_row(passes, team)
                row.update(_match_targets(meta_row, team))
                row["match_id"] = match_id
                row["team"] = team
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                skipped.append((match_id, team, repr(exc)))

    if skipped:
        print(f"[Phase 4] skipped {len(skipped)} match/team rows (data issues)")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Feature matrix is empty - no matches produced features.")

    # Imputation (PLAN_1.md 4.5 - feature matrix must have no NaN).
    for col in FEATURE_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    ordered = ["match_id", "team", "is_home"] + FEATURE_COLUMNS \
        + _DISPLAY_COLUMNS + TARGET_COLUMNS
    df = df[ordered]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


# --- 4.2 Tactical clustering ----------------------------------------------


def cluster_teams_by_network(
    feature_matrix: pd.DataFrame,
    n_clusters: int = 4,
    level: str = "team",
    random_state: int = 42,
) -> pd.DataFrame:
    """KMeans clustering on scaled network + telecom features.

    StandardScaler is applied before KMeans (mandatory - features are on very
    different scales). Uses only feature columns, never match outcomes.

    ``level="team"`` (default) clusters each team's *season-average* feature
    vector, giving one stable tactical label per team (this is what the function
    name implies and what the domain validation expects: a team consistently
    lands in one cluster). ``level="match"`` clusters individual match rows
    (returns match_id, team, cluster_label) for per-game style analysis.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in feature_matrix.columns]
    if missing:
        raise KeyError(f"cluster_teams_by_network: missing features {missing}.")
    if level not in {"team", "match"}:
        raise ValueError("level must be 'team' or 'match'.")

    if level == "team":
        agg = feature_matrix.groupby("team")[FEATURE_COLUMNS].mean()
        X = StandardScaler().fit_transform(agg)
        labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        ).fit_predict(X)
        return pd.DataFrame(
            {"team": agg.index, "cluster_label": labels.astype(int)}
        ).reset_index(drop=True)

    X = StandardScaler().fit_transform(feature_matrix[FEATURE_COLUMNS])
    labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(X)
    out = feature_matrix[["match_id", "team"]].copy()
    out["cluster_label"] = labels.astype(int)
    return out


# --- 4.3 Outcome prediction -----------------------------------------------


def train_outcome_model(
    feature_matrix: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Predict full-match W/D/L from (first-half) network features.

    RandomForest with stratified k-fold cross-validation. ``feature_matrix`` is
    expected to hold first-half features (built with ``half=1``) plus the
    full-match ``match_result`` target. Returns model, CV scores, weighted F1,
    confusion matrix, feature importances and the n_splits actually used.
    """
    for col in FEATURE_COLUMNS + ["match_result"]:
        if col not in feature_matrix.columns:
            raise KeyError(f"train_outcome_model: missing column '{col}'.")

    X = feature_matrix[FEATURE_COLUMNS]
    y = feature_matrix["match_result"]

    # k-fold cannot exceed the smallest class count.
    min_class = y.value_counts().min()
    used_splits = max(2, min(n_splits, int(min_class)))

    clf = RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight="balanced"
    )
    skf = StratifiedKFold(n_splits=used_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    labels = sorted(y.unique())

    clf.fit(X, y)  # fit on all data for feature importances
    importances = (
        pd.DataFrame({"feature": FEATURE_COLUMNS,
                      "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "model": clf,
        "cv_scores": cv_scores,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels),
        "confusion_labels": labels,
        "feature_importances": importances,
        "n_splits": used_splits,
    }


# --- 4.4 Tactical shift detection -----------------------------------------


def detect_tactical_shifts(
    dynamic_networks: list[tuple[float, nx.DiGraph]],
    threshold: float = 2.0,
) -> list[float]:
    """Change-point detection on a dynamic passing-network time series.

    For each window graph build the feature vector [density, top betweenness
    value, n_nodes], z-scale across the series, take the cosine distance between
    consecutive vectors, and flag timestamps where distance exceeds
    mean + threshold * std. Returns the list of flagged window-start timestamps.
    """
    if len(dynamic_networks) < 3:
        return []

    timestamps, vectors = [], []
    for ts, G in dynamic_networks:
        if G.number_of_edges() > 0:
            top_betw = max(nx.betweenness_centrality(G).values())
        else:
            top_betw = 0.0
        timestamps.append(float(ts))
        vectors.append([nx.density(G), top_betw, float(G.number_of_nodes())])

    X = np.asarray(vectors, dtype=float)
    # z-scale each feature so no single scale (e.g. n_nodes) dominates cosine.
    X = StandardScaler().fit_transform(X)

    distances = []
    for i in range(1, len(X)):
        a, b = X[i - 1], X[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        distances.append(1.0 - float(a @ b) / (na * nb) if na and nb else 0.0)
    distances = np.asarray(distances)

    if distances.std() == 0:
        return []
    cutoff = distances.mean() + threshold * distances.std()
    # distances[i-1] is the change arriving AT window i.
    return [timestamps[i + 1] for i, d in enumerate(distances) if d > cutoff]
