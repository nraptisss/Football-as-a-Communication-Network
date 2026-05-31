"""Compute the exact numbers quoted in the paper and dump to results.json."""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.ml import (FEATURE_COLUMNS, cluster_teams_by_network,
                                   train_outcome_model, detect_tactical_shifts)
from tactical_topology.network import build_passing_network, build_dynamic_network
from tactical_topology.telecom import compute_resilience
from tactical_topology.loader import load_pass_events, load_match_events


def _cached_match_events(match_id: int) -> pd.DataFrame:
    """Load full match events, caching to parquet so reruns are offline.

    The pass-only parquet lacks Substitution/Shot rows that the shift-detection
    validation needs, so we cache the full event frame once.
    """
    path = os.path.join(config.PROCESSED_DIR, f"events_{match_id}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    ev = load_match_events(match_id)
    try:
        ev.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - caching is best-effort, never fatal
        pass
    return ev


def _outcome_permutation_p(fh: pd.DataFrame, observed_acc: float,
                           n_perm: int = 200, random_state: int = 1) -> dict:
    """Label-permutation null for the outcome model's CV accuracy.

    Shuffles the target many times and recomputes stratified-CV accuracy to
    establish how often a model with no real signal reaches ``observed_acc``.
    A small p-value means the first-half features carry genuine signal.
    """
    X = fh[FEATURE_COLUMNS]
    y = fh["match_result"].to_numpy()
    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        clf = RandomForestClassifier(n_estimators=120, random_state=1,
                                     class_weight="balanced")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        null.append(cross_val_score(clf, X, yp, cv=skf, scoring="accuracy").mean())
    null = np.asarray(null)
    p = float((np.sum(null >= observed_acc) + 1) / (len(null) + 1))
    return {
        "n_permutations": n_perm,
        "null_mean": round(float(null.mean()), 3),
        "null_std": round(float(null.std()), 3),
        "null_p95": round(float(np.percentile(null, 95)), 3),
        "permutation_p": round(p, 4),
    }


def _cluster_silhouettes(full: pd.DataFrame, k_values=(2, 3, 4, 5, 6)) -> dict:
    """Silhouette score per k on the team-level feature aggregate."""
    agg = full.groupby("team")[FEATURE_COLUMNS].mean()
    X = StandardScaler().fit_transform(agg)
    out = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        out[k] = round(float(silhouette_score(X, labels)), 3)
    return out

PROC = config.PROCESSED_DIR
HERE = os.path.dirname(__file__)


def main() -> None:
    full = pd.read_parquet(os.path.join(PROC, "feature_matrix.parquet"))
    fh = pd.read_parquet(os.path.join(PROC, "feature_matrix_firsthalf.parquet"))
    res = {}

    res["n_match_team_rows"] = int(len(full))
    res["n_matches"] = int(full["match_id"].nunique())
    res["n_teams"] = int(full["team"].nunique())

    # Outcome model
    om = train_outcome_model(fh, n_splits=5)
    n_classes = int(fh["match_result"].nunique())
    res["outcome"] = {
        "cv_accuracy_mean": round(om["cv_accuracy_mean"], 3),
        "cv_accuracy_std": round(float(np.std(om["cv_scores"])), 3),
        "cv_f1_weighted": round(om["cv_f1_weighted"], 3),
        "n_splits": om["n_splits"],
        "n_features": len(FEATURE_COLUMNS),
        "majority_baseline": round(float(fh["match_result"].value_counts(normalize=True).max()), 3),
        "random_baseline": round(1.0 / n_classes, 3),
        "top_features": om["feature_importances"].head(5)
            .assign(importance=lambda d: d["importance"].round(3))
            .to_dict("records"),
        "result_distribution": fh["match_result"].value_counts().to_dict(),
    }
    # Permutation null: is the (modest) lift over the majority baseline real?
    res["outcome"]["permutation"] = _outcome_permutation_p(fh, om["cv_accuracy_mean"])

    # Clustering (team level), with silhouette-based cluster-count diagnostics.
    sil = _cluster_silhouettes(full)
    best_k = max(sil, key=sil.get)
    clusters = cluster_teams_by_network(full, 4).set_index("team")["cluster_label"]
    members = {}
    for c in sorted(clusters.unique()):
        members[int(c)] = sorted(clusters[clusters == c].index.tolist())
    res["clusters"] = members
    res["clusters_quality"] = {
        "silhouette_by_k": sil,
        "best_k_by_silhouette": int(best_k),
        "reported_k": 4,
        "note": ("k=4 is reported for tactical granularity; the strongest "
                 "silhouette is at k=%d (%.3f), the natural possession-vs-rest "
                 "split." % (best_k, sil[best_k])),
    }

    # Pressure pooled completion
    frames = [pd.read_parquet(os.path.join(PROC, f"passes_{m}.parquet"))
              for m in config.GROUND_TRUTH_MATCH_IDS]
    allp = pd.concat(frames)
    g = allp.groupby("under_pressure")["pass_complete"].mean()
    res["pressure"] = {
        "free_completion": round(float(g[False]), 3),
        "pressed_completion": round(float(g[True]), 3),
        "drop_pp": round(float(g[False] - g[True]) * 100, 1),
    }

    # Resilience top-3 (El Clasico)
    clasico = pd.read_parquet(os.path.join(PROC, "passes_266424.parquet"))
    rdf = compute_resilience(build_passing_network(clasico, "Barcelona"))
    res["resilience_top3_clasico"] = rdf["player"].head(3).tolist()

    # Flow efficiency by team (mean over season)
    eff = full.groupby("team")["attacking_flow_efficiency"].mean().sort_values(ascending=False)
    res["flow_efficiency"] = {
        "top3": [(t, round(float(v), 4)) for t, v in eff.head(3).items()],
        "barcelona": round(float(eff.get("Barcelona", float("nan"))), 4),
        "atletico": round(float(eff.get("Atlético Madrid", float("nan"))), 4),
    }

    # Density (full season mean by a few teams)
    res["density_examples"] = {
        t: round(float(full[full["team"] == t]["density"].mean()), 3)
        for t in ["Barcelona", "Real Madrid", "Atlético Madrid", "Sevilla"]
        if t in set(full["team"])
    }

    # Shift detection validation (265839 Barcelona) - offline from cached data.
    cached_passes = os.path.join(PROC, "passes_265839.parquet")
    p = (pd.read_parquet(cached_passes) if os.path.exists(cached_passes)
         else load_pass_events(265839))
    dyn = build_dynamic_network(p[p["team"] == "Barcelona"], "Barcelona")
    shifts = detect_tactical_shifts(dyn, threshold=2.0)
    ev = _cached_match_events(265839)
    subs = list(ev[(ev["type_name"] == "Substitution") & (ev["team"] == "Barcelona")]["seconds"])
    goals = list(ev[(ev["type_name"] == "Shot") & (ev["shot_outcome"] == "Goal")]["seconds"])
    events = sorted(subs + goals)
    tol = 180
    matched = sum(any(abs(e - s) <= tol for e in events) for s in shifts)

    # Null model: how often would the SAME number of randomly-placed shifts match
    # this many events within +/-tol? With a +/-3 min window and several events a
    # large fraction of the match is "hit territory", so 4/4 is only meaningful
    # against this baseline.
    t_lo = float(min(ts for ts, _ in dyn))
    t_hi = float(max(ts for ts, _ in dyn))
    rng = np.random.default_rng(0)
    n_shifts = len(shifts)
    trials = 5000
    if n_shifts and events and t_hi > t_lo:
        ge = 0
        for _ in range(trials):
            rand_shifts = rng.uniform(t_lo, t_hi, size=n_shifts)
            m = sum(any(abs(e - s) <= tol for e in events) for s in rand_shifts)
            ge += (m >= matched)
        shift_p = round(float((ge + 1) / (trials + 1)), 4)
        rand_pts = rng.uniform(t_lo, t_hi, size=trials)
        per_shift_hit = round(
            float(np.mean([any(abs(e - x) <= tol for e in events) for x in rand_pts])), 3)
    else:
        shift_p, per_shift_hit = None, None

    res["shifts"] = {
        "match": 265839, "team": "Barcelona",
        "detected": [round(s) for s in shifts],
        "events": [round(e) for e in events],
        "matched": matched, "total": len(shifts),
        "tolerance_seconds": tol,
        "null_match_prob_per_random_shift": per_shift_hit,
        "permutation_p": shift_p,
    }

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh_:
        json.dump(res, fh_, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
