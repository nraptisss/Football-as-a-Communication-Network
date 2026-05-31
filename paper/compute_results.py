"""Compute the exact numbers quoted in the paper and dump to results.json."""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.ml import (FEATURE_COLUMNS, cluster_teams_by_network,
                                   train_outcome_model, detect_tactical_shifts)
from tactical_topology.network import build_passing_network, build_dynamic_network
from tactical_topology.telecom import compute_resilience
from tactical_topology.loader import load_pass_events, load_match_events

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
    res["outcome"] = {
        "cv_accuracy_mean": round(om["cv_accuracy_mean"], 3),
        "cv_accuracy_std": round(float(np.std(om["cv_scores"])), 3),
        "cv_f1_weighted": round(om["cv_f1_weighted"], 3),
        "n_splits": om["n_splits"],
        "majority_baseline": round(float(fh["match_result"].value_counts(normalize=True).max()), 3),
        "top_features": om["feature_importances"].head(5)
            .assign(importance=lambda d: d["importance"].round(3))
            .to_dict("records"),
        "result_distribution": fh["match_result"].value_counts().to_dict(),
    }

    # Clustering (team level)
    clusters = cluster_teams_by_network(full, 4).set_index("team")["cluster_label"]
    members = {}
    for c in sorted(clusters.unique()):
        members[int(c)] = sorted(clusters[clusters == c].index.tolist())
    res["clusters"] = members

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

    # Shift detection validation (265839 Barcelona)
    p = load_pass_events(265839)
    dyn = build_dynamic_network(p[p["team"] == "Barcelona"], "Barcelona")
    shifts = detect_tactical_shifts(dyn, threshold=2.0)
    ev = load_match_events(265839)
    subs = list(ev[(ev["type_name"] == "Substitution") & (ev["team"] == "Barcelona")]["seconds"])
    goals = list(ev[(ev["type_name"] == "Shot") & (ev["shot_outcome"] == "Goal")]["seconds"])
    events = sorted(subs + goals)
    matched = sum(any(abs(e - s) <= 180 for e in events) for s in shifts)
    res["shifts"] = {
        "match": 265839, "team": "Barcelona",
        "detected": [round(s) for s in shifts],
        "events": [round(e) for e in events],
        "matched": matched, "total": len(shifts),
    }

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh_:
        json.dump(res, fh_, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
