"""Compute Phase 3 telecom metrics for all ground-truth matches and save CSVs.

Writes one tidy team-level file per match to
data/processed/telecom_metrics_<match_id>.csv containing all four metric
families (resilience, max-flow, tempo, pressure degradation). Run with:
    python scripts/build_telecom_metrics.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.network import build_passing_network
from tactical_topology import telecom as T


def team_metrics(pass_df: pd.DataFrame, team: str) -> dict:
    G = build_passing_network(pass_df, team)
    resilience = T.compute_resilience(G)
    top = resilience.iloc[0]

    zone_graph = T.build_zone_graph(pass_df, team)
    flow = T.compute_max_flow(zone_graph)
    efficiency = T.compute_attacking_flow_efficiency(zone_graph)

    tempo = T.compute_player_tempo(pass_df, team)
    team_tempo = float(np.average(tempo["avg_touch_seconds"], weights=tempo["n_touches"]))
    team_low_latency = float(
        np.average(tempo["low_latency_ratio"], weights=tempo["n_touches"])
    )

    pressure = T.compute_pressure_degradation(pass_df, team)

    return {
        "team": team,
        "top_resilience_player": top["player"],
        "top_resilience_score": round(float(top["resilience_score"]), 4),
        "max_flow_value": flow["max_flow_value"],
        "min_cut_zones": "|".join(flow["min_cut_zones"]),
        "attacking_flow_efficiency": round(efficiency, 4),
        "team_tempo_seconds": round(team_tempo, 3),
        "team_low_latency_ratio": round(team_low_latency, 4),
        "completion_rate_drop_under_pressure": round(pressure["completion_rate_drop"], 4),
        "pressed_router": pressure["centrality_shift"]["pressed_top_betweenness"],
        "free_router": pressure["centrality_shift"]["free_top_betweenness"],
    }


def main() -> None:
    for mid in config.GROUND_TRUTH_MATCH_IDS:
        path = os.path.join(config.PROCESSED_DIR, f"passes_{mid}.parquet")
        df = pd.read_parquet(path)
        rows = [team_metrics(df, team) for team in df["team"].unique()]
        out = pd.DataFrame(rows)
        out.insert(0, "match_id", mid)
        csv_path = os.path.join(config.PROCESSED_DIR, f"telecom_metrics_{mid}.csv")
        out.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[Phase 3] wrote {csv_path} ({len(out)} teams)")


if __name__ == "__main__":
    main()
