"""Phase 3: telecom-theory metrics.

Applies communication-network theory to football passing networks:

  3.1.1  compute_resilience            - node-failure / fault-tolerance analysis
  3.1.2  build_zone_graph / compute_max_flow / compute_attacking_flow_efficiency
                                       - spatial max-flow / min-cut
  3.1.3  compute_player_tempo / compute_team_tempo
                                       - processing latency per node
  3.1.4  compute_pressure_degradation  - pressing as signal interference

Column contract (Phase 1): player_name, pass_recipient_name, pass_outcome_name,
pass_complete (bool), team, team_id, location ([x, y]), pass_end_location
([x, y]), seconds (continuous), period.

All functions operate on completed passes (``pass_complete == True``) only,
EXCEPT :func:`compute_pressure_degradation`, which needs incomplete passes too
to measure the completion-rate drop under pressure.

Note on ``seconds``: it is continuous but built on a fixed 45-min period offset,
so it must not be used to compare across the half-time boundary. Tempo deltas
are therefore filtered on BOTH ``delta <= max_touch_seconds`` AND same period.
"""

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .network import _build_graph, network_summary

# --- shared helpers --------------------------------------------------------

_RESILIENCE_METRICS = {"betweenness", "eigenvector", "pagerank", "degree"}


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{context}: missing required columns {missing}. "
            f"Got columns: {sorted(df.columns)}"
        )


def _team_completed(pass_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Completed passes for one team; raise ValueError if none."""
    sub = pass_df[(pass_df["team"] == team) & (pass_df["pass_complete"])].copy()
    if sub.empty:
        raise ValueError(f"Team '{team}' has no completed passes in this DataFrame.")
    return sub


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; constant series -> all zeros."""
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


# --- 3.1.1 Resilience ------------------------------------------------------


def compute_resilience(G, metric: str = "betweenness") -> pd.DataFrame:
    """Simulate removing each player and measure network degradation.

    Inspired by fault-tolerance analysis in telecom networks. For each node
    removal we recompute density, average clustering and the number of weakly
    connected components (fragmentation). ``metric`` selects a node-centrality
    signal (one of betweenness/eigenvector/pagerank/degree) folded into the
    final score alongside the three structural impacts.

    Returns one row per player with columns:
        player, density_drop, clustering_drop, fragmentation_increase,
        resilience_score
    where ``resilience_score`` in [0, 1] is the mean of the four min-max
    normalised impact signals (higher = more critical player).
    """
    if metric not in _RESILIENCE_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_RESILIENCE_METRICS)}, got '{metric}'."
        )
    if G.number_of_nodes() == 0:
        raise ValueError("compute_resilience received an empty graph.")

    base_summary = network_summary(G)
    base_density = base_summary["density"]
    base_clustering = base_summary["avg_clustering"]
    base_components = nx.number_weakly_connected_components(G)
    centrality = _node_centrality(G, metric)

    records = []
    for node in G.nodes():
        H = G.copy()
        H.remove_node(node)
        records.append(
            {
                "player": node,
                "density_drop": base_density - (
                    nx.density(H) if H.number_of_nodes() > 1 else 0.0
                ),
                "clustering_drop": base_clustering - (
                    nx.average_clustering(H) if H.number_of_nodes() > 0 else 0.0
                ),
                "fragmentation_increase": (
                    nx.number_weakly_connected_components(H) - base_components
                ),
                "centrality": centrality.get(node, 0.0),
            }
        )

    df = pd.DataFrame.from_records(records)
    components = ["density_drop", "clustering_drop", "fragmentation_increase",
                 "centrality"]
    norm = pd.concat({c: _minmax(df[c]) for c in components}, axis=1)
    df["resilience_score"] = norm.mean(axis=1)
    df = df.drop(columns=["centrality"])
    return df.sort_values("resilience_score", ascending=False).reset_index(drop=True)


def _node_centrality(G, metric: str) -> dict:
    if metric == "betweenness":
        return nx.betweenness_centrality(G)
    if metric == "pagerank":
        return nx.pagerank(G, weight="weight")
    if metric == "degree":
        return dict(G.degree(weight="weight"))
    # eigenvector
    try:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=2000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, weight="weight")


# --- 3.1.2 Flow analysis ---------------------------------------------------

_SUPER_SOURCE = "__SOURCE__"
_SUPER_SINK = "__SINK__"


def _zone_label(ix: int, iy: int) -> str:
    return f"x{ix}y{iy}"


def _zone_of(x: float, y: float, x_edges: np.ndarray, y_edges: np.ndarray) -> str:
    # np.digitize returns 1..len(edges)-1 for interior; clip to valid band index.
    ix = int(np.clip(np.digitize(x, x_edges[1:-1]), 0, len(x_edges) - 2))
    iy = int(np.clip(np.digitize(y, y_edges[1:-1]), 0, len(y_edges) - 2))
    return _zone_label(ix, iy)


def build_zone_graph(
    pass_df: pd.DataFrame,
    team: str,
    n_zones_x: int = 5,
    n_zones_y: int = 3,
) -> "object":
    """Build a zone-level directed graph of completed-pass flow.

    The pitch is split into ``n_zones_x`` x ``n_zones_y`` zones (default 5x3).
    Nodes = zones (label ``x{ix}y{iy}``, ix increasing toward goal). Edge weight
    and capacity = number of completed passes from zone A to zone B. Origin uses
    ``location``, destination uses ``pass_end_location`` (both ``[x, y]`` lists).
    Same-zone passes become self-loops and are ignored by max-flow.
    """
    _require_columns(
        pass_df,
        ["team", "pass_complete", "location", "pass_end_location"],
        "build_zone_graph",
    )
    if n_zones_x < 1 or n_zones_y < 1:
        raise ValueError("n_zones_x and n_zones_y must be >= 1.")

    completed = _team_completed(pass_df, team).dropna(
        subset=["location", "pass_end_location"]
    )
    x_edges = np.linspace(0, config.PITCH_LENGTH, n_zones_x + 1)
    y_edges = np.linspace(0, config.PITCH_WIDTH, n_zones_y + 1)

    src_zones = [_zone_of(loc[0], loc[1], x_edges, y_edges)
                 for loc in completed["location"]]
    dst_zones = [_zone_of(loc[0], loc[1], x_edges, y_edges)
                 for loc in completed["pass_end_location"]]

    pairs = pd.DataFrame({"src": src_zones, "dst": dst_zones})
    counts = pairs.groupby(["src", "dst"]).size()

    G = nx.DiGraph()
    G.add_nodes_from(_zone_label(ix, iy)
                     for ix in range(n_zones_x) for iy in range(n_zones_y))
    G.graph["n_zones_x"] = n_zones_x
    G.graph["n_zones_y"] = n_zones_y
    for (src, dst), count in counts.items():
        w = int(count)
        G.add_edge(src, dst, weight=w, capacity=w)
    return G


def _band_zones(G, band: str) -> list[str]:
    """Return zones in the defensive ('def') or attacking ('att') third by ix."""
    nx_x = G.graph["n_zones_x"]
    if band == "def":
        ix_target = 0  # x-band 0-24 with default 5 zones
    elif band == "att":
        ix_target = nx_x - 1  # x-band 96-120
    else:
        raise ValueError("band must be 'def' or 'att'.")
    return [n for n in G.nodes()
            if not n.startswith("__") and int(n[1:].split("y")[0]) == ix_target]


def compute_max_flow(zone_graph) -> dict:
    """Maximum flow from the defensive third (source) to the attacking third (sink).

    A super-source feeds all defensive-third zones and a super-sink drains all
    attacking-third zones (high capacity), so multi-zone source/sink is handled.

    Returns ``{'max_flow_value': float, 'min_cut_zones': list[str],
    'flow_dict': dict}``.
    """
    src_zones = _band_zones(zone_graph, "def")
    sink_zones = _band_zones(zone_graph, "att")
    if not src_zones or not sink_zones:
        raise ValueError("Zone graph lacks defensive- or attacking-third zones.")

    H = zone_graph.copy()
    big = sum(d["capacity"] for _, _, d in zone_graph.edges(data=True)) + 1
    for z in src_zones:
        H.add_edge(_SUPER_SOURCE, z, capacity=big)
    for z in sink_zones:
        H.add_edge(z, _SUPER_SINK, capacity=big)

    flow_value, flow_dict = nx.maximum_flow(
        H, _SUPER_SOURCE, _SUPER_SINK, capacity="capacity"
    )
    _, (reachable, non_reachable) = nx.minimum_cut(
        H, _SUPER_SOURCE, _SUPER_SINK, capacity="capacity"
    )
    cut_edges = _cut_edges(H, reachable, non_reachable)
    # Bottleneck = real zones on either side of a saturated cut edge.
    min_cut_zones = sorted({
        z for u, v in cut_edges for z in (u, v) if not z.startswith("__")
    })
    # Tighter, more interpretable: the source-side zones the ball must funnel
    # out of (the last zones before the cut) - a clean band for visualisation.
    min_cut_source_frontier = sorted({
        u for u, v in cut_edges if not u.startswith("__") and not v.startswith("__")
    })

    # Strip super-nodes from the reported flow dict.
    clean_flow = {
        u: {v: f for v, f in flows.items() if not v.startswith("__")}
        for u, flows in flow_dict.items() if not u.startswith("__")
    }
    return {
        "max_flow_value": float(flow_value),
        "min_cut_zones": min_cut_zones,
        "min_cut_source_frontier": min_cut_source_frontier,
        "flow_dict": clean_flow,
    }


def _cut_edges(G, reachable: set, non_reachable: set) -> list[tuple]:
    return [(u, v) for u in reachable for v in G.successors(u)
            if v in non_reachable]


def compute_attacking_flow_efficiency(zone_graph) -> float:
    """Ratio of max-flow value to total completed passes in the zone graph.

    Higher = more of the team's passing translates into forward progression
    capacity from the defensive to the attacking third.
    """
    total = sum(d["weight"] for _, _, d in zone_graph.edges(data=True))
    if total == 0:
        raise ValueError("Zone graph has no passes; cannot compute efficiency.")
    return compute_max_flow(zone_graph)["max_flow_value"] / total


# --- 3.1.3 Tempo (latency) -------------------------------------------------


def compute_player_tempo(
    pass_df: pd.DataFrame,
    team: str,
    max_touch_seconds: float = 30.0,
    low_latency_seconds: float = 2.0,
) -> pd.DataFrame:
    """Per-player 'touch duration': time from receiving a pass to the next pass.

    Analogous to processing latency at a network node. For each completed pass a
    player receives, we find their next completed pass in the SAME period and
    take the delta in ``seconds``. Deltas are kept only if
    ``delta <= max_touch_seconds`` AND same period (so the half-time gap in the
    continuous ``seconds`` clock cannot create false long deltas).

    Returns columns: player, avg_touch_seconds, median_touch_seconds, n_touches,
    low_latency_ratio (fraction of touches under ``low_latency_seconds``).
    """
    _require_columns(
        pass_df,
        ["team", "player_name", "pass_recipient_name", "pass_complete",
         "seconds", "period"],
        "compute_player_tempo",
    )
    completed = _team_completed(pass_df, team)

    records = []
    for player, made in completed.groupby("player_name"):
        # Times this player received a completed pass, by period.
        received = completed[completed["pass_recipient_name"] == player]
        if received.empty:
            continue
        deltas: list[float] = []
        for period, made_p in made.groupby("period"):
            made_times = np.sort(made_p["seconds"].to_numpy())
            recv_times = received.loc[received["period"] == period, "seconds"]
            for t_recv in recv_times.to_numpy():
                # First pass this player makes strictly after receiving.
                idx = np.searchsorted(made_times, t_recv, side="right")
                if idx >= len(made_times):
                    continue
                delta = made_times[idx] - t_recv
                if 0 <= delta <= max_touch_seconds:
                    deltas.append(float(delta))
        if not deltas:
            continue
        arr = np.array(deltas)
        records.append(
            {
                "player": player,
                "avg_touch_seconds": float(arr.mean()),
                "median_touch_seconds": float(np.median(arr)),
                "n_touches": int(arr.size),
                "low_latency_ratio": float((arr < low_latency_seconds).mean()),
            }
        )

    if not records:
        raise ValueError(f"No touch sequences found for team '{team}'.")
    return pd.DataFrame.from_records(records).sort_values(
        "n_touches", ascending=False
    ).reset_index(drop=True)


def compute_team_tempo(
    pass_df: pd.DataFrame,
    team: str,
    max_touch_seconds: float = 30.0,
) -> float:
    """Team average touch duration, weighted by each player's touch count.

    Lower = faster tempo (less time for the opponent to reorganise).
    """
    player_tempo = compute_player_tempo(
        pass_df, team, max_touch_seconds=max_touch_seconds
    )
    weights = player_tempo["n_touches"]
    return float(np.average(player_tempo["avg_touch_seconds"], weights=weights))


# --- 3.1.4 Pressing as interference ---------------------------------------


def compute_pressure_degradation(pass_df: pd.DataFrame, team: str) -> dict:
    """Measure how being under pressure degrades a team's passing network.

    Splits the team's passes into under-pressure and free sub-sets, builds a
    completed-pass network for each (``network_summary``), and compares them.
    Completion rate uses ALL passes (complete + incomplete) in each subset, so
    incomplete passes are required here.

    Returns:
        free_network, pressed_network  - network_summary dicts
        density_degradation            - free density - pressed density
        completion_rate_drop           - free completion - pressed completion
        centrality_shift               - top betweenness node free vs pressed
    """
    _require_columns(
        pass_df,
        ["team", "player_name", "pass_recipient_name", "pass_complete",
         "under_pressure"],
        "compute_pressure_degradation",
    )
    team_df = pass_df[pass_df["team"] == team]
    if team_df.empty:
        raise ValueError(f"Team '{team}' has no passes in this DataFrame.")

    free_df = team_df[~team_df["under_pressure"]]
    pressed_df = team_df[team_df["under_pressure"]]

    free_summary = network_summary(_build_graph(free_df, min_passes=1))
    pressed_summary = network_summary(_build_graph(pressed_df, min_passes=1))

    free_completion = float(free_df["pass_complete"].mean()) if len(free_df) else float("nan")
    pressed_completion = (
        float(pressed_df["pass_complete"].mean()) if len(pressed_df) else float("nan")
    )

    return {
        "free_network": free_summary,
        "pressed_network": pressed_summary,
        "density_degradation": free_summary["density"] - pressed_summary["density"],
        "completion_rate_drop": free_completion - pressed_completion,
        "centrality_shift": {
            "free_top_betweenness": free_summary["top_betweenness_node"],
            "pressed_top_betweenness": pressed_summary["top_betweenness_node"],
            "router_changed": (
                free_summary["top_betweenness_node"]
                != pressed_summary["top_betweenness_node"]
            ),
        },
    }
