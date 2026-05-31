"""Phase 2: graph construction.

Builds weighted directed passing networks (static and dynamic) for a single
team in a single match from cleaned Phase 1 pass events.

Column contract (from Phase 1):
    player_name, pass_recipient_name, pass_outcome_name, pass_complete (bool),
    team, team_id, location ([x, y] list), seconds (continuous across periods).

Graph construction rules (PLAN_1.md 2.2):
  1. Node = player_name (string).
  2. Edge direction: passer (player_name) -> receiver (pass_recipient_name).
  3. Edge weight = count of *completed* passes between that ordered pair.
  4. Self-loops not allowed (passer == receiver dropped).
  5. Substitutes are kept as nodes; limited involvement is reflected naturally.
  6. Coordinates used as-is from StatsBomb (0-120 length, 0-80 width).

Note: only rows with ``pass_complete == True`` contribute edge weight in the
static network. Incomplete passes remain in the DataFrame (they matter for
Phase 3 resilience analysis) but are not counted here.
"""

import networkx as nx
import pandas as pd

from . import config

_MIN_NODES = 5  # fewer than this is not a meaningful team network (fail loudly)


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise KeyError if any required column is missing (data contract)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{context}: missing required columns {missing}. "
            f"Got columns: {sorted(df.columns)}"
        )


def _team_frame(pass_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return rows for ``team``; raise ValueError if the team is absent."""
    sub = pass_df[pass_df["team"] == team]
    if sub.empty:
        present = sorted(pass_df["team"].unique())
        raise ValueError(
            f"Team '{team}' has no pass events in this DataFrame. "
            f"Teams present: {present}"
        )
    return sub


def _build_graph(team_passes: pd.DataFrame, min_passes: int) -> nx.DiGraph:
    """Build a weighted DiGraph from one team's pass rows (no node-count guard).

    Nodes = every player who made or received a completed pass. Edges =
    ordered passer->receiver pairs with completed-pass count >= ``min_passes``.
    """
    completed = team_passes[team_passes["pass_complete"]].copy()
    # Rule 4 - drop self-loops (passer == receiver).
    completed = completed[
        completed["player_name"] != completed["pass_recipient_name"]
    ]
    completed = completed.dropna(subset=["player_name", "pass_recipient_name"])

    G = nx.DiGraph()
    # Rule 5 - add every involved player as a node (subs included), so a player
    # whose edges all fall below min_passes still appears (isolated).
    nodes = set(completed["player_name"]) | set(completed["pass_recipient_name"])
    G.add_nodes_from(nodes)

    # Rule 3 - edge weight = count of completed passes for the ordered pair.
    pair_counts = (
        completed.groupby(["player_name", "pass_recipient_name"]).size()
    )
    for (passer, receiver), count in pair_counts.items():
        weight = int(count)
        if weight >= min_passes:
            G.add_edge(passer, receiver, weight=weight)
    return G


def build_passing_network(
    pass_df: pd.DataFrame,
    team: str,
    min_passes: int = 2,
) -> nx.DiGraph:
    """Build a static passing network for one team in one match.

    Nodes = players. Edges = passer -> receiver. Edge weight = number of
    completed passes between that pair. Only edges with >= ``min_passes`` are
    included (noise reduction). Raises ValueError if the resulting network has
    fewer than 5 nodes (PLAN_1.md cross-phase rule 4 - fail loudly).
    """
    _require_columns(
        pass_df,
        ["team", "player_name", "pass_recipient_name", "pass_complete"],
        "build_passing_network",
    )
    team_passes = _team_frame(pass_df, team)
    G = _build_graph(team_passes, min_passes=min_passes)

    if G.number_of_nodes() < _MIN_NODES:
        raise ValueError(
            f"Team '{team}' produced only {G.number_of_nodes()} nodes "
            f"(< {_MIN_NODES}); not a usable passing network."
        )
    return G


def build_dynamic_network(
    pass_df: pd.DataFrame,
    team: str,
    window_seconds: float = 300.0,
    step_seconds: float = 60.0,
    min_passes: int = 1,
) -> list[tuple[float, nx.DiGraph]]:
    """Build a time series of passing networks using a sliding window.

    Returns a list of ``(window_start_seconds, graph)`` tuples. Default:
    5-minute windows sliding every 1 minute. ``min_passes`` defaults to 1
    because per-window pass counts are small. The node-count guard is not
    applied per window (early windows are naturally sparse).
    """
    _require_columns(
        pass_df,
        ["team", "player_name", "pass_recipient_name", "pass_complete", "seconds"],
        "build_dynamic_network",
    )
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive.")

    team_passes = _team_frame(pass_df, team)
    t_start = float(team_passes["seconds"].min())
    t_end = float(team_passes["seconds"].max())

    series: list[tuple[float, nx.DiGraph]] = []
    start = t_start
    while start < t_end:
        window = team_passes[
            (team_passes["seconds"] >= start)
            & (team_passes["seconds"] < start + window_seconds)
        ]
        G = _build_graph(window, min_passes=min_passes)
        series.append((start, G))
        start += step_seconds
    return series


def compute_node_positions(
    pass_df: pd.DataFrame,
    team: str,
) -> dict[str, tuple[float, float]]:
    """Average pitch position (x, y) per player from pass *origin* locations.

    ``location`` is stored as a ``[x, y]`` list and is unpacked here. Returns
    ``{player_name: (avg_x, avg_y)}`` in the StatsBomb coordinate system.
    """
    _require_columns(
        pass_df, ["team", "player_name", "location"], "compute_node_positions"
    )
    team_passes = _team_frame(pass_df, team).dropna(subset=["location", "player_name"])

    xs = team_passes["location"].map(lambda loc: float(loc[0]))
    ys = team_passes["location"].map(lambda loc: float(loc[1]))
    frame = pd.DataFrame(
        {"player_name": team_passes["player_name"].values, "x": xs.values, "y": ys.values}
    )
    means = frame.groupby("player_name")[["x", "y"]].mean()
    return {player: (float(row.x), float(row.y)) for player, row in means.iterrows()}


def network_summary(G: nx.DiGraph) -> dict:
    """Return standard graph metrics for a passing network.

    Keys: density, avg_clustering, n_nodes, n_edges,
          top_betweenness_node, top_eigenvector_node.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    summary = {
        "density": nx.density(G) if n_nodes > 1 else 0.0,
        "avg_clustering": nx.average_clustering(G) if n_nodes > 0 else 0.0,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "top_betweenness_node": None,
        "top_eigenvector_node": None,
    }
    if n_nodes == 0:
        return summary

    # Topological (unweighted) betweenness: how often a player lies on the
    # shortest path between others - the "router" role.
    betweenness = nx.betweenness_centrality(G)
    if betweenness:
        summary["top_betweenness_node"] = max(betweenness, key=betweenness.get)

    # Eigenvector centrality weighted by pass volume - who is connected to other
    # well-connected players. The iterative solver handles the (typically not
    # strongly connected) passing graph; the numpy solver raises on disconnected
    # graphs, so it is only a fallback on the largest strongly connected
    # component. Falls back to None if neither converges.
    if n_edges > 0:
        summary["top_eigenvector_node"] = _top_eigenvector_node(G)
    return summary


def _top_eigenvector_node(G: nx.DiGraph) -> str | None:
    """Return the highest weighted-eigenvector-centrality node, or None."""
    try:
        ev = nx.eigenvector_centrality(G, weight="weight", max_iter=2000, tol=1e-6)
        return max(ev, key=ev.get)
    except nx.PowerIterationFailedConvergence:
        pass
    # Fallback: largest strongly connected component (numpy solver needs it).
    try:
        largest = max(nx.strongly_connected_components(G), key=len)
        if len(largest) < 2:
            return None
        ev = nx.eigenvector_centrality_numpy(G.subgraph(largest), weight="weight")
        return max(ev, key=ev.get)
    except Exception:  # noqa: BLE001 - give up gracefully
        return None
