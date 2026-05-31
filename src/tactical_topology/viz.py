"""Phase 5: visualisations.

Publication-quality, blog-ready figures. Every figure has a title and a one-line
caption (``fig.text`` below the plot) written for a reader who has never seen the
project. Pitch plots use mplsoccer's ``Pitch(pitch_type='statsbomb')`` so the
120x80 StatsBomb coordinate system needs no rescaling.

Public API mirrors PLAN_1.md 5.1:
    plot_passing_network_on_pitch, plot_resilience_bar, plot_zone_flow_heatmap,
    plot_dynamic_network_metrics, plot_pressure_comparison, plot_cluster_scatter
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle
from mplsoccer import Pitch
from sklearn.decomposition import PCA

from . import config
from .ml import FEATURE_COLUMNS
from .telecom import compute_max_flow

# --- consistent palette + typography (defined once) -----------------------
NAVY = "#1a1a2e"     # pitch background
GOLD = "#f0c040"     # highlighted / key node
RED = "#e63946"      # min-cut zones, pressure-degraded metrics
BLUE = "#457b9d"     # normal network elements
GREY = "#cccccc"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

# Cluster names are derived from each cluster's own feature profile at plot time
# (see ``name_clusters_by_density``), NOT hardcoded by integer label. KMeans
# label integers are arbitrary and reshuffle whenever the data or seed changes,
# so a fixed {0: "Possession elite", ...} map silently mislabels the figure as
# soon as anything upstream changes. Naming by mean density keeps the legend
# correct by construction.
_TIER_COLORS = [BLUE, "#2a9d8f", "#e9c46a", RED, "#9b5de5", "#8d99ae"]
_SCATTER_ANNOTATE = ["Barcelona", "Real Madrid", "Atlético Madrid",
                     "Rayo Vallecano", "Villarreal"]


def name_clusters_by_density(team_features, labels):
    """Map each cluster id to a content-derived name and a stable colour.

    Clusters are ranked by their mean network ``density`` (a possession proxy):
    the densest is named 'possession-dominant', the sparsest 'direct /
    low-possession', the rest 'mid-block'. The name embeds the cluster's actual
    mean density so the legend is self-describing and cannot drift from the data.
    Returns ``(names, colors)`` dicts keyed by integer cluster id.

    ``team_features`` is the team-indexed feature frame; ``labels`` is an array
    aligned to ``team_features`` rows.
    """
    import pandas as pd

    df = pd.DataFrame({"density": team_features["density"].to_numpy(),
                       "c": np.asarray(labels)})
    centroid_density = df.groupby("c")["density"].mean()
    order = centroid_density.sort_values(ascending=False).index.tolist()
    n = len(order)
    names, colors = {}, {}
    for rank, c in enumerate(order):
        if rank == 0:
            tag = "possession-dominant"
        elif rank == n - 1:
            tag = "direct / low-possession"
        else:
            tag = "mid-block"
        names[int(c)] = f"C{int(c)}: {tag} (mean density {centroid_density[c]:.2f})"
        colors[int(c)] = _TIER_COLORS[rank % len(_TIER_COLORS)]
    return names, colors


def _save(fig, save_path: str | None) -> None:
    """Save a figure as 300-DPI PNG and SVG (same stem)."""
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    png = save_path if save_path.endswith(".png") else save_path + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png[:-4] + ".svg", bbox_inches="tight", facecolor=fig.get_facecolor())


def _caption(fig, text: str, color: str = "black") -> None:
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=10,
             style="italic", color=color, wrap=True)


# --- 5.1 figure 1: passing network on a pitch ------------------------------


def plot_passing_network_on_pitch(
    G: nx.DiGraph,
    node_positions: dict,
    team: str,
    match_label: str,
    highlight_node: str | None = None,
    save_path: str | None = None,
    pass_df=None,
):
    """Draw a passing network on a football pitch (mplsoccer, StatsBomb coords).

    Node size proportional to betweenness centrality; node colour gold for the
    highlighted player (or the top-betweenness 'router' if none is given), blue
    otherwise. Edge width proportional to pass count. If ``pass_df`` is provided,
    edge colour follows the per-pair completion rate (red->green); otherwise
    edges are shaded by pass volume.
    """
    betw = nx.betweenness_centrality(G)
    top_router = max(betw, key=betw.get) if betw else None
    gold_node = highlight_node or top_router

    pitch = Pitch(pitch_type="statsbomb", pitch_color=NAVY, line_color="#cfd8e3",
                  linewidth=1.2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(NAVY)

    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        if u not in node_positions or v not in node_positions:
            continue
        x0, y0 = node_positions[u]
        x1, y1 = node_positions[v]
        width = 0.6 + 4.5 * d["weight"] / max_w
        color, alpha = BLUE, 0.25 + 0.6 * d["weight"] / max_w
        if pass_df is not None:
            color = _edge_completion_color(pass_df, team, u, v)
            alpha = 0.7
        pitch.lines(x0, y0, x1, y1, lw=width, color=color, alpha=alpha,
                    zorder=1, ax=ax)

    max_b = max(betw.values()) if betw else 1.0
    for n in G.nodes():
        if n not in node_positions:
            continue
        x, y = node_positions[n]
        size = 250 + 2600 * (betw.get(n, 0.0) / max_b if max_b else 0)
        is_gold = n == gold_node
        pitch.scatter(x, y, s=size, ax=ax, zorder=3,
                      color=GOLD if is_gold else BLUE,
                      edgecolors="white", linewidth=1.5)
        ax.annotate(n.split()[0], (x, y), color="white", fontsize=9,
                    ha="center", va="center", zorder=4,
                    fontweight="bold" if is_gold else "normal")

    ax.set_title(f"{team} passing network — {match_label}",
                 color="white", fontsize=13, pad=12)
    fig.subplots_adjust(bottom=0.10)
    _caption(
        fig,
        "Players placed at their average pass location; circle size = how often "
        f"a player links others (betweenness), line width = passes. Gold = {gold_node}.",
        color="#dddddd",
    )
    _save(fig, save_path)
    return fig


def _edge_completion_color(pass_df, team, passer, receiver):
    """Red->green completion-rate colour for a passer->receiver pair."""
    sub = pass_df[(pass_df["team"] == team)
                  & (pass_df["player_name"] == passer)
                  & (pass_df["pass_recipient_name"] == receiver)]
    if sub.empty:
        return BLUE
    rate = float(sub["pass_complete"].mean())
    return plt.cm.RdYlGn(rate)


# --- 5.1 figure 2: resilience bar ------------------------------------------


def plot_resilience_bar(resilience_df, team: str, save_path: str | None = None,
                        n_annotate: int = 3):
    """Horizontal bar chart of resilience score per player (descending).

    The single most critical player is labelled 'Key Node'; the top
    ``n_annotate`` players are highlighted in gold.
    """
    df = resilience_df.sort_values("resilience_score", ascending=True)
    names = [p.split()[0] for p in df["player"]]
    colors = [GOLD if i >= len(df) - n_annotate else BLUE for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, df["resilience_score"], color=colors, edgecolor="black",
            linewidth=0.4)
    ax.set_xlabel("resilience score  (higher = more critical to the network)")
    ax.set_title(f"{team}: how critical is each player to the passing network?")
    top = df.iloc[-1]
    ax.annotate("Key Node", (top["resilience_score"], len(df) - 1),
                xytext=(-60, 0), textcoords="offset points", va="center",
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    fig.subplots_adjust(bottom=0.12)
    _caption(
        fig,
        "Each player is removed from the passing network in turn; the bar shows "
        "how much the network degrades without them. Gold = the 3 most critical.",
    )
    _save(fig, save_path)
    return fig


# --- 5.1 figure 3: zone-flow heatmap ---------------------------------------


def plot_zone_flow_heatmap(zone_graph, save_path: str | None = None,
                           ax=None, title: str | None = None):
    """Pitch heatmap of pass-flow intensity per zone, with min-cut zones in red.

    Zone intensity = total passes into + out of the zone. The defensive-to-
    attacking-third bottleneck (min-cut zones) is outlined in red. Pass ``ax`` to
    draw into an existing axis (e.g. for side-by-side comparison).
    """
    nx_x = zone_graph.graph["n_zones_x"]
    nx_y = zone_graph.graph["n_zones_y"]
    grid = np.zeros((nx_y, nx_x))
    for node in zone_graph.nodes():
        ix = int(node[1:].split("y")[0])
        iy = int(node.split("y")[1])
        grid[iy, ix] = (zone_graph.degree(node, weight="weight"))

    try:
        # Source-side frontier = the clean bottleneck band the ball funnels through.
        min_cut = set(compute_max_flow(zone_graph)["min_cut_source_frontier"])
    except ValueError:
        min_cut = set()

    standalone = ax is None
    if standalone:
        # Let mplsoccer own figure creation so it controls the (landscape) aspect.
        pitch = Pitch(pitch_type="statsbomb", pitch_color="white",
                      line_color="#333333", linewidth=1.2)
        fig, ax = pitch.draw(figsize=(12, 8))
    else:
        # Caller already drew the pitch on this axis (e.g. via pitch.draw(ncols=2)).
        fig = ax.figure

    # aspect='equal' preserves the true 120x80 pitch proportions (no stretching).
    ax.imshow(grid, extent=[0, config.PITCH_LENGTH, 0, config.PITCH_WIDTH],
              origin="lower", aspect="equal", cmap="YlOrRd", alpha=0.75, zorder=0)

    dx = config.PITCH_LENGTH / nx_x
    dy = config.PITCH_WIDTH / nx_y
    for zone in min_cut:
        ix = int(zone[1:].split("y")[0])
        iy = int(zone.split("y")[1])
        ax.add_patch(Rectangle((ix * dx, iy * dy), dx, dy, fill=False,
                               edgecolor=RED, linewidth=3, zorder=5))

    ax.set_title(title or "Zone pass-flow intensity", fontsize=13)
    if standalone:
        fig.subplots_adjust(bottom=0.10)
        _caption(
            fig,
            "Brighter zones see more passing; the red-outlined zones are the "
            "bottleneck (min-cut) the ball must pass through to reach attack.",
        )
        _save(fig, save_path)
    return fig


# --- 5.1 figure 4: dynamic network metrics ---------------------------------


def plot_dynamic_network_metrics(dynamic_networks, shift_timestamps, team: str,
                                 save_path: str | None = None):
    """Network density and top-betweenness over match time, with shift markers."""
    times, density, top_betw = [], [], []
    for ts, G in dynamic_networks:
        times.append(ts / 60.0)  # minutes
        density.append(nx.density(G))
        top_betw.append(max(nx.betweenness_centrality(G).values())
                        if G.number_of_edges() else 0.0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(times, density, color=BLUE, lw=2, label="network density")
    ax1.set_xlabel("match time (minutes)")
    ax1.set_ylabel("network density", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)

    ax2 = ax1.twinx()
    ax2.plot(times, top_betw, color="#2a9d8f", lw=1.6, ls="-",
             label="top betweenness")
    ax2.set_ylabel("top betweenness centrality", color="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#2a9d8f")

    for i, s in enumerate(shift_timestamps):
        ax1.axvline(s / 60.0, color=RED, ls="--", lw=1.5,
                    label="detected tactical shift" if i == 0 else None)

    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc="upper right", fontsize=9)
    ax1.set_title(f"{team}: passing-network structure over the match")
    fig.subplots_adjust(bottom=0.14)
    _caption(
        fig,
        "How connected the team's passing is, minute by minute. Red dashed lines "
        "mark automatically detected changes in tactical structure.",
    )
    _save(fig, save_path)
    return fig


# --- 5.1 figure 5: pressure comparison radar -------------------------------


def plot_pressure_comparison(degradation_dict: dict, team: str,
                             save_path: str | None = None):
    """Radar chart comparing the free vs under-pressure passing sub-networks."""
    free = degradation_dict["free_network"]
    pressed = degradation_dict["pressed_network"]
    axes_keys = ["density", "avg_clustering", "n_nodes", "n_edges"]
    labels = ["density", "clustering", "players", "connections"]

    # Normalise each axis to [0, 1] by its max across the two networks.
    free_vals, pressed_vals = [], []
    for k in axes_keys:
        hi = max(free[k], pressed[k], 1e-9)
        free_vals.append(free[k] / hi)
        pressed_vals.append(pressed[k] / hi)

    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    angles += angles[:1]
    free_vals += free_vals[:1]
    pressed_vals += pressed_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, free_vals, color=BLUE, lw=2, label="free (no pressure)")
    ax.fill(angles, free_vals, color=BLUE, alpha=0.25)
    ax.plot(angles, pressed_vals, color=RED, lw=2, label="under pressure")
    ax.fill(angles, pressed_vals, color=RED, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(f"{team}: passing network — free vs under pressure", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    drop = degradation_dict["completion_rate_drop"]
    fig.subplots_adjust(bottom=0.12)
    _caption(
        fig,
        "Each axis is a network property (scaled). The red shape (under opponent "
        f"pressure) shrinking inside the blue shows degradation; completion fell "
        f"{drop*100:.1f} points under pressure.",
    )
    _save(fig, save_path)
    return fig


# --- 5.1 figure 6: cluster scatter -----------------------------------------


def plot_cluster_scatter(feature_matrix, cluster_labels, save_path: str | None = None,
                         annotate_teams=None):
    """2D PCA scatter of teams coloured by tactical cluster.

    ``feature_matrix`` is team-level (index or 'team' column = team name) holding
    FEATURE_COLUMNS. ``cluster_labels`` is a per-team Series/array of cluster ids.
    The 5 reference teams are annotated; the legend names each cluster's tactical
    interpretation.
    """
    fm = feature_matrix.copy()
    if "team" in fm.columns:
        fm = fm.set_index("team")
    labels = (cluster_labels.reindex(fm.index)
              if hasattr(cluster_labels, "reindex") else np.asarray(cluster_labels))

    coords = PCA(n_components=2, random_state=42).fit_transform(fm[FEATURE_COLUMNS])
    annotate_teams = annotate_teams or _SCATTER_ANNOTATE

    labels_arr = np.asarray(labels)
    # Names/colours derived from each cluster's own density profile (not hardcoded).
    names, colors = name_clusters_by_density(fm, labels_arr)

    fig, ax = plt.subplots(figsize=(12, 8))
    for c in sorted(set(int(x) for x in labels_arr)):
        mask = labels_arr == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=120,
                   color=colors.get(c, GREY),
                   label=names.get(c, f"C{c}"),
                   edgecolors="black", linewidth=0.5, alpha=0.85)

    teams = list(fm.index)
    for team in annotate_teams:
        if team in teams:
            i = teams.index(team)
            ax.annotate(team, (coords[i, 0], coords[i, 1]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=10, fontweight="bold")

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("La Liga 2015/16 teams clustered by passing-network style")
    ax.legend(loc="best", fontsize=9, title="tactical cluster")
    fig.subplots_adjust(bottom=0.12)
    _caption(
        fig,
        "Each point is a team; nearby teams play structurally similar football. "
        "Colours are the four tactical clusters found automatically from network "
        "metrics.",
    )
    _save(fig, save_path)
    return fig
