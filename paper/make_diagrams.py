"""Generate the two original diagrams used in the paper:

  fig_system_architecture.png - the data/processing pipeline (the 5 modules)
  fig_concept_mapping.png     - the football <-> communication-network analogy

Both use the project palette and are saved at 300 DPI to paper/figures/.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#1a1a2e"
GOLD = "#f0c040"
RED = "#e63946"
BLUE = "#457b9d"
TEAL = "#2a9d8f"
LIGHT = "#eef2f7"

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def _box(ax, x, y, w, h, text, fc, ec="black", tc="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="#444444", style="-|>", lw=1.8):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        connectionstyle="arc3,rad=0"))


# --------------------------------------------------------------------------
# 1. System architecture / pipeline
# --------------------------------------------------------------------------

def _chip(ax, x, y, w, h, text, ec, fs=8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05",
        linewidth=1.1, edgecolor=ec, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#222222", zorder=3)


def system_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")

    W = 2.0
    xs = [0.3, 2.65, 5.0, 7.35, 9.7]
    top_y, top_h = 5.25, 0.95
    modules = [
        ("StatsBomb\nOpen Data", BLUE, []),
        ("loader.py\nload & clean", BLUE, [("Parquet store", BLUE)]),
        ("network.py\npassing graphs", TEAL,
         [("static network", TEAL), ("dynamic windows", TEAL)]),
        ("telecom.py\nmetrics", RED,
         [("resilience", RED), ("max-flow / min-cut", RED),
          ("tempo (latency)", RED), ("pressing", RED)]),
        ("ml.py\nmodels", "#6a4c93",
         [("clustering", "#6a4c93"), ("outcome model", "#6a4c93"),
          ("shift detection", "#6a4c93")]),
    ]

    # Top pipeline row + connecting arrows.
    for i, (label, color, _chips) in enumerate(modules):
        _box(ax, xs[i], top_y, W, top_h, label, color, fs=9)
        if i > 0:
            _arrow(ax, xs[i - 1] + W, top_y + top_h / 2, xs[i],
                   top_y + top_h / 2)

    # Sub-component chips beneath each module.
    ch_h, gap = 0.46, 0.12
    for i, (_label, color, chips) in enumerate(modules):
        cy = top_y - 0.45
        if chips:
            _arrow(ax, xs[i] + W / 2, top_y, xs[i] + W / 2, cy, color=color,
                   lw=1.2)
        for text, ec in chips:
            cy -= ch_h
            _chip(ax, xs[i], cy, W, ch_h, text, ec)
            cy -= gap

    ax.text(6, 6.35, "Tactical Topology — analysis pipeline",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(6, 0.35,
            "StatsBomb event data is cleaned, turned into passing networks, "
            "scored with communication-network metrics,\nfed to ML models, and "
            "rendered as figures. Intermediate results are cached as Parquet/CSV.",
            ha="center", fontsize=8.5, color="#555555", style="italic")
    fig.savefig(os.path.join(OUT, "fig_system_architecture.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_system_architecture.svg"),
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Conceptual mapping: football <-> communication network
# --------------------------------------------------------------------------

def concept_mapping():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    # Left panel: a tiny passing network (nodes = players)
    ax.text(3, 5.6, "Football team", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)
    nodes = {
        "GK": (0.7, 3.0), "CB": (1.8, 2.0), "FB": (1.8, 4.2),
        "DM": (3.0, 3.0), "CM": (4.0, 2.1), "AM": (4.1, 4.0),
        "FW": (5.2, 3.0),
    }
    edges = [("GK", "CB"), ("CB", "DM"), ("FB", "DM"), ("DM", "CM"),
             ("DM", "AM"), ("CM", "FW"), ("AM", "FW"), ("CB", "FB")]
    for a, b in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.plot([x0, x1], [y0, y1], color=BLUE, lw=2, alpha=0.6, zorder=1)
    for name, (x, y) in nodes.items():
        color = GOLD if name == "DM" else BLUE
        ax.scatter(x, y, s=620, color=color, edgecolors="white",
                   linewidth=1.5, zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=8,
                color="black" if name == "DM" else "white",
                fontweight="bold", zorder=4)

    # Mapping arrow
    _arrow(ax, 5.6, 3.0, 6.6, 3.0, color=NAVY, lw=2.2)
    ax.text(6.1, 3.35, "maps to", ha="center", fontsize=9, style="italic")

    # Right panel: communication-network concepts
    ax.text(9.2, 5.6, "Communication network", ha="center", fontsize=12,
            fontweight="bold", color=NAVY)
    rows = [
        ("Player", "Node / router", BLUE),
        ("Pass", "Message on a channel", TEAL),
        ("Key player out", "Fault tolerance / resilience", RED),
        ("Ball to attack", "Max-flow / min-cut", RED),
        ("Time on the ball", "Processing latency (tempo)", RED),
        ("Opponent pressing", "Channel interference / noise", RED),
    ]
    y = 5.0
    for left, right, color in rows:
        _box(ax, 6.9, y, 2.0, 0.55, left, "white", ec="#999999",
             tc="black", fs=8.5, bold=False)
        _arrow(ax, 8.9, y + 0.27, 9.15, y + 0.27, color="#888888", lw=1.2)
        _box(ax, 9.15, y, 2.5, 0.55, right, color, fs=8.5)
        y -= 0.72

    fig.savefig(os.path.join(OUT, "fig_concept_mapping.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_concept_mapping.svg"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    system_architecture()
    concept_mapping()
    print("[paper] wrote diagrams to", OUT)
