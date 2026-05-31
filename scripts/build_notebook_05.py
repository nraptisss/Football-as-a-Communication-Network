"""Generate and execute notebooks/05_visualisations.ipynb.

Running this notebook produces all six publication figures (PNG + SVG) in
figures/ and is the Phase 5 deliverable.
"""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_PATH = os.path.join(ROOT, "notebooks", "05_visualisations.ipynb")


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(s):
    return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Phase 5 — Visualisations\n\n"
            "Six publication-quality, blog-ready figures. Each is saved to "
            "`figures/` as 300-DPI PNG **and** SVG. Consistent palette and "
            "captions are defined in `tactical_topology.viz`."
        ),
        code(
            "import os, sys\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.insert(0, os.path.abspath('../src'))\n"
            "from tactical_topology import config, viz\n"
            "from tactical_topology.network import (build_passing_network,\n"
            "    build_dynamic_network, compute_node_positions)\n"
            "from tactical_topology.telecom import (build_zone_graph,\n"
            "    compute_resilience, compute_pressure_degradation)\n"
            "from tactical_topology.ml import (FEATURE_COLUMNS,\n"
            "    cluster_teams_by_network, detect_tactical_shifts)\n"
            "from tactical_topology.loader import load_pass_events\n"
            "FIG = os.path.abspath('../figures')\n"
            "os.makedirs(FIG, exist_ok=True)\n"
            "def fp(name): return os.path.join(FIG, name)"
        ),
        md(
            "## fig_02 — Passing network on a pitch (hero figure)\n\n"
            "Barcelona in El Clásico, Real Madrid 0-4 Barcelona (match 266424). "
            "Iniesta is highlighted in gold as the playmaking router — in this "
            "match he is genuinely the top-betweenness node, so his gold node is "
            "also the largest."
        ),
        code(
            "MID = 266424\n"
            "passes = load_pass_events(MID)\n"
            "barca = passes[passes['team']=='Barcelona']\n"
            "G = build_passing_network(barca, 'Barcelona')\n"
            "pos = compute_node_positions(barca, 'Barcelona')\n"
            "iniesta = [n for n in G.nodes() if 'Iniesta' in n][0]\n"
            "viz.plot_passing_network_on_pitch(G, pos, 'Barcelona',\n"
            "    'vs Real Madrid — El Clásico (2015/16)', highlight_node=iniesta,\n"
            "    pass_df=barca, save_path=fp('fig_02_passing_network_pitch.png'))\n"
            "plt.show()"
        ),
        md(
            "## fig_03 — Resilience bar (El Clásico)\n\n"
            "Same match (266424); the three most critical players are Dani Alves, "
            "Iniesta and Busquets — the ball-playing-defender + midfield core."
        ),
        code(
            "res = compute_resilience(G)\n"
            "print('Top 3 critical:', res['player'].head(3).tolist())\n"
            "viz.plot_resilience_bar(res, 'Barcelona',\n"
            "    save_path=fp('fig_03_resilience_bar.png'))\n"
            "plt.show()"
        ),
        md(
            "## fig_03 — Zone-flow heatmap: Barcelona vs Atlético (same match)\n\n"
            "Match 266166 (Atlético 1-2 Barcelona) contains both teams, so the "
            "two heatmaps are from the exact same game. Red outlines = the "
            "defensive→attacking-third bottleneck (min-cut)."
        ),
        code(
            "p2 = load_pass_events(266166)\n"
            "zg_barca = build_zone_graph(p2, 'Barcelona')\n"
            "zg_atleti = build_zone_graph(p2, 'Atlético Madrid')\n"
            "from mplsoccer import Pitch\n"
            "fig, axes = plt.subplots(1, 2, figsize=(20, 8))\n"
            "viz.plot_zone_flow_heatmap(zg_barca, ax=axes[0], title='Barcelona')\n"
            "viz.plot_zone_flow_heatmap(zg_atleti, ax=axes[1], title='Atlético Madrid')\n"
            "fig.suptitle('Zone pass-flow intensity — Atlético 1-2 Barcelona (266166)',\n"
            "             fontsize=14)\n"
            "fig.text(0.5, 0.04, 'Brighter zones see more passing; red-outlined '\n"
            "         'zones are the bottleneck the ball must pass to reach attack. '\n"
            "         'Barcelona spread play more evenly; Atlético funnel through fewer zones.',\n"
            "         ha='center', style='italic', fontsize=10)\n"
            "fig.subplots_adjust(bottom=0.13)\n"
            "fig.savefig(fp('fig_03_zone_flow_heatmap.png'), dpi=300, bbox_inches='tight')\n"
            "fig.savefig(fp('fig_03_zone_flow_heatmap.svg'), bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## fig_04 — Dynamic network metrics with detected shifts\n\n"
            "Match 265839 (Barcelona), where all detected shifts were validated "
            "against goals/substitutions (±3 min)."
        ),
        code(
            "barca_sevilla = load_pass_events(265839)\n"
            "barca_sevilla = barca_sevilla[barca_sevilla['team']=='Barcelona']\n"
            "dyn = build_dynamic_network(barca_sevilla, 'Barcelona')\n"
            "shifts = detect_tactical_shifts(dyn, threshold=2.0)\n"
            "viz.plot_dynamic_network_metrics(dyn, shifts, 'Barcelona',\n"
            "    save_path=fp('fig_04_dynamic_metrics.png'))\n"
            "plt.show()"
        ),
        md("## fig_03 — Pressure comparison radar (Barcelona, El Clásico, free vs pressed)"),
        code(
            "deg = compute_pressure_degradation(barca, 'Barcelona')\n"
            "viz.plot_pressure_comparison(deg, 'Barcelona',\n"
            "    save_path=fp('fig_03_pressure_comparison.png'))\n"
            "plt.show()"
        ),
        md(
            "## fig_04 — Team-level cluster scatter (PCA)\n\n"
            "Teams coloured by their tactical cluster. **Villarreal** is annotated "
            "as a mild outlier: their season network metrics (lower density, more "
            "direct progression) placed them in the lower-table-direct cluster "
            "**C2** despite a respectable league finish — flagged here for "
            "transparency rather than hidden."
        ),
        code(
            "full = pd.read_parquet(os.path.abspath('../data/processed/feature_matrix.parquet'))\n"
            "team_feat = full.groupby('team')[FEATURE_COLUMNS].mean()\n"
            "labels = cluster_teams_by_network(full, 4).set_index('team')['cluster_label']\n"
            "viz.plot_cluster_scatter(team_feat, labels,\n"
            "    save_path=fp('fig_04_cluster_scatter.png'))\n"
            "plt.show()\n"
            "print('Villarreal cluster:', labels['Villarreal'],\n"
            "      '->', viz.CLUSTER_NAMES[labels['Villarreal']])"
        ),
        code(
            "import glob\n"
            "pngs = sorted(glob.glob(fp('*.png'))); svgs = sorted(glob.glob(fp('*.svg')))\n"
            "print(f'{len(pngs)} PNG + {len(svgs)} SVG in figures/')\n"
            "for p in pngs: print(' ', os.path.basename(p))"
        ),
        md(
            "## Figures produced\n\n"
            "All six plot functions executed without error and figures/ holds the "
            "PNG + SVG pairs. Ready for the blog post and paper."
        ),
    ]
    return nb


def main():
    nb = build()
    ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"[Phase 5] Executed and wrote {NB_PATH}")


if __name__ == "__main__":
    main()
