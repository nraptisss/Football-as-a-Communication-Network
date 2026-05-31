"""Generate and execute notebooks/02_network_construction.ipynb.

Builds the Phase 2 notebook programmatically with nbformat and executes it
top-to-bottom so the committed notebook contains real outputs.
"""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_PATH = os.path.join(ROOT, "notebooks", "02_network_construction.ipynb")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Phase 2 — Network Construction\n\n"
            "Build weighted directed passing networks for both teams in each of "
            "the 5 ground-truth matches, inspect `network_summary`, and draw one "
            "team's network with NetworkX (no mplsoccer yet — that is Phase 5)."
        ),
        code(
            "import os, sys\n"
            "import pandas as pd\n"
            "import networkx as nx\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.insert(0, os.path.abspath('../src'))\n"
            "from tactical_topology import config\n"
            "from tactical_topology.network import (\n"
            "    build_passing_network, build_dynamic_network,\n"
            "    compute_node_positions, network_summary,\n"
            ")\n"
            "PROC = os.path.abspath('../data/processed/')\n"
            "def load(mid):\n"
            "    return pd.read_parquet(os.path.join(PROC, f'passes_{mid}.parquet'))"
        ),
        md("## network_summary for every team in all 5 ground-truth matches"),
        code(
            "rows = []\n"
            "for mid in config.GROUND_TRUTH_MATCH_IDS:\n"
            "    df = load(mid)\n"
            "    for team in df['team'].unique():\n"
            "        G = build_passing_network(df, team)\n"
            "        s = network_summary(G)\n"
            "        s['match_id'] = mid\n"
            "        s['team'] = team\n"
            "        s['dyn_steps'] = len(build_dynamic_network(df, team))\n"
            "        rows.append(s)\n"
            "summary = pd.DataFrame(rows)[[\n"
            "    'match_id', 'team', 'n_nodes', 'n_edges', 'density',\n"
            "    'avg_clustering', 'top_betweenness_node', 'top_eigenvector_node',\n"
            "    'dyn_steps']]\n"
            "summary.round(3)"
        ),
        md(
            "**Sanity checks (PLAN_1.md 2.3):** 10–14 nodes per team, density "
            "roughly 0.15–0.60, a clear high-betweenness 'router', and ≥10 "
            "dynamic time steps per 90-min match."
        ),
        md(
            "## NetworkX draw — El Clásico, Barcelona passing network\n\n"
            "Node positions are average pass-origin locations on the pitch "
            "(StatsBomb 0–120 × 0–80). Node size ∝ betweenness centrality; the "
            "top-betweenness 'router' is highlighted in gold."
        ),
        code(
            "MID = 266424  # Real Madrid 0-4 Barcelona\n"
            "TEAM = 'Barcelona'\n"
            "df = load(MID)\n"
            "G = build_passing_network(df, TEAM)\n"
            "pos = compute_node_positions(df, TEAM)\n"
            "pos = {n: pos[n] for n in G.nodes() if n in pos}\n"
            "btw = nx.betweenness_centrality(G)\n"
            "top = max(btw, key=btw.get)\n"
            "sizes = [300 + 4000 * btw[n] for n in G.nodes()]\n"
            "colors = ['gold' if n == top else 'lightsteelblue' for n in G.nodes()]\n"
            "weights = [d['weight'] for _, _, d in G.edges(data=True)]\n"
            "fig, ax = plt.subplots(figsize=(11, 7.5))\n"
            "nx.draw_networkx_edges(G, pos, ax=ax, width=[0.3 * w for w in weights],\n"
            "                       edge_color='grey', alpha=0.5,\n"
            "                       arrowsize=8, connectionstyle='arc3,rad=0.07')\n"
            "nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,\n"
            "                       edgecolors='black')\n"
            "labels = {n: n.split()[0] for n in G.nodes()}\n"
            "nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)\n"
            "ax.set_xlim(0, 120); ax.set_ylim(0, 80)\n"
            "ax.set_xlabel('pitch length (attacking ->)'); ax.set_ylabel('pitch width')\n"
            "ax.set_title(f'Phase 2: {TEAM} passing network (match {MID})\\n'\n"
            "             f'gold = top betweenness: {top}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('top betweenness:', top)\n"
            "print('top eigenvector:', network_summary(G)['top_eigenvector_node'])"
        ),
        md(
            "## Notes for human verification\n\n"
            "- **Node positions** reproduce Barcelona's possession shape: a back "
            "line low, midfield triangle, front three high — confirm it looks "
            "like a football shape, not a random blob.\n"
            "- **High-betweenness 'router':** across the marquee teams the top "
            "betweenness node is a deep distributor (e.g. Iniesta for Barcelona, "
            "Kroos for Real Madrid, Koke/Gabi for Atlético). Note that "
            "Busquets-type metronomes are so *uniformly* connected that the "
            "bridging role often falls to ball-playing centre-backs / full-backs "
            "(e.g. Piqué, Jordi Alba) — this is expected in dense possession "
            "networks. Weighted **eigenvector** centrality surfaces the "
            "metronomes more directly (Kroos, Modrić, Koke).\n\n"
            "**Human approval question:** Does the network structure (shape + key "
            "nodes) match your football intuition for these teams?"
        ),
    ]
    return nb


def main():
    nb = build()
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"[Phase 2] Executed and wrote {NB_PATH}")


if __name__ == "__main__":
    main()
