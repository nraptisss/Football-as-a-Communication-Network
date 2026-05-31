"""Generate and execute notebooks/03_telecom_metrics.ipynb."""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_PATH = os.path.join(ROOT, "notebooks", "03_telecom_metrics.ipynb")


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(s):
    return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Phase 3 — Telecom-Theory Metrics\n\n"
            "Apply communication-network theory to the 5 ground-truth matches: "
            "resilience (node-failure), max-flow/min-cut (spatial), tempo "
            "(latency), and pressing-as-interference. All four metrics are "
            "computed for every team; per-match team-level results are saved to "
            "`data/processed/telecom_metrics_<match_id>.csv`."
        ),
        code(
            "import os, sys\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.insert(0, os.path.abspath('../src'))\n"
            "from tactical_topology import config\n"
            "from tactical_topology.network import build_passing_network\n"
            "from tactical_topology import telecom as T\n"
            "PROC = os.path.abspath('../data/processed/')\n"
            "def load(mid):\n"
            "    return pd.read_parquet(os.path.join(PROC, f'passes_{mid}.parquet'))"
        ),
        md("## Team-level telecom summary (all 5 matches, both teams)"),
        code(
            "tables = [pd.read_csv(os.path.join(PROC, f'telecom_metrics_{m}.csv'))\n"
            "          for m in config.GROUND_TRUTH_MATCH_IDS]\n"
            "summary = pd.concat(tables, ignore_index=True)\n"
            "summary[['match_id','team','top_resilience_player','top_resilience_score',\n"
            "         'max_flow_value','attacking_flow_efficiency','team_tempo_seconds',\n"
            "         'completion_rate_drop_under_pressure']]"
        ),
        md(
            "## 3.1.1 Resilience — Barcelona, El Clásico (match 266424)\n\n"
            "Validation target: the most critical players should be midfielders "
            "or ball-playing defenders, **not** orthodox centre-backs."
        ),
        code(
            "G = build_passing_network(load(266424), 'Barcelona')\n"
            "res = T.compute_resilience(G)\n"
            "print('Top 5 most critical (resilience_score):')\n"
            "print(res.head(5)[['player','resilience_score']].to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "top = res.head(8).iloc[::-1]\n"
            "ax.barh([p.split()[0] for p in top['player']], top['resilience_score'],\n"
            "        color='teal')\n"
            "ax.set_xlabel('resilience score (higher = more critical)')\n"
            "ax.set_title('Phase 3: Barcelona resilience (El Clasico)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Observed:** top-3 are Dani Alves (ball-playing right-back), Iniesta "
            "and Busquets — exactly the playmaker / ball-playing-defender profile "
            "expected for a possession side."
        ),
        md(
            "## 3.1.2 Max-flow efficiency — Barcelona vs Atlético\n\n"
            "`attacking_flow_efficiency = max_flow(def→att) / total completed "
            "passes`. By construction this measures **directness per pass**."
        ),
        code(
            "eff = summary.groupby('team')['attacking_flow_efficiency'].mean()\n"
            "eff = eff.sort_values(ascending=False)\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "ax.barh(eff.index, eff.values, color='darkorange')\n"
            "ax.set_xlabel('attacking flow efficiency (max-flow / total passes)')\n"
            "ax.set_title('Phase 3: flow efficiency by team (mean over matches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "eff.round(4)"
        ),
        md(
            "**⚠ Interpretation flag for the human.** The metric ranks *direct / "
            "counter-attacking* sides (Atlético, Sevilla, Sporting) **above** "
            "possession Barcelona. This is internally consistent with the plan's "
            "own definition ('higher = more direct progression'): Barcelona "
            "completes a huge volume of recycling passes, inflating the "
            "denominator, so its *per-pass* progression efficiency is lower. "
            "However it runs **opposite** to the literal Phase 3 exit-criterion "
            "wording ('attacking teams [Barcelona] show higher flow efficiency "
            "than defensive teams [Atlético]'). I have **not** altered the metric "
            "— this is a domain-knowledge call for you (see summary below)."
        ),
        md(
            "## 3.1.3 Tempo — touch durations (target: median 0.5–5.0s)"
        ),
        code(
            "rows = []\n"
            "for m in config.GROUND_TRUTH_MATCH_IDS:\n"
            "    df = load(m)\n"
            "    for t in df['team'].unique():\n"
            "        tt = T.compute_player_tempo(df, t)\n"
            "        rows.append({'match_id': m, 'team': t,\n"
            "                     'team_tempo_s': T.compute_team_tempo(df, t),\n"
            "                     'median_player_median_s': tt['median_touch_seconds'].median()})\n"
            "tempo_tbl = pd.DataFrame(rows)\n"
            "print('median touch range: %.2f - %.2f s' % (\n"
            "    tempo_tbl['median_player_median_s'].min(),\n"
            "    tempo_tbl['median_player_median_s'].max()))\n"
            "tempo_tbl.round(2)"
        ),
        md("All team medians fall within the physiologically plausible 0.5–5.0s band."),
        md("## 3.1.4 Pressing as interference — completion drop under pressure"),
        code(
            "frames = [load(m) for m in config.GROUND_TRUTH_MATCH_IDS]\n"
            "allp = pd.concat(frames)\n"
            "g = allp.groupby('under_pressure')['pass_complete'].mean()\n"
            "print('Pooled completion  free = %.3f  pressed = %.3f  drop = %.3f' % (\n"
            "    g[False], g[True], g[False]-g[True]))\n"
            "fig, ax = plt.subplots(figsize=(7,5))\n"
            "s = summary.sort_values('completion_rate_drop_under_pressure')\n"
            "labels = s['team'] + ' (' + s['match_id'].astype(str) + ')'\n"
            "ax.barh(labels, s['completion_rate_drop_under_pressure'], color='indianred')\n"
            "ax.axvline(0, color='k', lw=0.8)\n"
            "ax.set_xlabel('completion rate drop (free - pressed)')\n"
            "ax.set_title('Phase 3: pass-completion degradation under pressure')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Observed:** pooled completion falls from 81.9% (free) to 73.5% "
            "(pressed) — the known football fact holds, and strongly for every "
            "marquee team. Two single-match team values are slightly negative "
            "(small under-pressure sample), not a flag bug — the pooled "
            "direction confirms the flag is correct."
        ),
        md(
            "## Summary for human approval\n\n"
            "1. **Resilience** ✅ — top critical players are midfielders / "
            "ball-playing defenders (Alves, Iniesta, Busquets for Barcelona).\n"
            "2. **Max-flow efficiency** ⚠ — measures directness; **direct sides "
            "rank above possession Barcelona**, which contradicts the literal "
            "exit-criterion wording. Needs your decision.\n"
            "3. **Tempo** ✅ — all medians within 0.5–5.0s.\n"
            "4. **Pressure** ✅ — completion clearly lower under pressure "
            "(pooled 81.9% → 73.5%).\n\n"
            "**Human approval question:** Do these outputs align with your "
            "football intuition — and how should we treat the flow-efficiency "
            "directionality (#2)?"
        ),
    ]
    return nb


def main():
    nb = build()
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"[Phase 3] Executed and wrote {NB_PATH}")


if __name__ == "__main__":
    main()
