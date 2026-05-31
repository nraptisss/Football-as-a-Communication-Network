"""Generate and execute notebooks/04_ml_models.ipynb."""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_PATH = os.path.join(ROOT, "notebooks", "04_ml_models.ipynb")


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(s):
    return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Phase 4 — ML Models\n\n"
            "Three interpretable models on the full La Liga 2015/16 season "
            "(~380 matches x 2 teams): tactical clustering (unsupervised), "
            "first-half-to-full-match outcome prediction (supervised), and "
            "tactical-shift detection (change-point on dynamic networks)."
        ),
        code(
            "import os, sys\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.insert(0, os.path.abspath('../src'))\n"
            "from tactical_topology import config\n"
            "from tactical_topology.ml import (FEATURE_COLUMNS, cluster_teams_by_network,\n"
            "    train_outcome_model, detect_tactical_shifts)\n"
            "from tactical_topology.network import build_dynamic_network\n"
            "from tactical_topology.loader import load_pass_events, load_match_events\n"
            "PROC = os.path.abspath('../data/processed/')\n"
            "full = pd.read_parquet(os.path.join(PROC, 'feature_matrix.parquet'))\n"
            "fh = pd.read_parquet(os.path.join(PROC, 'feature_matrix_firsthalf.parquet'))\n"
            "print('full-match matrix:', full.shape, '| first-half matrix:', fh.shape)"
        ),
        md(
            "## Model 1 — Tactical clustering (KMeans, k=4, StandardScaler)\n\n"
            "Validation: Barcelona (possession) and Atlético (direct-defensive) "
            "should fall in different clusters. Recall `attacking_flow_efficiency` "
            "measures *directness*, so direct sides score high — that is expected."
        ),
        code(
            "# Team-level clustering: one tactical label per team (season average).\n"
            "clusters = cluster_teams_by_network(full, n_clusters=4)  # level='team'\n"
            "team_feat = full.groupby('team')[FEATURE_COLUMNS].mean()\n"
            "team_feat = team_feat.merge(clusters.set_index('team'), left_index=True,\n"
            "                            right_index=True)\n"
            "profile = team_feat.groupby('cluster_label')[\n"
            "    ['density','attacking_flow_efficiency','team_tempo','low_latency_ratio',\n"
            "     'completion_rate_drop']].mean()\n"
            "profile['n_teams'] = team_feat.groupby('cluster_label').size()\n"
            "profile.round(3)"
        ),
        code(
            "members = (team_feat.reset_index()\n"
            "           .groupby('cluster_label')['team']\n"
            "           .apply(lambda s: ', '.join(sorted(s))))\n"
            "for c, names in members.items():\n"
            "    print(f'cluster {c}: {names}')\n"
            "lab = clusters.set_index('team')['cluster_label']\n"
            "print('\\nBarcelona  -> cluster', lab.get('Barcelona'))\n"
            "print('Atlético   -> cluster', lab.get('Atlético Madrid'))\n"
            "print('Barcelona vs Atlético different cluster:',\n"
            "      lab.get('Barcelona') != lab.get('Atlético Madrid'))"
        ),
        md(
            "## Model 2 — Match outcome (RandomForest, first-half features)\n\n"
            "Predict full-match W/D/L from **first-half** network features only "
            "(harder, more interesting). Stratified k-fold CV; baseline is 33%."
        ),
        code(
            "res = train_outcome_model(fh, n_splits=5)\n"
            "print('n_splits used        :', res['n_splits'])\n"
            "print('CV accuracy (mean)   : %.3f' % res['cv_accuracy_mean'])\n"
            "print('CV scores            :', np.round(res['cv_scores'], 3))\n"
            "print('CV weighted F1       : %.3f' % res['cv_f1_weighted'])\n"
            "print('labels               :', res['confusion_labels'])\n"
            "print('confusion matrix     :\\n', res['confusion_matrix'])"
        ),
        code(
            "imp = res['feature_importances']\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "top = imp.iloc[::-1]\n"
            "ax.barh(top['feature'], top['importance'], color='seagreen')\n"
            "ax.set_title('Phase 4: outcome-model feature importances (first half)')\n"
            "ax.set_xlabel('importance'); plt.tight_layout(); plt.show()\n"
            "print('Top 3 features:', imp['feature'].head(3).tolist())"
        ),
        md(
            "## Model 3 — Tactical shift detection\n\n"
            "Change-point detection on the dynamic passing network, cross-"
            "referenced against substitutions and goals (within ±180s). Worked "
            "example: Barcelona vs Sevilla (265839), where all detected shifts "
            "align with a goal or substitution. Alignment quality varies by match "
            "(an unsupervised detector also flags genuine phase changes that are "
            "not subs/goals), which is why the human picks the match to verify."
        ),
        code(
            "SHIFT_MID = 265839  # Barcelona 2-1 Sevilla\n"
            "SHIFT_TEAM = 'Barcelona'\n"
            "passes = load_pass_events(SHIFT_MID)\n"
            "dyn = build_dynamic_network(passes[passes['team']==SHIFT_TEAM], SHIFT_TEAM)\n"
            "shifts = detect_tactical_shifts(dyn, threshold=2.0)  # plan default\n"
            "\n"
            "ev = load_match_events(SHIFT_MID)\n"
            "subs = ev[(ev['type_name']=='Substitution') & (ev['team']==SHIFT_TEAM)]['seconds']\n"
            "goals = ev[(ev['type_name']=='Shot') & (ev['shot_outcome']=='Goal')]['seconds']\n"
            "events = sorted(list(subs) + list(goals))\n"
            "print('detected shifts (s):', [round(s) for s in shifts])\n"
            "print('sub/goal events (s):', [round(e) for e in events])\n"
            "TOL = 180\n"
            "for s in shifts:\n"
            "    near = [round(e) for e in events if abs(e - s) <= TOL]\n"
            "    print(f'  shift @ {round(s)}s -> events within +-180s: {near}')\n"
            "matched = sum(any(abs(e - s) <= TOL for e in events) for s in shifts)\n"
            "print(f'{matched}/{len(shifts)} detected shifts are within +-180s of a sub/goal')"
        ),
        code(
            "den = [nx_density for nx_density in [__import__('networkx').density(g) for _,g in dyn]]\n"
            "ts = [t for t,_ in dyn]\n"
            "fig, ax = plt.subplots(figsize=(11,4))\n"
            "ax.plot(ts, den, label='network density')\n"
            "for e in events: ax.axvline(e, color='grey', ls=':', alpha=0.7)\n"
            "for s in shifts: ax.axvline(s, color='red', ls='--')\n"
            "ax.set_xlabel('match seconds'); ax.set_ylabel('density')\n"
            "ax.set_title('Phase 4: Barcelona density over time — red=detected shift, grey=sub/goal')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "## Summary for human approval\n\n"
            "- **Clustering**: Barcelona and Atlético land in different clusters "
            "(domain check). Inspect the per-cluster feature profile above.\n"
            "- **Outcome model**: CV accuracy reported vs the 33% baseline; "
            "inspect the top-3 features for football sense.\n"
            "- **Tactical shifts**: detected change-points cross-referenced with "
            "subs/goals within ±180s for match 266424.\n\n"
            "**Human approval questions:** (1) Do Barcelona/Atlético clusters and "
            "the cluster profiles make sense? (2) Do the top-3 outcome features "
            "make football sense? (3) Pick a match and confirm a detected shift "
            "sits within ±3 min of a known event."
        ),
    ]
    return nb


def main():
    nb = build()
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"[Phase 4] Executed and wrote {NB_PATH}")


if __name__ == "__main__":
    main()
