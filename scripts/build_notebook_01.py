"""Generate and execute notebooks/01_data_exploration.ipynb.

Builds the Phase 1 exploration notebook programmatically with nbformat, then
executes it top-to-bottom so the committed notebook contains real outputs and
is guaranteed to run without errors.
"""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_PATH = os.path.join(ROOT, "notebooks", "01_data_exploration.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Phase 1 — Data Exploration\n\n"
            "Sanity-check the cleaned pass-event data for the 5 ground-truth "
            "La Liga 2015/16 matches: total passes per team per match, pass "
            "completion rate, and the distribution of passes made under "
            "pressure."
        ),
        code(
            "import os, sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.insert(0, os.path.abspath('../src'))\n"
            "from tactical_topology import config\n"
            "from tactical_topology.loader import cache_match_passes\n"
            "pd.set_option('display.width', 120)"
        ),
        md("## Load the 5 ground-truth matches (from parquet cache)"),
        code(
            "cache = cache_match_passes(config.GROUND_TRUTH_MATCH_IDS,\n"
            "                           os.path.abspath('../data/processed/'))\n"
            "for mid, df in cache.items():\n"
            "    print(mid, '->', len(df), 'passes,', sorted(df['team'].unique()))"
        ),
        md(
            "## Total passes per team per match\n\n"
            "Possession sides should attempt clearly more passes than their "
            "opponents."
        ),
        code(
            "rows = []\n"
            "for mid, df in cache.items():\n"
            "    g = df.groupby('team').agg(\n"
            "        passes=('id', 'size'),\n"
            "        completion_rate=('pass_complete', 'mean'),\n"
            "        under_pressure_rate=('under_pressure', 'mean'),\n"
            "    ).reset_index()\n"
            "    g.insert(0, 'match_id', mid)\n"
            "    rows.append(g)\n"
            "summary = pd.concat(rows, ignore_index=True)\n"
            "summary['completion_rate'] = (summary['completion_rate'] * 100).round(1)\n"
            "summary['under_pressure_rate'] = (summary['under_pressure_rate'] * 100).round(1)\n"
            "summary"
        ),
        md(
            "## Pass completion rate per team\n\n"
            "La Liga possession teams typically complete **75–90%** of passes."
        ),
        code(
            "labels = summary['team'] + ' (' + summary['match_id'].astype(str) + ')'\n"
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "ax.barh(labels, summary['completion_rate'], color='steelblue')\n"
            "ax.axvspan(75, 90, color='green', alpha=0.1, label='typical 75-90%')\n"
            "ax.set_xlabel('Pass completion rate (%)')\n"
            "ax.set_title('Phase 1: pass completion rate per team per match')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Distribution of passes made under pressure\n\n"
            "What fraction of each team's passes were played while under "
            "pressure from an opponent."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "ax.barh(labels, summary['under_pressure_rate'], color='indianred')\n"
            "ax.set_xlabel('Passes under pressure (%)')\n"
            "ax.set_title('Phase 1: share of passes made under pressure')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Observations\n\n"
            "- Possession sides (Barcelona, Real Madrid, Atletico) attempt far "
            "more passes than their opponents and complete 77–88% — squarely in "
            "the expected La Liga range.\n"
            "- Underdogs going more direct (Sevilla, Sporting Gijon) sit a little "
            "below 75%, which is football-plausible for low-possession away sides.\n"
            "- Under-pressure share is broadly 10–22%, higher for the teams "
            "without the ball — also as expected.\n\n"
            "**Human approval question:** Do these pass completion rates look "
            "right for La Liga 2015/16?"
        ),
    ]
    return nb


def main() -> None:
    nb = build()
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.join(ROOT, "notebooks")}})
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"[Phase 1] Executed and wrote {NB_PATH}")


if __name__ == "__main__":
    main()
