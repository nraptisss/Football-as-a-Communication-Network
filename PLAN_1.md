# Tactical Topology: Football as a Communication Network
## Implementation Plan for Claude Code

> **Purpose:** This document guides Claude Code through a phased implementation of a
> football network analysis project. Each phase has explicit success criteria that
> **must be verified before proceeding**. Do not carry assumptions or unverified
> outputs from one phase into the next.

---

## 🧭 Project Overview

**Goal:** Analyse football team tactical structure using communication network theory
(graph resilience, max-flow, latency/tempo, pressing-as-interference), layered with
ML models and clean visualisations.

**Data source:** StatsBomb Open Data (free, no API key required)
**Primary language:** Python 3.10+
**Output:** Reproducible Jupyter notebooks + a standalone Python package + blog-ready figures

---

## 📋 What You (the Human) Need to Provide

### Required — before Phase 0
| Item | Why it's needed | Where to get it |
|---|---|---|
| A working Python 3.10+ environment | All code runs here | [python.org](https://python.org) or `conda` |
| Git installed | For cloning StatsBomb data | [git-scm.com](https://git-scm.com) |
| ~5 GB free disk space | StatsBomb open data repo | Your machine |
| A code editor with Jupyter support | To run notebooks | VS Code + Jupyter extension recommended |

### Optional but Recommended
| Item | Why |
|---|---|
| GitHub account | To publish the project publicly |
| A Kaggle or Medium account | To publish write-ups and reach the analytics community |

### You Do NOT Need to Provide
- Any API keys (StatsBomb open data is public)
- Football domain labels or annotations (derived from data)
- Pre-trained models (all trained from scratch in the project)

### Your Role During Implementation
Claude Code will generate all code. Your role in each phase is to:
1. Run the phase's validation script and confirm outputs match expected results
2. Approve or flag any domain-knowledge decisions (e.g. "does this metric make
   tactical sense?") — your football knowledge is critical here
3. Git commit at the end of each phase before moving on

---

## 🗂️ Project Structure (Final)

```
tactical-topology/
│
├── data/                          # Raw + processed data (gitignored for raw)
│   ├── raw/                       # Cloned StatsBomb open-data (symlink or copy)
│   └── processed/                 # Parquet/CSV outputs from Phase 1
│
├── src/
│   └── tactical_topology/
│       ├── __init__.py
│       ├── loader.py              # Phase 1: data loading & cleaning
│       ├── network.py             # Phase 2: graph construction
│       ├── telecom.py             # Phase 3: telecom-theory metrics
│       ├── ml.py                  # Phase 4: ML models
│       └── viz.py                 # Phase 5: visualisations
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_network_construction.ipynb
│   ├── 03_telecom_metrics.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_visualisations.ipynb
│
├── tests/
│   ├── test_loader.py
│   ├── test_network.py
│   ├── test_telecom.py
│   └── test_ml.py
│
├── figures/                       # Exported publication-quality figures
├── requirements.txt
├── setup.py
├── README.md
└── PLAN.md                        # This file
```

---

## ✅ Phase 0 — Environment & Data Setup

**Objective:** Reproducible environment with verified StatsBomb data access.
This phase is purely infrastructure — no analysis yet.

### Tasks

#### 0.1 Create project scaffold
- Create the full directory structure above
- Initialise a git repository
- Create `.gitignore` (ignore `data/raw/`, `__pycache__/`, `.ipynb_checkpoints/`)

#### 0.2 Create `requirements.txt`
Exact versions pinned:
```
statsbombpy>=1.0.3
pandas>=2.0.0
numpy>=1.26.0
networkx>=3.2.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
mplsoccer>=1.2.0
plotly>=5.18.0
scipy>=1.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
tqdm>=4.66.0
pyarrow>=15.0.0
pytest>=8.0.0
```

#### 0.3 Verify StatsBomb data access
Using `statsbombpy` (no credentials required for open data):
```python
from statsbombpy import sb
competitions = sb.competitions()
# Must return a DataFrame with competitions including:
# - FIFA World Cup
# - UEFA Champions League (2003/04, 2004/05, 2005/06)
# - La Liga (multiple seasons)
# - Women's Super League
```

#### 0.4 Select working dataset
For development and validation, use a **single fixed dataset**:
- **Competition:** La Liga, Season 2015/16 (Messi's Barcelona — tactically rich,
  well-documented, good for sanity-checking against known football knowledge)
- This dataset has ~380 matches with full event data

### ✅ Phase 0 Exit Criteria — ALL must pass before Phase 1

```
[ ] pip install -r requirements.txt completes with no errors
[ ] sb.competitions() returns a non-empty DataFrame
[ ] sb.matches(competition_id=11, season_id=27) returns exactly 380 matches
[ ] sb.events(match_id=<any valid id>) returns a DataFrame with columns:
      ['id', 'index', 'period', 'timestamp', 'type', 'team', 'player',
       'location', 'pass', 'carry', 'shot', 'under_pressure']
[ ] Project directory structure matches the spec above (verified with `tree`)
[ ] Initial git commit made with message: "chore: phase 0 — environment setup"
```

**⛔ Do not proceed to Phase 1 until all 6 criteria are checked.**

---

## ✅ Phase 1 — Data Loading & Cleaning

**Objective:** A clean, validated, match-level events dataset saved to disk.
Every downstream phase reads from this — errors here corrupt everything.

### Tasks

#### 1.1 Implement `src/tactical_topology/loader.py`

The module must expose exactly these public functions:

```python
def load_match_events(match_id: int) -> pd.DataFrame:
    """Load and clean all events for a single match."""

def load_pass_events(match_id: int) -> pd.DataFrame:
    """Return only pass events with required columns validated."""

def load_season_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return match metadata for a full season."""

def load_and_cache_season_events(
    competition_id: int,
    season_id: int,
    cache_dir: str = "data/processed/"
) -> dict[int, pd.DataFrame]:
    """Load all match events for a season, cache as parquet files."""
```

#### 1.2 Data cleaning rules (implement in loader.py)
Apply these in order — document each as a comment in code:

1. **Remove null locations:** Drop pass events where `location` or
   `pass_end_location` is NaN — these cannot be used in network analysis
2. **Standardise team names:** Store a `team_id` integer alongside `team` string
   to avoid string-matching bugs downstream
3. **Period filter:** Keep only periods 1 and 2 (remove extra time, penalties)
   by default; expose a parameter to include them
4. **Timestamp to seconds:** Convert `timestamp` (HH:MM:SS.mmm) to `seconds`
   (float) for tempo calculations in Phase 3
5. **Incomplete passes:** Keep incomplete passes (they matter for resilience
   analysis) but add a boolean column `pass_complete: bool`
6. **Pressure flag:** Ensure `under_pressure` column is bool (fill NaN → False)

#### 1.3 Validation dataset
Load **5 specific matches** from La Liga 2015/16 and save their cleaned pass
events as parquet. These 5 matches are the **ground truth** for all subsequent
phases. Record their match IDs in a `config.py` file.

#### 1.4 Write `tests/test_loader.py`
Tests must cover:
- `load_pass_events` returns only rows where `type.name == "Pass"`
- No NaN values in `location`, `pass_end_location`, `player.name`, `team.name`
- `seconds` column is numeric and within `[0, 6000]` (100 min in seconds)
- `pass_complete` is boolean
- `under_pressure` is boolean with no NaN

### ✅ Phase 1 Exit Criteria — ALL must pass before Phase 2

```
[ ] pytest tests/test_loader.py — all tests GREEN
[ ] 5 match parquet files exist in data/processed/ and are non-empty
[ ] loader.py has no TODO comments or placeholder logic
[ ] Notebook 01_data_exploration.ipynb runs top-to-bottom without errors
[ ] Notebook shows: total passes per team per match, pass completion rate,
    distribution of under_pressure passes — confirm these look reasonable
    (La Liga teams typically complete 75–90% of passes)
[ ] Human approval: "Do the pass completion rates look right for La Liga?"
[ ] Git commit: "feat: phase 1 — data loader and cleaning"
```

**⛔ Do not proceed to Phase 2 until all criteria are checked.**

---

## ✅ Phase 2 — Network Construction

**Objective:** Build validated passing networks (static and dynamic) for both
teams in a match, represented as weighted directed graphs.

### Tasks

#### 2.1 Implement `src/tactical_topology/network.py`

Expose these public functions:

```python
def build_passing_network(
    pass_df: pd.DataFrame,
    team: str,
    min_passes: int = 2
) -> nx.DiGraph:
    """
    Build a static passing network for one team in one match.
    Nodes = players. Edges = passes between players.
    Edge weight = number of successful passes between that pair.
    Only include edges with >= min_passes to reduce noise.
    """

def build_dynamic_network(
    pass_df: pd.DataFrame,
    team: str,
    window_seconds: float = 300.0,
    step_seconds: float = 60.0
) -> list[tuple[float, nx.DiGraph]]:
    """
    Build a time series of passing networks using a sliding window.
    Returns list of (timestamp, graph) tuples.
    Default: 5-minute windows, sliding every 1 minute.
    """

def compute_node_positions(
    pass_df: pd.DataFrame,
    team: str
) -> dict[str, tuple[float, float]]:
    """
    Compute average pitch position (x, y) for each player
    based on pass origin locations. Used for pitch overlays.
    Returns dict: {player_name: (avg_x, avg_y)}
    """

def network_summary(G: nx.DiGraph) -> dict:
    """
    Return a dict of standard graph metrics for a network.
    Keys: density, avg_clustering, n_nodes, n_edges,
          top_betweenness_node, top_eigenvector_node
    """
```

#### 2.2 Graph construction rules
Document these explicitly in code comments:

1. **Node = player name** (string). Use cleaned `player.name` from loader.
2. **Edge direction:** Passer → Receiver (directed graph, `nx.DiGraph`)
3. **Edge weight:** Count of completed passes between that pair
4. **Self-loops:** Not allowed — drop any pass where passer == receiver
5. **Substitutes:** Include them as nodes; their network position will naturally
   reflect limited involvement. Do not remove them — they matter for resilience.
6. **Position coordinates:** Use pitch coordinate system as-is from StatsBomb
   (0–120 length, 0–80 width). Do not invert or normalise yet.

#### 2.3 Sanity checks for network outputs
For a La Liga match, a well-formed team passing network should have:
- **10–14 nodes** (starting 11 minus early red cards, plus subs who passed)
- **Density between 0.15 and 0.60** (not too sparse, not fully connected)
- **One clear high-betweenness node** in possession-based teams (the "router")

#### 2.4 Write `tests/test_network.py`
Tests must cover:
- Graph is directed (`isinstance(G, nx.DiGraph)`)
- No self-loops (`nx.number_of_selfloops(G) == 0`)
- All edge weights are positive integers
- Node count is between 5 and 14
- `compute_node_positions` returns coordinates within pitch bounds [0–120, 0–80]
- `network_summary` returns all required keys

### ✅ Phase 2 Exit Criteria — ALL must pass before Phase 3

```
[ ] pytest tests/test_network.py — all tests GREEN
[ ] For all 5 ground-truth matches: both teams produce valid graphs
[ ] network_summary() output inspected and confirmed sensible for La Liga teams
[ ] Dynamic network produces at least 10 time steps for a 90-min match
[ ] Notebook 02_network_construction.ipynb runs top-to-bottom without errors
[ ] Notebook shows a basic NetworkX draw of one team's passing network
    (no mplsoccer yet — that's Phase 5). Visually confirm it makes sense:
    does the high-betweenness node correspond to the team's known playmaker?
[ ] Human approval: "Does the network structure match your football intuition
    for these teams?"
[ ] Git commit: "feat: phase 2 — passing network construction"
```

**⛔ Do not proceed to Phase 3 until all criteria are checked.**

---

## ✅ Phase 3 — Telecom-Theory Metrics

**Objective:** Implement the novel metrics that differentiate this project —
apply communication network theory to football passing networks.

This is the intellectual core of the project. Implement carefully and validate
each metric conceptually before moving to the next.

### Tasks

#### 3.1 Implement `src/tactical_topology/telecom.py`

Implement these metrics in order (each builds on the previous):

---

**3.1.1 — Network Resilience (Node Failure Simulation)**

```python
def compute_resilience(G: nx.DiGraph, metric: str = "betweenness") -> pd.DataFrame:
    """
    Simulate the removal of each player (node) one at a time and measure
    network degradation. Inspired by fault-tolerance analysis in telecom networks.

    For each node removal:
    - Recompute network density
    - Recompute average clustering coefficient
    - Recompute number of weakly connected components
      (fragmentation = more components = worse)

    Returns a DataFrame with one row per player:
    columns = ['player', 'density_drop', 'clustering_drop',
               'fragmentation_increase', 'resilience_score']

    resilience_score = combined normalised impact (higher = more critical player)
    """
```

**Expected behaviour check:** For Barcelona 2015/16, removing Iniesta or Busquets
should produce the highest resilience_score. If a random defender scores highest,
the metric is likely wrong.

---

**3.1.2 — Flow Analysis (Max-Flow / Min-Cut on Pitch Zones)**

```python
def build_zone_graph(
    pass_df: pd.DataFrame,
    team: str,
    n_zones_x: int = 5,
    n_zones_y: int = 3
) -> nx.DiGraph:
    """
    Divide the pitch into a grid of zones (default: 5 vertical × 3 horizontal = 15 zones).
    Build a zone-level directed graph where:
    - Nodes = pitch zones
    - Edge weight = number of completed passes from zone A to zone B

    This abstracts away individual players and focuses on spatial ball flow.
    """

def compute_max_flow(zone_graph: nx.DiGraph) -> dict:
    """
    Compute maximum flow from defensive third (source) to attacking third (sink).
    Uses scipy/networkx max-flow algorithms.

    Returns:
    {
        'max_flow_value': float,       # Total flow from def → att third
        'min_cut_zones': list[str],    # Zones forming the bottleneck
        'flow_dict': dict              # Full flow on each edge
    }
    """

def compute_attacking_flow_efficiency(zone_graph: nx.DiGraph) -> float:
    """
    Ratio of max_flow_value to total passes attempted.
    Measures how efficiently the team moves the ball forward.
    Higher = more direct and efficient progression.
    """
```

---

**3.1.3 — Tempo (Latency Modelling)**

```python
def compute_player_tempo(pass_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For each player, compute their average 'touch duration':
    time between receiving a pass and playing the next pass.

    This is analogous to processing latency in a network node.

    Method:
    - For each player, find all events where they receive a pass
    - Find the next pass they make
    - Delta = next_pass_seconds - receive_seconds
    - Filter out deltas > 30s (likely a different possession sequence)

    Returns DataFrame: ['player', 'avg_touch_seconds', 'median_touch_seconds',
                        'n_touches', 'low_latency_ratio']
    low_latency_ratio = fraction of touches under 2 seconds (one-touch / quick play)
    """

def compute_team_tempo(pass_df: pd.DataFrame, team: str) -> float:
    """
    Team-level average touch duration (weighted by number of touches per player).
    Lower = faster tempo = less time for opponent to reorganise.
    """
```

---

**3.1.4 — Pressing as Interference (Signal Degradation)**

```python
def compute_pressure_degradation(pass_df: pd.DataFrame, team: str) -> dict:
    """
    Measure how pressing (under_pressure flag) affects passing network quality.

    Compute two sub-networks:
    1. passes_under_pressure: pass_df where under_pressure == True
    2. passes_not_under_pressure: pass_df where under_pressure == False

    For each sub-network, compute network_summary().

    Return comparison dict:
    {
        'free_network': dict,       # network_summary for unpressed passes
        'pressed_network': dict,    # network_summary for pressed passes
        'density_degradation': float,        # free_density - pressed_density
        'completion_rate_drop': float,       # completion rate difference
        'centrality_shift': dict,   # how betweenness centrality changes
                                    # under pressure (do different players emerge?)
    }
    """
```

#### 3.2 Write `tests/test_telecom.py`
Tests must cover:
- `compute_resilience` returns a row for every node in the graph
- `resilience_score` is between 0 and 1 (normalised)
- `compute_max_flow` returns a positive flow value for any valid zone graph
- `compute_player_tempo` returns no negative touch durations
- `compute_pressure_degradation` returns both sub-networks

### ✅ Phase 3 Exit Criteria — ALL must pass before Phase 4

```
[ ] pytest tests/test_telecom.py — all tests GREEN
[ ] Resilience: for a known possession team (Barcelona), the top 3 most
    critical players are midfielders/playmakers, not centre-backs
    (Human to verify against football knowledge)
[ ] Max-flow: attacking teams show higher flow efficiency than defensive teams
    (Compare Barcelona vs Atletico Madrid from the dataset)
[ ] Tempo: computed touch durations are physiologically plausible
    (median between 0.5s and 5.0s — if outside this range, logic error)
[ ] Pressure degradation: completion rate is lower under pressure
    (this is a known football fact — if not, bug in pressure flag filtering)
[ ] Notebook 03_telecom_metrics.ipynb runs top-to-bottom without errors
[ ] All 4 metrics computed for all 5 ground-truth matches
[ ] Results saved as CSV to data/processed/telecom_metrics_<match_id>.csv
[ ] Human approval: "Do the metric outputs align with your football intuition?"
[ ] Git commit: "feat: phase 3 — telecom-theory metrics"
```

**⛔ Do not proceed to Phase 4 until all criteria are checked.**

---

## ✅ Phase 4 — ML Models

**Objective:** Three focused ML models that use network + telecom features.
Prioritise interpretability over raw accuracy — this is a research project,
not a production prediction system.

### Tasks

#### 4.1 Feature engineering pipeline
Before model training, build a feature matrix. For each match × team:

```
Network features:
  - density
  - avg_clustering
  - top_betweenness_centrality (value)
  - top_eigenvector_centrality (value)
  - n_nodes (players involved)
  - is_centralized (bool: betweenness gini > 0.5)

Telecom features:
  - max_flow_value
  - attacking_flow_efficiency
  - team_tempo (avg touch duration)
  - low_latency_ratio
  - density_degradation (under pressure)
  - completion_rate_drop (under pressure)

Target variables (per match per team):
  - goals_scored (from match metadata)
  - match_result (W/D/L from team perspective)
  - xg_proxy (shots × 0.1 as simple proxy if xG not available)
```

Save feature matrix as `data/processed/feature_matrix.parquet`.

#### 4.2 Model 1 — Tactical Clustering (Unsupervised)

```python
def cluster_teams_by_network(
    feature_matrix: pd.DataFrame,
    n_clusters: int = 4
) -> pd.DataFrame:
    """
    KMeans clustering of teams based on network + telecom features.
    Use only network topology features (not match outcomes).

    Expected clusters (for La Liga):
    - High density, low tempo → possession-based
    - Low density, high flow efficiency → direct/counter-attacking
    - High resilience spread → collective pressing teams
    - High centralisation → one-player-dependent teams

    Returns DataFrame with team, match, cluster_label.
    """
```

Validation: Manually inspect if Barcelona consistently lands in one cluster,
Atletico Madrid in another. Use your football knowledge.

#### 4.3 Model 2 — Match Outcome Prediction (Supervised)

```python
def train_outcome_model(
    feature_matrix: pd.DataFrame
) -> dict:
    """
    Predict match result (W/D/L) from pre/during-match network features.
    Use first-half features only to predict full-match outcome
    (more interesting and realistic than using full-match data).

    Model: RandomForestClassifier (interpretable, handles small datasets)
    Evaluation: Stratified 5-fold cross-validation
    Metrics: accuracy, weighted F1, confusion matrix

    Also compute feature importances — which network metric is most
    predictive of winning?

    Returns: {'model': fitted_model, 'cv_scores': array,
              'feature_importances': DataFrame}
    """
```

**Important:** Do not overfit. La Liga 2015/16 has ~380 matches × 2 teams = 760
samples. With ~12 features this is manageable. Report CV scores honestly — even
55% accuracy on W/D/L (vs 33% random) is meaningful and publishable.

#### 4.4 Model 3 — Tactical Shift Detection (Unsupervised, Time Series)

```python
def detect_tactical_shifts(
    dynamic_networks: list[tuple[float, nx.DiGraph]],
    threshold: float = 2.0
) -> list[float]:
    """
    Use change-point detection on the time series of network metrics
    to identify when a team's tactical structure changed (formation shift,
    substitution effect, red card response, etc.)

    Method:
    1. For each time-window graph, compute [density, top_betweenness,
       n_nodes] as a feature vector
    2. Compute cosine distance between consecutive feature vectors
    3. Flag timestamps where distance > mean + threshold * std as shift points

    Returns list of timestamps (seconds) where tactical shifts were detected.
    """
```

Validation: Cross-reference detected shifts with known events (substitutions,
goals, red cards) from the raw event data. Expect shifts within ±3 minutes of
major game events.

#### 4.5 Write `tests/test_ml.py`
Tests must cover:
- Feature matrix has no NaN values (imputation must be applied)
- Cluster labels are integers in range [0, n_clusters-1]
- CV scores array has length == n_folds
- `detect_tactical_shifts` returns timestamps within [0, 5400] seconds

### ✅ Phase 4 Exit Criteria — ALL must pass before Phase 5

```
[ ] pytest tests/test_ml.py — all tests GREEN
[ ] Feature matrix saved and verified: no NaN, correct shape
[ ] Clustering: Barcelona and Atletico Madrid consistently in different clusters
    (Human to verify — this is a domain knowledge check)
[ ] Outcome model: CV accuracy > 40% on W/D/L (if < 40%, feature bug likely)
[ ] Feature importances: inspect top 3 features — do they make football sense?
    (Human approval required)
[ ] Tactical shift detection: for one manually chosen match, shifts are within
    ±3 minutes of a known event (goal, red card, or substitution)
    (Human to pick match and verify using match timeline)
[ ] Notebook 04_ml_models.ipynb runs top-to-bottom without errors
[ ] Git commit: "feat: phase 4 — ML models"
```

**⛔ Do not proceed to Phase 5 until all criteria are checked.**

---

## ✅ Phase 5 — Visualisations

**Objective:** Publication-quality and blog-ready figures. Each visualisation
must tell a clear story — label everything, assume the reader has no prior
context.

### Tasks

#### 5.1 Implement `src/tactical_topology/viz.py`

```python
def plot_passing_network_on_pitch(
    G: nx.DiGraph,
    node_positions: dict,
    team: str,
    match_label: str,
    highlight_node: str = None,    # Optional: highlight one player
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Draw passing network overlaid on a football pitch using mplsoccer.
    - Node size ∝ betweenness centrality
    - Edge width ∝ pass count
    - Edge colour ∝ pass completion rate (gradient: red → green)
    - Node colour: gold for top betweenness, grey for others
    """

def plot_resilience_bar(
    resilience_df: pd.DataFrame,
    team: str,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart of resilience_score per player.
    Sorted descending. Annotate the top player as "Key Node".
    """

def plot_zone_flow_heatmap(
    zone_graph: nx.DiGraph,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Pitch heatmap showing pass flow intensity between zones.
    Overlay the min-cut zones with a red border.
    """

def plot_dynamic_network_metrics(
    dynamic_networks: list[tuple[float, nx.DiGraph]],
    shift_timestamps: list[float],
    team: str,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Line chart of network density and betweenness over match time.
    Vertical dashed lines at detected tactical shift timestamps.
    """

def plot_pressure_comparison(
    degradation_dict: dict,
    team: str,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Side-by-side comparison: free vs pressed network metrics.
    Use a radar/spider chart for multi-metric comparison.
    """

def plot_cluster_scatter(
    feature_matrix: pd.DataFrame,
    cluster_labels: pd.Series,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    2D PCA scatter of teams coloured by cluster.
    Annotate notable teams (Barcelona, Real Madrid, Atletico).
    """
```

#### 5.2 Figure export standards
- All figures saved as PNG (300 DPI) and SVG to `figures/`
- Consistent colour palette throughout (define once in `viz.py`)
- All axes labelled with units
- All figures include a title and a one-line caption below it
- Figure filenames: `fig_<phase>_<description>.png`

### ✅ Phase 5 Exit Criteria — Project Complete

```
[ ] All 6 plot functions execute without errors
[ ] figures/ directory contains at least 6 PNG + 6 SVG files
[ ] Notebook 05_visualisations.ipynb runs top-to-bottom without errors
[ ] Human approval: review all figures for clarity and correctness
[ ] Passing network pitch plot: visually distinguishable from basic NetworkX plots
[ ] Pressure comparison radar: clearly shows degradation (not identical shapes)
[ ] Git commit: "feat: phase 5 — visualisations"
[ ] Final git commit: "docs: update README with project summary and figures"
```

---

## 🔁 Cross-Phase Rules for Claude Code

These apply throughout all phases:

1. **Never skip a phase.** If a later phase requires data from an earlier one,
   go back and extend the earlier module — don't add workarounds inline.

2. **Tests before notebooks.** Unit tests must pass before a notebook is written.
   Notebooks are for exploration and presentation, not for debugging logic.

3. **No magic numbers.** All constants (e.g. `min_passes=2`, `window_seconds=300`)
   must be function parameters with documented defaults. No hardcoded values
   inside function bodies.

4. **Fail loudly.** If a function receives data it cannot handle (e.g. a team
   with < 5 players in a network), raise a `ValueError` with a descriptive
   message. Do not silently return empty results.

5. **Log progress.** Use `tqdm` for any loop over matches. Use `print` statements
   with `[Phase X]` prefixes for major steps. This makes debugging easy.

6. **Data contracts.** Every function that takes a DataFrame must validate its
   required columns at the start and raise `KeyError` with a helpful message
   if they're missing.

7. **Commit at each phase end.** The git log is the audit trail. One commit per
   phase minimum, with a descriptive message.

---

## 📝 README.md (To Be Written in Phase 5)

The final README must include:
- Project summary (3 sentences)
- The "Football as a Network" concept explanation (for non-engineers)
- Installation instructions (`pip install -r requirements.txt`)
- Quick start (one command to run the full pipeline)
- One key figure embedded inline
- Link to the blog post (to be added after publication)
- Citation format for academic use

---

## 🎯 Final Deliverables

| Deliverable | Location | Purpose |
|---|---|---|
| Python package | `src/tactical_topology/` | Reusable, importable code |
| 5 Jupyter notebooks | `notebooks/` | Reproducible analysis narrative |
| Test suite | `tests/` | Correctness guarantee |
| Figures | `figures/` | Blog post + paper |
| Feature matrix | `data/processed/` | For community to build on |
| README | root | GitHub landing page |
| This PLAN.md | root | Transparency + reproducibility |
