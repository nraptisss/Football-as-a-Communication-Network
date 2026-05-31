"""Project-wide configuration constants.

The working dataset and the fixed set of ground-truth matches used to
validate every downstream phase are defined here so they are recorded in
one place and never hardcoded inside function bodies (see PLAN_1.md
cross-phase rule 3).
"""

# --- Working dataset: La Liga, Season 2015/16 ---
COMPETITION_ID: int = 11
SEASON_ID: int = 27

# --- Pitch dimensions (StatsBomb coordinate system) ---
PITCH_LENGTH: float = 120.0  # x-axis, defensive -> attacking
PITCH_WIDTH: float = 80.0    # y-axis

# --- Ground-truth matches ---
# A fixed set of 5 La Liga 2015/16 matches used as the validation baseline
# for all subsequent phases. Chosen to include Barcelona, Atletico Madrid
# and Real Madrid so later phases can compare possession vs. counter-attacking
# styles against football intuition.
#
#   265839  : Barcelona 2-1 Sevilla
#   266166  : Atletico Madrid 1-2 Barcelona
#   266424  : Real Madrid 0-4 Barcelona (El Clasico)
#   3825567 : Sporting Gijon 0-0 Real Madrid
#   3825563 : Atletico Madrid 1-0 Las Palmas
GROUND_TRUTH_MATCH_IDS: list[int] = [
    265839,
    266166,
    266424,
    3825567,
    3825563,
]

# Default cache directory for processed parquet outputs.
PROCESSED_DIR: str = "data/processed/"
