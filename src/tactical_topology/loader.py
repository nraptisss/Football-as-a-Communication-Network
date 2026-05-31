"""Phase 1: data loading & cleaning.

Loads StatsBomb open data via ``statsbombpy`` (default ``flatten_attrs=True``)
and applies a fixed set of cleaning rules so every downstream phase reads a
stable, validated schema. Public API:

    load_match_events       - all cleaned events for one match
    load_pass_events        - cleaned pass events only (validated schema)
    load_season_matches     - match metadata for a full season
    load_and_cache_season_events - load + cache all season pass events

See PLAN_1.md Phase 1 for the cleaning-rule specification.
"""

import os

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from . import config

# --- Schema ----------------------------------------------------------------
#
# statsbombpy flattens nested {id, name} attributes to the *name* string under
# the base key (e.g. ``type`` already holds "Pass"). We rename these to an
# explicit ``*_name`` schema so column meaning is unambiguous downstream.
_RAW_RENAME = {
    "type": "type_name",
    "player": "player_name",
    "pass_recipient": "pass_recipient_name",
    "pass_outcome": "pass_outcome_name",
}

# Raw columns we depend on existing in the statsbombpy output (data contract).
_RAW_REQUIRED = ["id", "index", "period", "timestamp", "type", "team",
                 "team_id", "player", "location", "under_pressure"]

# Final validated schema for pass events (the contract for Phase 2+).
PASS_REQUIRED_COLUMNS = [
    "id", "index", "period", "timestamp", "type_name", "team", "player_name",
    "location", "pass_end_location", "pass_recipient_name", "pass_outcome_name",
    "under_pressure",
]

# Columns that must never be NaN in a clean pass row.
_PASS_NON_NULL = ["location", "pass_end_location", "player_name", "team"]

# Per-period offset (45 min) applied so match ``seconds`` do not overlap across
# periods. Used only as an additive offset; deltas within a period are exact.
_SECONDS_PER_PERIOD: float = 2700.0


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise KeyError if any required column is missing (data contract)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{context}: missing required columns {missing}. "
            f"Got columns: {sorted(df.columns)}"
        )


def _timestamp_to_seconds(timestamp: str) -> float:
    """Convert a StatsBomb 'HH:MM:SS.mmm' timestamp to seconds (float)."""
    hours, minutes, secs = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def load_season_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return match metadata for a full season.

    Raises ValueError if the season has no matches (fail loudly).
    """
    print(f"[Phase 1] Loading season matches "
          f"(competition_id={competition_id}, season_id={season_id})")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if matches is None or len(matches) == 0:
        raise ValueError(
            f"No matches found for competition_id={competition_id}, "
            f"season_id={season_id}."
        )
    return matches


def load_match_events(match_id: int, periods: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Load and clean all events for a single match.

    Cleaning rules applied in order (see PLAN_1.md 1.2):
      2. Standardise team names: keep integer ``team_id`` alongside ``team``.
      3. Period filter: keep only ``periods`` (default first & second half).
      4. Timestamp -> ``seconds`` (float), continuous across periods.
      6. ``under_pressure`` coerced to bool (NaN -> False).

    Rules 1 and 5 (null-location drop, ``pass_complete``) are pass-specific and
    applied in :func:`load_pass_events`.
    """
    raw = sb.events(match_id=match_id)
    _require_columns(raw, _RAW_REQUIRED, f"load_match_events(match_id={match_id})")

    df = raw.rename(columns=_RAW_RENAME).copy()

    # Rule 3 - period filter.
    df = df[df["period"].isin(periods)].copy()
    if df.empty:
        raise ValueError(
            f"Match {match_id} has no events in periods {periods}."
        )

    # Rule 2 - integer team_id alongside the team string.
    df["team_id"] = df["team_id"].astype("int64")

    # Rule 4 - timestamp to continuous match seconds.
    df["seconds"] = (
        (df["period"] - 1) * _SECONDS_PER_PERIOD
        + df["timestamp"].map(_timestamp_to_seconds)
    )

    # Rule 6 - pressure flag as bool (StatsBomb stores True or NaN).
    df["under_pressure"] = df["under_pressure"] == True  # noqa: E712

    return df


def load_pass_events(match_id: int, periods: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Return only pass events for a match, with the validated schema.

    Additional pass-specific cleaning (see PLAN_1.md 1.2):
      1. Drop passes with NaN ``location`` or ``pass_end_location`` (and any
         row missing ``player_name``/``team``) - unusable in network analysis.
      5. Keep incomplete passes but add boolean ``pass_complete``
         (``pass_complete = pass_outcome_name.isna()``).
    """
    events = load_match_events(match_id, periods=periods)
    _require_columns(
        events,
        PASS_REQUIRED_COLUMNS,
        f"load_pass_events(match_id={match_id})",
    )

    passes = events[events["type_name"] == "Pass"].copy()

    # Rule 1 - drop rows missing structural fields needed for the network.
    before = len(passes)
    passes = passes.dropna(subset=_PASS_NON_NULL).copy()
    dropped = before - len(passes)
    if dropped:
        print(f"[Phase 1]   match {match_id}: dropped {dropped} pass rows "
              f"with null {_PASS_NON_NULL}")

    if passes.empty:
        raise ValueError(
            f"Match {match_id} produced zero valid pass events after cleaning."
        )

    # Rule 5 - completion flag (a NaN outcome means the pass was completed).
    passes["pass_complete"] = passes["pass_outcome_name"].isna()

    # Final defensive check on the no-null contract.
    for col in _PASS_NON_NULL:
        if passes[col].isna().any():
            raise ValueError(
                f"Match {match_id}: column '{col}' still contains NaN after "
                f"cleaning."
            )

    keep = PASS_REQUIRED_COLUMNS + ["team_id", "seconds", "pass_complete"]
    return passes[keep].reset_index(drop=True)


def load_and_cache_season_events(
    competition_id: int,
    season_id: int,
    cache_dir: str = config.PROCESSED_DIR,
) -> dict[int, pd.DataFrame]:
    """Load all match pass events for a season, caching each as parquet.

    Returns a dict mapping ``match_id -> cleaned pass DataFrame``. Existing
    parquet caches are reused rather than re-downloaded.
    """
    matches = load_season_matches(competition_id, season_id)
    match_ids = [int(m) for m in matches["match_id"].tolist()]
    return cache_match_passes(match_ids, cache_dir=cache_dir)


def cache_match_passes(
    match_ids: list[int],
    cache_dir: str = config.PROCESSED_DIR,
) -> dict[int, pd.DataFrame]:
    """Load and cache cleaned pass events for a specific list of matches.

    Writes ``passes_<match_id>.parquet`` into ``cache_dir`` and returns a dict
    mapping ``match_id -> DataFrame``. Reuses an existing cache file if present.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[int, pd.DataFrame] = {}
    for match_id in tqdm(match_ids, desc="[Phase 1] caching pass events"):
        path = os.path.join(cache_dir, f"passes_{match_id}.parquet")
        if os.path.exists(path):
            out[match_id] = pd.read_parquet(path)
            continue
        passes = load_pass_events(match_id)
        passes.to_parquet(path, index=False)
        out[match_id] = passes
    return out
