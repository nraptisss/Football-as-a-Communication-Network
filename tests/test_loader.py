"""Tests for tactical_topology.loader (Phase 1).

These tests hit StatsBomb open data via statsbombpy (cached locally by
requests-cache after first run). They validate the cleaning contract on a
single ground-truth match.
"""

import pandas as pd
import pytest

from tactical_topology import config
from tactical_topology.loader import load_pass_events

MATCH_ID = config.GROUND_TRUTH_MATCH_IDS[0]


@pytest.fixture(scope="module")
def passes() -> pd.DataFrame:
    return load_pass_events(MATCH_ID)


def test_only_pass_events(passes: pd.DataFrame) -> None:
    assert (passes["type_name"] == "Pass").all()


def test_no_nan_in_required_columns(passes: pd.DataFrame) -> None:
    for col in ["location", "pass_end_location", "player_name", "team"]:
        assert not passes[col].isna().any(), f"NaN found in {col}"


def test_seconds_numeric_and_in_range(passes: pd.DataFrame) -> None:
    assert pd.api.types.is_numeric_dtype(passes["seconds"])
    assert passes["seconds"].between(0, 6000).all()


def test_pass_complete_is_boolean(passes: pd.DataFrame) -> None:
    assert passes["pass_complete"].dtype == bool


def test_under_pressure_is_boolean_no_nan(passes: pd.DataFrame) -> None:
    assert passes["under_pressure"].dtype == bool
    assert not passes["under_pressure"].isna().any()


def test_completion_flag_matches_outcome(passes: pd.DataFrame) -> None:
    # A completed pass is exactly one with a null outcome.
    assert (passes["pass_complete"] == passes["pass_outcome_name"].isna()).all()
