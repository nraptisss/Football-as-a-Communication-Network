"""Phase 0 verification script.

Checks StatsBomb open-data access and the working dataset, per the Phase 0
exit criteria in PLAN_1.md. Run with: python scripts/verify_phase0.py
"""

from statsbombpy import sb

# Working dataset: La Liga, Season 2015/16
COMPETITION_ID = 11
SEASON_ID = 27
EXPECTED_MATCHES = 380

REQUIRED_EVENT_COLUMNS = [
    "id", "index", "period", "timestamp", "type", "team", "player",
    "location", "pass", "carry", "shot", "under_pressure",
]


def main() -> None:
    print("[Phase 0] Checking sb.competitions() ...")
    competitions = sb.competitions()
    assert len(competitions) > 0, "competitions() returned empty DataFrame"
    print(f"[Phase 0]   competitions: {len(competitions)} rows, "
          f"{competitions['competition_name'].nunique()} unique competitions")

    print("[Phase 0] Checking La Liga 2015/16 matches ...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    print(f"[Phase 0]   matches returned: {len(matches)}")
    assert len(matches) == EXPECTED_MATCHES, (
        f"expected {EXPECTED_MATCHES} matches, got {len(matches)}"
    )

    sample_match_id = int(matches["match_id"].iloc[0])
    print(f"[Phase 0] Checking sb.events(match_id={sample_match_id}) ...")
    # flatten_attrs=False keeps nested StatsBomb attributes (pass, carry, shot)
    # as single object columns, matching the raw event schema.
    events = sb.events(match_id=sample_match_id, flatten_attrs=False)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events.columns]
    assert not missing, f"events missing required columns: {missing}"
    print(f"[Phase 0]   events: {len(events)} rows, all required columns present")

    print("[Phase 0] ALL DATA-ACCESS CHECKS PASSED")


if __name__ == "__main__":
    main()
