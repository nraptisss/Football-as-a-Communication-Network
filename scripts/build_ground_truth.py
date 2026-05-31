"""Build the Phase 1 ground-truth parquet caches.

Loads and cleans pass events for the 5 fixed ground-truth matches
(config.GROUND_TRUTH_MATCH_IDS) and writes them to data/processed/.
Run with: python scripts/build_ground_truth.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.loader import cache_match_passes


def main() -> None:
    cache = cache_match_passes(config.GROUND_TRUTH_MATCH_IDS, config.PROCESSED_DIR)
    print("[Phase 1] Cached ground-truth pass events:")
    for match_id, df in cache.items():
        print(f"[Phase 1]   match {match_id}: {len(df)} passes, "
              f"{df['team'].nunique()} teams")


if __name__ == "__main__":
    main()
