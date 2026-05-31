"""Build the full-season Phase 4 feature matrices.

Computes two matrices for La Liga 2015/16 (all ~380 matches x 2 teams):
  - data/processed/feature_matrix.parquet            (full-match, for clustering)
  - data/processed/feature_matrix_firsthalf.parquet  (period==1, for outcome model)

Run with: python scripts/build_feature_matrix.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.ml import build_feature_matrix

FULL = os.path.join(config.PROCESSED_DIR, "feature_matrix.parquet")
FIRST_HALF = os.path.join(config.PROCESSED_DIR, "feature_matrix_firsthalf.parquet")


def main() -> None:
    full = build_feature_matrix(half=None, cache_path=FULL)
    print(f"[Phase 4] full-match matrix: {full.shape} -> {FULL}")
    fh = build_feature_matrix(half=1, cache_path=FIRST_HALF)
    print(f"[Phase 4] first-half matrix: {fh.shape} -> {FIRST_HALF}")


if __name__ == "__main__":
    main()
