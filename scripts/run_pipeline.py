"""Run the full Tactical Topology pipeline end-to-end.

Steps (each is idempotent / cached):
  1. Cache cleaned pass events for the 5 ground-truth matches (Phase 1)
  2. Compute telecom metrics for those matches (Phase 3)
  3. Build the full-season feature matrices (Phase 4)
  4. Generate all publication figures (Phase 5)

Usage: python scripts/run_pipeline.py
Note: step 3 downloads/loads all ~380 matches on first run (several minutes).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tactical_topology import config
from tactical_topology.loader import cache_match_passes
from tactical_topology.ml import build_feature_matrix

HERE = os.path.dirname(__file__)
PROC = config.PROCESSED_DIR


def main() -> None:
    print("[pipeline] 1/4 caching ground-truth pass events")
    cache_match_passes(config.GROUND_TRUTH_MATCH_IDS, PROC)

    print("[pipeline] 2/4 telecom metrics")
    import build_telecom_metrics
    build_telecom_metrics.main()

    print("[pipeline] 3/4 full-season feature matrices")
    build_feature_matrix(half=None, cache_path=os.path.join(PROC, "feature_matrix.parquet"))
    build_feature_matrix(half=1, cache_path=os.path.join(PROC, "feature_matrix_firsthalf.parquet"))

    print("[pipeline] 4/4 figures")
    import build_notebook_05
    build_notebook_05.main()

    print("[pipeline] done.")


if __name__ == "__main__":
    main()
