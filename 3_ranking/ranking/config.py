"""Paths for Component 3. Everything resolves from this component's folder.

Inputs come only from HANDOFF - this folder never reads 1_ship_detection or
2_factors directly.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 3_ranking/
HANDOFF = ROOT.parent / "handoff"

# Inputs (handoff contract - see handoff/README.md)
SHIP_COUNTS = HANDOFF / "ship_counts.csv"
FACTORS_DAILY = HANDOFF / "factors_daily.csv"
DETECTIONS = HANDOFF / "detections"

# Outputs
OUTPUTS = ROOT / "outputs"
FACTOR_RANKING = OUTPUTS / "factor_ranking.csv"
STUDY_FEATURES = OUTPUTS / "study_features.csv"   # the design matrix modelled
STUDY_RESULT = OUTPUTS / "study_result.json"
RESULTS_MD = OUTPUTS / "results.md"
