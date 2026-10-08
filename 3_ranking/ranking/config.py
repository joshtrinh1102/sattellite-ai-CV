"""Paths for Component 3. Everything resolves from this component's folder.

Inputs come only from HANDOFF - this folder never reads 1_ship_detection or
2_factors directly.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 3_ranking/
HANDOFF = ROOT.parent / "handoff"
FACTORS_DIR = ROOT.parent / "2_factors"   # only the dashboard's Update button runs it

# Inputs (handoff contract - see handoff/README.md)
SHIP_COUNTS = HANDOFF / "ship_counts.csv"
FACTORS_DAILY = HANDOFF / "factors_daily.csv"
DETECTIONS = HANDOFF / "detections"

# The 12 headline topics from 2_factors, folded into broader analytical
# dimensions. With ~76 capture dates, 12 sparse topic shares compete for very
# little signal; six dimensions are what the sample can plausibly support.
# Grouping happens here, on summed daily counts, so 2_factors is untouched.
TOPIC_GROUPS = {
    "economy_demand": ["inflation_econ", "demand_volatility"],
    "supply_capacity": ["material_shortages", "labor_shortages", "operational_risks"],
    "port_flow": ["logistics_reliability"],
    "policy_geopolitics": ["global_regulations", "conflict_war"],
    "external_shocks": ["natural_disasters", "health_pandemic"],
    "digital_reputation": ["cybersecurity", "reputation_risks"],
}

# Outputs
OUTPUTS = ROOT / "outputs"
FACTOR_RANKING = OUTPUTS / "factor_ranking.csv"
STUDY_FEATURES = OUTPUTS / "study_features.csv"   # the design matrix modelled
STUDY_RESULT = OUTPUTS / "study_result.json"
RESULTS_MD = OUTPUTS / "results.md"
