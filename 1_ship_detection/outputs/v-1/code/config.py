"""Project paths and environment configuration.

Everything resolves from the repo root (code/ sits one level below it), so code behaves the same whether it is
run from the root or from a notebook.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The split is between what goes IN and what comes OUT.
#
# data/    inputs and intermediates - imagery, model weights, chips, cached API
#          responses. Large, mostly regenerable, gitignored.
# results/ findings - the tables the write-up cites, the annotated detections,
#          the detector's held-out scorecard, and results.md itself. Small
#          enough to version, and the only directory anyone needs to read to
#          see what the study concluded.
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DETECTIONS = RESULTS / "detections"   # annotated scenes, one per capture date
NEWS_RAW = DATA / "news_raw"   # untouched GDELT responses, one file per day
DOCS = RESULTS / "docs"

# Kept as an alias so older notebook cells that import OUTPUTS still resolve.
OUTPUTS = RESULTS

SHIP_DETECTOR = DATA / "ship_detector_fixed.keras"
SHIP_DETECTOR_S2 = DATA / "ship_detector_s2.keras"

# Result tables (CSV rather than parquet - they are tens of KB, and being
# readable in a diff is worth more here than columnar storage).
OCCUPANCY = RESULTS / "occupancy.csv"
ARTICLES = RESULTS / "articles.csv"
RISK_DAILY = RESULTS / "risk_daily.csv"
STUDY_FEATURES = RESULTS / "study_features.csv"
FACTOR_RANKING = RESULTS / "factor_ranking.csv"
STUDY_RESULT = RESULTS / "study_result.json"
DETECTOR_REPORT = RESULTS / "ship_detector_s2.report.json"
RESULTS_MD = RESULTS / "results.md"


def env(name, default=None, required=False):
    """Read an API key or setting from the environment (see settings/.env.example)."""
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Copy settings/.env.example to .env and fill it in."
        )
    return value
