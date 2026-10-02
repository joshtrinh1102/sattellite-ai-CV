"""Paths for Component 1. Everything resolves from this component's folder.

The only path that leaves the folder is HANDOFF, where the ship counts and the
annotated scenes are written for 3_ranking to read.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 1_ship_detection/
HANDOFF = ROOT.parent / "handoff"

DATA = ROOT / "data"           # scenes, labelled chips, the aerial test image
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"     # single-image runs and frozen snapshots

SHIP_DETECTOR = MODELS / "ship_detector_fixed.keras"   # aerial (ShipsNet)
SHIP_DETECTOR_S2 = MODELS / "ship_detector_s2.keras"   # fine-tuned, Sentinel-2
DETECTOR_REPORT = MODELS / "ship_detector_s2.report.json"

# Handoff contract - see handoff/README.md
SHIP_COUNTS = HANDOFF / "ship_counts.csv"
DETECTIONS = HANDOFF / "detections"
