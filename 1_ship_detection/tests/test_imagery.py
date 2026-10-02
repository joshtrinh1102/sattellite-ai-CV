"""Component 1 - imagery and proposal tests that do not need TensorFlow."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import config
from vision.detector import propose_candidates
from vision.imagery import covers, stretch, usable


def test_outputs_go_to_handoff():
    """The ship counts and overlays are the only things 3_ranking may read."""
    assert config.SHIP_COUNTS.parent == config.HANDOFF
    assert config.DETECTIONS.parent == config.HANDOFF


def test_covers_requires_full_containment():
    aoi = (-118.32, 33.58, -118.06, 33.78)
    # 11SLT contains the AOI; 11SMT starts east of its west edge and does not.
    assert covers((-119.17, 33.33, -117.97, 34.34), aoi)
    assert not covers((-118.09, 33.35, -116.89, 34.34), aoi)


def test_usable_rejects_hazy_scenes():
    # measured values: clear dates sit at frac 0.78 / median 7-12
    assert usable({"water_frac": 0.78, "water_median": 11.7})
    # marine layer lifts the water brightness and shrinks the mask
    assert not usable({"water_frac": 0.25, "water_median": 51.0})
    assert not usable({"water_frac": 0.78, "water_median": 24.3})


def test_stretch_is_fixed_not_per_scene():
    """Two scenes of differing overall brightness must map the same input DN to
    the same output, or counts stop being comparable between dates."""
    dark = np.full((4, 4, 3), 10, dtype="uint8")
    mixed = np.full((4, 4, 3), 10, dtype="uint8")
    mixed[0, 0] = 200
    assert stretch(dark)[1, 1, 0] == stretch(mixed)[1, 1, 0]


def test_propose_candidates_finds_a_bright_blob_on_water():
    img = np.full((200, 200, 3), 10, dtype="uint8")
    img[100:106, 90:120] = 90          # a vessel-shaped bright patch
    mask = np.ones((200, 200), dtype=bool)
    cands = propose_candidates(img, mask, contrast=12.0, min_area=6)
    assert len(cands) == 1
    cx, cy, area = cands[0]
    assert 90 <= cx <= 120 and 98 <= cy <= 108


def test_propose_candidates_ignores_blobs_off_water():
    img = np.full((200, 200, 3), 10, dtype="uint8")
    img[100:106, 90:120] = 90
    mask = np.zeros((200, 200), dtype=bool)   # nothing is water
    assert propose_candidates(img, mask, contrast=12.0, min_area=6) == []
