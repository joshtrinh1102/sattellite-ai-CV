"""Smoke tests for Component 1 helpers that do not need TensorFlow."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.detector import nms
from vision.occupancy import (
    fit_scale,
    fixed_thresholds,
    ship_density,
    water_area_km2,
)


def test_nms_collapses_overlapping_boxes():
    boxes = [
        [0, 0, 100, 100, 0.9],
        [5, 5, 105, 105, 0.8],   # ~85% IoU with the first, should be dropped
        [500, 500, 600, 600, 0.7],
    ]
    kept = nms(boxes, iou_threshold=0.15)
    assert len(kept) == 2
    assert kept[0][4] == 0.9


def test_nms_handles_empty():
    assert nms([]) == []


def test_water_area_accounts_for_upscaling():
    # Upscaling 4x means each scaled pixel covers 1/4 the ground distance,
    # so 4x the pixels describe the same area.
    unscaled = water_area_km2(1_000_000, metres_per_pixel=10, scale_factor=1)
    scaled = water_area_km2(16_000_000, metres_per_pixel=10, scale_factor=4)
    assert unscaled == pytest.approx(scaled)


def test_density_rejects_zero_area():
    with pytest.raises(ValueError):
        ship_density(5, 0)


def test_fit_scale_buckets_by_quantile():
    scale = fit_scale(np.arange(1, 101))
    assert scale.label(5) == "low"
    assert scale.label(50) == "medium"
    assert scale.label(95) == "high"


def test_fit_scale_needs_enough_baseline():
    with pytest.raises(ValueError, match="10 baseline"):
        fit_scale([1, 2, 3])


def test_fixed_thresholds_are_inclusive_of_upper_bucket():
    scale = fixed_thresholds(low_cut=2.0, high_cut=5.0)
    assert scale.label(1.9) == "low"
    assert scale.label(2.0) == "medium"
    assert scale.label(5.0) == "high"
