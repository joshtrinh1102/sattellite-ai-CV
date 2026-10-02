"""Turn raw ship counts into the low / medium / high occupancy label the
requirements ask for.

Two normalisations matter before bucketing, because raw counts are not
comparable across scenes:

  * density  - ships per km2 of water in the ROI, so a wider crop does not
               automatically look busier.
  * baseline - quantiles are fitted on a reference window (a "normal times"
               period) and then applied to every scene, so a COVID-era scene
               can legitimately land in `low` instead of being re-centred to
               medium by its own distribution.

The cut points are quantiles by default; swap in `fixed_thresholds` once
there is a domain-anchored definition of what "high" means for a given port.
"""

from dataclasses import dataclass

import numpy as np

LABELS = ("low", "medium", "high")


@dataclass
class OccupancyScale:
    """Fitted cut points mapping a density to a LABELS bucket."""
    low_cut: float
    high_cut: float
    unit: str = "ships_per_km2"

    def label(self, density):
        if density < self.low_cut:
            return "low"
        if density < self.high_cut:
            return "medium"
        return "high"


def water_area_km2(water_area_px, metres_per_pixel, scale_factor=1):
    """Water pixels -> km2. `metres_per_pixel` is the ground resolution of the
    *source* image; `scale_factor` is the upscaling the detector applied."""
    effective_mpp = metres_per_pixel / scale_factor
    return water_area_px * (effective_mpp ** 2) / 1e6


def ship_density(n_ships, water_area_km2_value):
    if water_area_km2_value <= 0:
        raise ValueError("water area must be positive to compute density")
    return n_ships / water_area_km2_value


def fit_scale(baseline_densities, quantiles=(0.33, 0.67)):
    """Fit cut points on a baseline period (e.g. 2018-2019 'normal times').

    Fit this ONCE on the reference window and reuse the returned scale for all
    other periods - refitting per period would erase the very signal the COVID
    comparison is looking for.
    """
    arr = np.asarray(list(baseline_densities), dtype="float64")
    if arr.size < 10:
        raise ValueError(
            f"need >=10 baseline observations to fit quantiles, got {arr.size}"
        )
    low_cut, high_cut = np.quantile(arr, quantiles)
    return OccupancyScale(low_cut=float(low_cut), high_cut=float(high_cut))


def fixed_thresholds(low_cut, high_cut):
    """Use domain-set cut points instead of fitted quantiles."""
    return OccupancyScale(low_cut=float(low_cut), high_cut=float(high_cut))
