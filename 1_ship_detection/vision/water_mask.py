"""Water/land segmentation for port scenes.

Lifted verbatim from the notebook's "Water Mask" cell. The mask is what stops
the sliding-window classifier from firing on parked trucks, container stacks
and rooftops, which look a lot like ships at 80px.
"""

import cv2
import numpy as np


def build_water_mask(image_bgr, debug=False):
    h, w = image_bgr.shape[:2]
    b = image_bgr[:, :, 0].astype("float32")
    g = image_bgr[:, :, 1].astype("float32")
    r = image_bgr[:, :, 2].astype("float32")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blue_dom = ((b - r) > 3) & ((b - g) > -5)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype("float32")
    k = np.ones((9, 9), dtype="float32") / 81.0
    mean = cv2.filter2D(gray, -1, k)
    mean_sq = cv2.filter2D(gray ** 2, -1, k)
    local_std = np.sqrt(np.maximum(0, mean_sq - mean ** 2))
    low_tex = local_std < 15.0
    hsv_water = ((hue >= 60) & (hue <= 160) & (val >= 40) & (val <= 210) & (sat < 140))
    raw = blue_dom & low_tex & hsv_water

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(raw.astype("uint8"), cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    min_area = int(0.005 * h * w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1

    if debug:
        print(f"  blue_dom={blue_dom.mean():.1%}  low_tex={low_tex.mean():.1%}  "
              f"hsv_water={hsv_water.mean():.1%}  combined={raw.mean():.1%}  "
              f"after_morph={out.mean():.1%}")
    return out.astype(bool)


def build_water_mask_s2(image_bgr, debug=False):
    """Water mask for raw Sentinel-2 true-colour imagery.

    `build_water_mask` above was tuned on a bright Google Earth screenshot and
    finds nothing here: Sentinel-2 TCI is radiometrically raw, so open water
    sits at DN 6-20 with median value 15, well below that mask's `val >= 40`
    floor, and its saturation runs past the `sat < 140` ceiling. Measured on
    the 2021-10-10 scene, the aerial mask returns 0.0% water.

    Over water the useful facts are simpler than in the aerial case: it is
    dark, it is never red-dominant (water absorbs NIR and red strongly), and
    it is smooth. Land in this AOI is dense urban port, which is bright.

    Run this on the RAW scene, before any contrast stretch - the DN
    thresholds below are absolute, which is what keeps the mask consistent
    from date to date.
    """
    h, w = image_bgr.shape[:2]
    b = image_bgr[:, :, 0].astype("float32")
    r = image_bgr[:, :, 2].astype("float32")
    grey = image_bgr.mean(axis=2).astype("float32")

    k = np.ones((9, 9), dtype="float32") / 81.0
    mean = cv2.filter2D(grey, -1, k)
    mean_sq = cv2.filter2D(grey ** 2, -1, k)
    local_std = np.sqrt(np.maximum(0, mean_sq - mean ** 2))

    dark = grey < 45.0
    not_red = b >= r
    smooth = local_std < 25.0
    raw = dark & not_red & smooth

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(raw.astype("uint8"), cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    # Keep only large basins. A ship sitting in open water gets closed over by
    # the 15px close above, so this does not punch holes where the ships are.
    min_area = int(0.005 * h * w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1

    if debug:
        print(f"  dark={dark.mean():.1%}  not_red={not_red.mean():.1%}  "
              f"smooth={smooth.mean():.1%}  combined={raw.mean():.1%}  "
              f"after_morph={out.mean():.1%}")
    return out.astype(bool)
