"""Sliding-window ship detector.

Run as a CLI:
    python -m scsai.detector --image ../data/la_port_scene.png


Water mask -> 80px patches over water only -> batched CNN inference -> NMS.
Same pipeline as the notebook's "Reusable Ship Bounding Box Detector" cell,
with the parameters pulled out into DetectorConfig so a run over many dates
uses one set of settings.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .water_mask import build_water_mask


@dataclass
class DetectorConfig:
    scale_factor: int = 4
    patch_size: int = 80
    stride: int = 20
    threshold: float = 0.6
    visual_multiplier: float = 1.5
    water_threshold: float = 0.35   # patch must be >= this fraction water to be scored
    batch_size: int = 256
    nms_iou: float = 0.15
    chunk_patches: int = 4096   # patches held in memory at once (see detect_ships)

    # Proposal path (see propose_candidates). Off by default so the aerial
    # pipeline from the notebook keeps behaving exactly as it did.
    proposals: bool = False
    proposal_min_area: int = 6      # source px; 6 px at 10 m/px is ~600 m2
    proposal_max_area: int = 4000   # rejects breakwater and land bleed
    proposal_contrast: float = 12.0   # raw DN above local background


def nms(boxes, iou_threshold=0.15):
    """Greedy non-max suppression. boxes: list of [x1, y1, x2, y2, conf]."""
    if not boxes:
        return []

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_threshold]
    return kept


def propose_candidates(image_bgr, water_mask, min_area=6, max_area=4000,
                       contrast=12.0, bg_kernel=41):
    """Bright-blob proposals on water, at source resolution.

    Why this exists: on a 25 km Sentinel-2 scene the sliding window evaluates
    ~24,000 water patches, of which all but a few dozen are empty ocean. On CPU
    that is ~6 minutes per date, so a 60-date study is a 6-hour run. Almost all
    of it is spent confirming that open water is open water.

    A vessel in Sentinel-2 true colour is a compact bright object on a dark,
    smooth background, so the candidates are cheap to find: subtract a
    morphological background (opened with a kernel wider than any ship, so the
    ships themselves are removed from it) and keep what stands proud of it.

    Run this on the RAW scene, not the stretched one. The stretch applies a
    gamma of 2.0, which expands exactly the low-DN range where water's sensor
    noise lives: on 2021-09-30 the same threshold yields 4,765 proposals from
    the stretched scene against 172 from the raw one, for a scene holding on
    the order of 50-90 vessels. On raw DNs the contrast is unambiguous - open
    water sits at 7-12 and hulls above 60 - and, as with the water mask, the
    thresholds stay absolute and therefore comparable between dates.

    This is a RECALL filter, not a detector. The threshold is deliberately
    loose, so wakes, whitecaps, buoys, breakwater rock and Sentinel-2's
    band-misregistration fringes all survive it. The CNN is what supplies
    precision by rejecting them. Keep that division of labour in mind when
    reading the numbers: the prescreen bounds recall, the classifier bounds
    precision.

    Returns a list of (cx, cy, area) in source-image pixels.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Opening with a kernel wider than a ship erases ships, leaving background.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel, bg_kernel))
    background = cv2.morphologyEx(grey, cv2.MORPH_OPEN, k)
    residual = cv2.subtract(grey, background)

    hot = (residual > contrast) & water_mask
    # Join the bow and stern of a vessel split by a dark midships band.
    hot = cv2.morphologyEx(hot.astype("uint8"), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(hot, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            out.append((float(cx), float(cy), int(area)))
    return out


def load_detector(model_path):
    """Load the trained Keras classifier. Imported lazily so that importing
    this module does not pull in TensorFlow."""
    from tensorflow.keras.models import load_model
    return load_model(str(model_path))


def detect_ships(image_path, model, crop_x_range=None, config=None, draw=True,
                 water_mask_fn=None, preprocess=None):
    """Find ship bounding boxes in a satellite image.

    crop_x_range: (start, end) as fractions of width, or None for the whole image.
    water_mask_fn: mask builder, default `build_water_mask` (aerial imagery).
                   Pass `build_water_mask_s2` for raw Sentinel-2 scenes.
    preprocess:    optional BGR->BGR transform applied to the pixels the
                   classifier sees, AFTER the mask is built from the raw
                   scene. Used to stretch Sentinel-2 into the training range.

    Returns a dict with:
      boxes       - [x1, y1, x2, y2, confidence] in *scaled ROI* pixel space
      roi_scaled  - the RGB region the boxes were detected on
      output_bgr  - roi_scaled (BGR) with boxes drawn, or None if draw=False
      water_mask  - boolean water mask for the ROI, at scaled resolution
      water_area_px - water pixels in the ROI, used to normalise ship counts
    """
    cfg = config or DetectorConfig()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    water_mask_full = (water_mask_fn or build_water_mask)(image_bgr)

    # The mask reads the raw scene; the classifier reads the preprocessed one.
    if preprocess is not None:
        image_rgb = cv2.cvtColor(preprocess(image_bgr), cv2.COLOR_BGR2RGB)

    if crop_x_range is not None:
        x_start = int(w_orig * crop_x_range[0])
        x_end = int(w_orig * crop_x_range[1])
    else:
        x_start, x_end = 0, w_orig

    roi_rgb = image_rgb[:, x_start:x_end]
    roi_mask = water_mask_full[:, x_start:x_end]
    # Proposals are found on RAW pixels, never on preprocessed ones - see
    # propose_candidates for why the distinction matters.
    roi_raw_bgr = image_bgr[:, x_start:x_end]

    roi_scaled = cv2.resize(
        roi_rgb, None, fx=cfg.scale_factor, fy=cfg.scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    # INTER_NEAREST for the mask - it is binary, interpolation would corrupt it
    mask_scaled = cv2.resize(
        roi_mask.astype("uint8"), None, fx=cfg.scale_factor, fy=cfg.scale_factor,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    h, w = roi_scaled.shape[:2]
    ps, stride = cfg.patch_size, cfg.stride

    boxes = []

    if cfg.proposals:
        # Proposal path: score one patch per bright blob instead of every
        # window. Proposals are found on the ROI at source resolution, then
        # mapped into scaled space, so a proposal centre lands at the centre of
        # its patch - which is how the ShipsNet training chips were framed.
        sf = cfg.scale_factor
        cands = propose_candidates(
            roi_raw_bgr, roi_mask,
            min_area=cfg.proposal_min_area,
            max_area=cfg.proposal_max_area,
            contrast=cfg.proposal_contrast,
        )
        patches, coords = [], []
        for cx, cy, _area in cands:
            x = int(round(cx * sf)) - ps // 2
            y = int(round(cy * sf)) - ps // 2
            x = max(0, min(w - ps, x))
            y = max(0, min(h - ps, y))
            patches.append(roi_scaled[y:y + ps, x:x + ps].astype("float32") / 255.0)
            coords.append((x, y))

        def score_proposals(chunk_patches, chunk_coords):
            if not chunk_patches:
                return
            arr = np.asarray(chunk_patches, dtype="float32")
            preds = model.predict(arr, batch_size=cfg.batch_size, verbose=0).flatten()
            for (x, y), conf in zip(chunk_coords, preds):
                if conf > cfg.threshold:
                    cx_, cy_ = x + ps // 2, y + ps // 2
                    half = int((ps * cfg.visual_multiplier) // 2)
                    boxes.append([max(0, cx_ - half), max(0, cy_ - half),
                                  min(w, cx_ + half), min(h, cy_ + half), float(conf)])

        for i in range(0, len(patches), cfg.chunk_patches):
            score_proposals(patches[i:i + cfg.chunk_patches],
                            coords[i:i + cfg.chunk_patches])

        final_boxes = nms(boxes, iou_threshold=cfg.nms_iou)
        output_bgr = None
        if draw:
            output_bgr = cv2.cvtColor(roi_scaled, cv2.COLOR_RGB2BGR)
            for (x1, y1, x2, y2, conf) in final_boxes:
                cv2.rectangle(output_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(output_bgr, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return {
            "boxes": final_boxes,
            "roi_scaled": roi_scaled,
            "output_bgr": output_bgr,
            "water_mask": mask_scaled,
            "water_area_px": int(mask_scaled.sum()),
            "n_proposals": len(cands),
        }

    def score(chunk_patches, chunk_coords):
        """Run one chunk through the model and keep the boxes above threshold."""
        if not chunk_patches:
            return
        arr = np.asarray(chunk_patches, dtype="float32")
        preds = model.predict(arr, batch_size=cfg.batch_size, verbose=0).flatten()
        for (x, y), conf in zip(chunk_coords, preds):
            if conf > cfg.threshold:
                cx, cy = x + ps // 2, y + ps // 2
                half = int((ps * cfg.visual_multiplier) // 2)
                x1, y1 = max(0, cx - half), max(0, cy - half)
                x2, y2 = min(w, cx + half), min(h, cy + half)
                boxes.append([x1, y1, x2, y2, float(conf)])

    # Integral image of the mask, so the "is this patch mostly water?" test is
    # O(1) per window instead of an 80x80 mean. Over a full 25km scene that
    # test runs ~200k times, and it dominated the sliding window's runtime.
    integral = cv2.integral(mask_scaled.astype("uint8"))
    patch_px = ps * ps

    def water_fraction(x, y):
        total = (integral[y + ps, x + ps] - integral[y, x + ps]
                 - integral[y + ps, x] + integral[y, x])
        return total / patch_px

    # Score in chunks rather than materialising every patch first. A 25km scene
    # at stride 30 yields ~90k water patches; as one float32 array that is 6+ GB.
    chunk_limit = max(cfg.batch_size, cfg.chunk_patches)
    patches, coords = [], []
    for y in range(0, h - ps + 1, stride):
        for x in range(0, w - ps + 1, stride):
            if water_fraction(x, y) < cfg.water_threshold:
                continue
            patches.append(roi_scaled[y:y + ps, x:x + ps].astype("float32") / 255.0)
            coords.append((x, y))
            if len(patches) >= chunk_limit:
                score(patches, coords)
                patches, coords = [], []
    score(patches, coords)

    final_boxes = nms(boxes, iou_threshold=cfg.nms_iou)

    output_bgr = None
    if draw:
        output_bgr = cv2.cvtColor(roi_scaled, cv2.COLOR_RGB2BGR)
        for (x1, y1, x2, y2, conf) in final_boxes:
            cv2.rectangle(output_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output_bgr, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return {
        "boxes": final_boxes,
        "roi_scaled": roi_scaled,
        "output_bgr": output_bgr,
        "water_mask": mask_scaled,
        "water_area_px": int(mask_scaled.sum()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    from matplotlib import pyplot as plt

    from .config import DATA, RESULTS, SHIP_DETECTOR

    p = argparse.ArgumentParser(description="Detect ships in a port image.")
    p.add_argument("--image", type=Path, default=DATA / "la_port_scene.png")
    p.add_argument("--model", type=Path, default=SHIP_DETECTOR)
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    p.add_argument("--crop", type=float, nargs=2, metavar=("START", "END"),
                   default=[0.25, 0.60],
                   help="crop as fractions of width; pass 0 1 for the full image")
    p.add_argument("--threshold", type=float, default=DetectorConfig.threshold)
    p.add_argument("--stride", type=int, default=DetectorConfig.stride)
    p.add_argument("--no-plot", action="store_true",
                   help="skip the side-by-side figure, write boxes only")
    args = p.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(
            f"Model not found at {args.model}.\n"
            "Download ship_detector_fixed.keras into data/ (link in README)."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = DetectorConfig(threshold=args.threshold, stride=args.stride)
    crop = tuple(args.crop) if args.crop != [0.0, 1.0] else None

    model = load_detector(args.model)
    result = detect_ships(args.image, model, crop_x_range=crop, config=config)

    boxes = result["boxes"]
    print(f"Found {len(boxes)} ships  |  water area {result['water_area_px']} px")

    boxes_path = args.out_dir / "detection_result_boxes_only.png"
    cv2.imwrite(str(boxes_path), result["output_bgr"])
    print(f"Saved: {boxes_path}")

    if not args.no_plot:
        mask = result["water_mask"]
        tint = result["output_bgr"].copy()
        tint[mask] = (tint[mask] * 0.85 + np.array([40, 20, 0]) * 0.15).astype("uint8")

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        axes[0].imshow(cv2.cvtColor(tint, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Water mask overlay")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(result["output_bgr"], cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Detections ({len(boxes)} ships)")
        axes[1].axis("off")
        plt.tight_layout()

        fig_path = args.out_dir / "detection_result.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
