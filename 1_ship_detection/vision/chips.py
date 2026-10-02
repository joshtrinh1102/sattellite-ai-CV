"""Build a labelled Sentinel-2 chip set, to adapt the detector to the domain.

WHY THIS EXISTS
---------------
`ship_detector_fixed.keras` was trained on Kaggle ShipsNet: Planet chips at
~3 m/px, bright, already framed on the vessel. Sentinel-2 true colour is
10 m/px and radiometrically raw. Measured on the 2021-10-10 San Pedro Bay
scene, with 12 hand-identified vessels and 60 random water patches:

    scale   ship mean conf    water mean conf
    2x      0.449             0.270
    3x      lower still       -
    4x      lower still       -

Overlapping distributions, no threshold separates them, and a sweep over the
whole scene recovered 4-6 of 12 known vessels. The model does not transfer.
That is a domain gap, not a bug, and the fix is the same one the notebook
already used once for land false positives: give it examples from the domain
it has to work in.

This module extracts candidate chips and writes contact sheets so they can be
labelled by eye, then packs the labelled result into arrays for fine-tuning.

    python -m vision.chips --extract      # candidates -> contact sheets
    python -m vision.chips --pack         # reviewed labels -> npz
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import propose_candidates
from .imagery import stretch
from .water_mask import build_water_mask_s2

CHIP = 80          # chip size fed to the classifier
SCALE = 2          # upscale factor; 80px at 2x covers 400m, which fits a ULCV


def elongation(stats_row, area):
    """Rough aspect proxy from a component's bounding box."""
    w = max(1, int(stats_row[cv2.CC_STAT_WIDTH]))
    h = max(1, int(stats_row[cv2.CC_STAT_HEIGHT]))
    return max(w, h) / min(w, h)


def extract_chips(scene_path, max_per_scene=140, seed=0):
    """Pull candidate chips from one scene.

    Sampling is stratified rather than "the brightest N": the hard cases for
    the classifier are wakes, whitecaps, breakwater rock and small craft, and
    a brightest-first sample contains none of them. Three strata:

      blob   - proposals that survive the prescreen (mostly ships, plus the
               false positives that matter)
      water  - random points on water with no proposal nearby (true negatives)
      edge   - points near the land/water boundary, where the mask leaks
    """
    rng = np.random.default_rng(seed)
    raw = cv2.imread(str(scene_path))
    if raw is None:
        return []
    mask = build_water_mask_s2(raw)
    st = stretch(raw)
    scaled = cv2.resize(st, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    H, W = scaled.shape[:2]

    # proposals on RAW pixels (see propose_candidates), chips from the
    # stretched image, which is what the classifier is shown
    cands = propose_candidates(raw, mask, contrast=12.0, min_area=6)
    rng.shuffle(cands)
    picks = [("blob", c[0], c[1], c[2]) for c in cands[: int(max_per_scene * 0.7)]]

    # true-negative water, away from any proposal
    if cands:
        pts = np.array([(c[0], c[1]) for c in cands])
    else:
        pts = np.zeros((0, 2))
    ys, xs = np.where(mask)
    if len(xs):
        for i in rng.choice(len(xs), min(len(xs), int(max_per_scene * 0.2)), replace=False):
            x, y = float(xs[i]), float(ys[i])
            if len(pts) and np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)) < 8:
                continue
            picks.append(("water", x, y, 0))

    # land/water edge, where the mask is least trustworthy
    edge = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ys, xs = np.where(edge > 0)
    if len(xs):
        for i in rng.choice(len(xs), min(len(xs), int(max_per_scene * 0.1)), replace=False):
            picks.append(("edge", float(xs[i]), float(ys[i]), 0))

    out = []
    for stratum, cx, cy, area in picks:
        x = max(0, min(W - CHIP, int(round(cx * SCALE)) - CHIP // 2))
        y = max(0, min(H - CHIP, int(round(cy * SCALE)) - CHIP // 2))
        out.append({
            "scene": Path(scene_path).name,
            "stratum": stratum,
            "cx": round(cx, 1), "cy": round(cy, 1), "area": int(area),
            "chip": scaled[y:y + CHIP, x:x + CHIP].copy(),
        })
    return out


def contact_sheet(chips, cols=12, pad=4, scale=2):
    """Lay chips out in a labelled grid for review by eye.

    Each cell is numbered so a reviewer can record verdicts as a list of
    indices, which is the cheapest labelling interface that still produces
    real labels.
    """
    n = len(chips)
    rows = (n + cols - 1) // cols
    cell = CHIP * scale + pad * 2
    label_h = 16
    sheet = np.full((rows * (cell + label_h), cols * cell, 3), 30, dtype="uint8")
    for i, c in enumerate(chips):
        r, col = divmod(i, cols)
        y0 = r * (cell + label_h) + label_h
        x0 = col * cell
        img = cv2.resize(c["chip"], (CHIP * scale, CHIP * scale),
                         interpolation=cv2.INTER_NEAREST)
        sheet[y0 + pad:y0 + pad + CHIP * scale, x0 + pad:x0 + pad + CHIP * scale] = img
        cv2.rectangle(sheet, (x0 + pad, y0 + pad),
                      (x0 + pad + CHIP * scale, y0 + pad + CHIP * scale), (90, 90, 90), 1)
        cv2.putText(sheet, str(i), (x0 + pad + 2, y0 + label_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return sheet


def main(argv=None):
    from .config import DATA

    p = argparse.ArgumentParser(description="Build Sentinel-2 chips for fine-tuning.")
    p.add_argument("--scenes", type=Path, default=DATA / "scenes")
    p.add_argument("--out", type=Path, default=DATA / "chips")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--pack", action="store_true")
    p.add_argument("--per-scene", type=int, default=140)
    p.add_argument("--sheet-size", type=int, default=120)
    p.add_argument("--max-scenes", type=int, default=8)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.extract:
        scenes = sorted(args.scenes.glob("USLAX_*.png"))[: args.max_scenes]
        allchips = []
        for i, s in enumerate(scenes):
            got = extract_chips(s, max_per_scene=args.per_scene, seed=i)
            allchips.extend(got)
            print(f"  {s.name}: {len(got)} chips", flush=True)

        np.save(args.out / "chips.npy",
                np.stack([c["chip"] for c in allchips]).astype("uint8"))
        with open(args.out / "chips.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=["idx", "scene", "stratum", "cx", "cy", "area"])
            wr.writeheader()
            for i, c in enumerate(allchips):
                wr.writerow({"idx": i, **{k: c[k] for k in ("scene", "stratum", "cx", "cy", "area")}})

        for s0 in range(0, len(allchips), args.sheet_size):
            sheet = contact_sheet(allchips[s0:s0 + args.sheet_size])
            cv2.imwrite(str(args.out / f"sheet_{s0:04d}.png"), sheet)
        print(f"\n{len(allchips)} chips, "
              f"{(len(allchips) + args.sheet_size - 1) // args.sheet_size} sheets -> {args.out}")

    if args.pack:
        chips = np.load(args.out / "chips.npy")
        labels = json.loads((args.out / "labels.json").read_text(encoding="utf-8"))
        idx = np.array(sorted(int(k) for k in labels))
        y = np.array([labels[str(i)] for i in idx], dtype="float32")
        np.savez_compressed(args.out / "labelled.npz", X=chips[idx], y=y, idx=idx)
        print(f"packed {len(idx)} labelled chips "
              f"({int(y.sum())} ship / {int((1 - y).sum())} not) -> labelled.npz")


if __name__ == "__main__":
    main()
