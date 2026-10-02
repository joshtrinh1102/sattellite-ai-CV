"""Run the detector across every collected scene -> results/occupancy.csv.

This is the Component 1 output contract that Component 3 Stage B joins against:
one row per (port_id, date) carrying the ship count, the water area it was
measured over, the resulting density, and the occupancy label.

    python -m scsai.pipeline --detector ../data/ship_detector_s2.keras
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, detect_ships, load_detector
from .imagery import METRES_PER_PIXEL, stretch
from .occupancy import LABELS, fit_scale, ship_density, water_area_km2
from .water_mask import build_water_mask_s2

# Sentinel-2 settings. 80px at 2x covers 400 m of ground, which fits the
# largest container ships calling at San Pedro Bay with margin. The proposal
# thresholds are DetectorConfig's tuned defaults (raw DN 12 over local
# background, 6 px minimum) rather than being restated here, so there is one
# place to change them. 0.5 is the F1-optimal threshold from the held-out
# fine-tuning sweep in results/ship_detector_s2.report.json.
S2_CONFIG = DetectorConfig(
    scale_factor=2,
    threshold=0.5,
    proposals=True,
)


def read_manifest(scenes_dir):
    with open(Path(scenes_dir) / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(scenes_dir, model_path, out_csv, config=None, limit=None, save_overlays=0,
        overlay_dir=None):
    """Detect over every scene in the manifest and write the occupancy table."""
    cfg = config or S2_CONFIG
    rows = read_manifest(scenes_dir)
    if limit:
        rows = rows[:limit]
    model = load_detector(model_path)

    # Overlays are written in a second pass (see write_overlays), for the dates
    # worth looking at. Writing the first N in date order - which is what this
    # used to do - produced eight consecutive 2018 scenes: the least
    # informative possible sample, none showing the congestion the study is
    # about.
    out = []
    for i, r in enumerate(rows):
        path = Path(scenes_dir) / r["path"]
        t0 = time.time()
        res = detect_ships(path, model, config=cfg, draw=False,
                           water_mask_fn=build_water_mask_s2, preprocess=stretch)
        n = len(res["boxes"])

        # water_area_px is measured at scaled resolution, so scale_factor has to
        # be divided back out - that is what `scale_factor` is for here.
        area_km2 = water_area_km2(res["water_area_px"],
                                  metres_per_pixel=float(r["metres_per_pixel"]),
                                  scale_factor=cfg.scale_factor)
        out.append({
            "port_id": r["port_id"],
            "date": r["date"],
            "n_ships": n,
            "n_proposals": res.get("n_proposals", -1),
            "water_area_km2": round(area_km2, 3),
            "density": round(ship_density(n, area_km2), 4),
            "mean_conf": round(float(np.mean([b[4] for b in res["boxes"]])), 3) if n else 0.0,
            "aoi_cloud_frac": float(r.get("aoi_cloud_frac", 0) or 0),
            "scene": r["path"],
        })
        print(f"  [{i + 1}/{len(rows)}] {r['date']}  {n:4d} ships  "
              f"{area_km2:7.2f} km2  density {out[-1]['density']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # low / medium / high, anchored on pre-COVID "normal times"
    try:
        scale, out = label_occupancy(out)
        print(f"\nOccupancy scale fitted on pre-2020-03 dates: "
              f"low < {scale.low_cut:.3f} <= medium < {scale.high_cut:.3f} <= high "
              f"({scale.unit})")
        counts = {}
        for r in out:
            counts[r["occupancy"]] = counts.get(r["occupancy"], 0) + 1
        print("  " + "   ".join(f"{k}: {counts.get(k, 0)}" for k in LABELS))
    except ValueError as err:
        print(f"\nOccupancy labels skipped: {err}")

    if save_overlays and overlay_dir:
        write_overlays(out, scenes_dir, model, cfg, overlay_dir, n=save_overlays)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"\n{len(out)} dates -> {out_csv}")
    return out


def write_overlays(rows, scenes_dir, model, cfg, overlay_dir, n=6):
    """Re-detect the most informative dates and write annotated images.

    Picks the busiest and the quietest rather than the first few, so the folder
    shows the range the study found - the 2021 backlog next to a quiet date is
    the evidence that the counts mean something.

    JPEG, not PNG: these are visual evidence, and at full scene resolution the
    PNGs ran 9 MB each, which made the results directory 80 MB. Quality 92
    keeps the boxes and the hulls crisp at a tenth of that.
    """
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(rows, key=lambda r: -r["n_ships"])
    half = max(1, n // 2)
    picks, seen = [], set()
    for r in ranked[:half] + ranked[-(n - half):]:
        if r["date"] not in seen:
            seen.add(r["date"])
            picks.append(r)

    written = []
    for r in picks:
        res = detect_ships(Path(scenes_dir) / r["scene"], model, config=cfg,
                           draw=True, water_mask_fn=build_water_mask_s2,
                           preprocess=stretch)
        img = cv2.resize(res["output_bgr"], None,
                         fx=1.0 / cfg.scale_factor, fy=1.0 / cfg.scale_factor)
        name = f"{r['date']}_{r['n_ships']:03d}ships_{r.get('occupancy', 'na')}.jpg"
        cv2.imwrite(str(overlay_dir / name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append(name)
    print(f"\nWrote {len(written)} annotated scenes to {overlay_dir}")
    for name in written:
        print(f"  {name}")
    return written


def label_occupancy(rows, baseline_end="2020-03-01", quantiles=(0.33, 0.67)):
    """Attach the low/medium/high label settings/requirements.md asks for.

    The cut points are fitted on a BASELINE PERIOD and then applied to every
    row, which is what `scsai.occupancy.fit_scale` is documented to require.
    The baseline here is everything before `baseline_end` - "normal times".

    Fitting on all dates instead would be the subtle mistake: quantiles over
    the full series force a third of dates into `high` by construction, so the
    2021 backlog would come out looking exactly as busy as a quiet 2018, and
    the COVID comparison the requirements ask for would be erased by its own
    normalisation.
    """
    baseline = [r for r in rows if r["date"] < baseline_end]
    if len(baseline) < 10:
        # Not enough "normal times" to anchor the scale; say so rather than
        # silently falling back to whole-series quantiles.
        raise ValueError(
            f"only {len(baseline)} dates before {baseline_end}; need >=10 to fit "
            f"the occupancy scale on a baseline period"
        )
    scale = fit_scale([r["density"] for r in baseline], quantiles=quantiles)
    for r in rows:
        r["occupancy"] = scale.label(r["density"])
    return scale, rows


def main(argv=None):
    from .config import DATA, DETECTIONS, OCCUPANCY, SHIP_DETECTOR_S2

    p = argparse.ArgumentParser(description="Detect over all scenes.")
    p.add_argument("--scenes", type=Path, default=DATA / "scenes")
    p.add_argument("--detector", type=Path, default=SHIP_DETECTOR_S2)
    p.add_argument("--out", type=Path, default=OCCUPANCY)
    p.add_argument("--threshold", type=float, default=S2_CONFIG.threshold)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overlays", type=int, default=6,
                   help="write annotated PNGs for the first N scenes")
    args = p.parse_args(argv)

    cfg = DetectorConfig(**{**vars(S2_CONFIG), "threshold": args.threshold})
    run(args.scenes, args.detector, args.out, config=cfg, limit=args.limit,
        save_overlays=args.overlays, overlay_dir=DETECTIONS)


if __name__ == "__main__":
    main()
