# 1 — Ship detection

**In:** Sentinel-2 scenes of San Pedro Bay. **Out:** `handoff/ship_counts.csv`
(one row per capture date) and `handoff/detections/` (annotated scenes).

Run every command from this folder.

```bash
# scenes - one PNG per capture date, into data/scenes/
python -m vision.imagery --port USLAX --start 2018-06-01 --end 2026-09-01
python -m vision.imagery --requalify          # re-screen for haze, rebuild manifest

# detector - adapt the aerial classifier to Sentinel-2 (see "Domain gap")
python -m vision.chips --extract              # candidates -> data/chips/sheet_*.png
#   label the sheets by eye into data/chips/labels.json, then:
python -m vision.chips --pack
python -m vision.finetune --train             # -> models/ship_detector_s2.keras

# counts - the handoff
python -m vision.pipeline                     # -> handoff/ship_counts.csv, handoff/detections/
```

| Path | What |
|---|---|
| `vision/imagery.py` | Sentinel-2 acquisition, fixed stretch, haze screen |
| `vision/water_mask.py` | water/land segmentation |
| `vision/detector.py` | proposals + CNN + NMS; single-image CLI |
| `vision/chips.py`, `finetune.py` | labelled chip set and Sentinel-2 fine-tune |
| `vision/pipeline.py`, `occupancy.py` | all scenes → counts, density, low/medium/high |
| `models/` | `.keras` weights + held-out scorecard `ship_detector_s2.report.json` |
| `outputs/aerial/` | original single-image aerial results |
| `outputs/v-1/` | frozen snapshot of the code, models and images that produced the detections |
| `ship_detection.ipynb` | original ShipsNet training notebook |

The chip labels (`data/chips/labels.json`) are versioned because they were made
by eye. Scenes, chip pixels and weights are not, so rebuild them with the
commands above. The aerial base model (`ship_detector_fixed.keras`, 51 MB) goes
in `models/`:
https://drive.google.com/file/d/1UEX9-FNIaJNLMMjJHE4Qc5Phz9FhO0ED/view?usp=drive_link

Single image (aerial): `python -m vision.detector --image data/la_port_scene.png`

## How it works

**Acquisition.** Sentinel-2 L2A from the Earth Search STAC API (no key, back
to 2015). Only the AOI window is read from each remote COG. The AOI covers both
ports' berths plus the offshore anchorage, where most ships wait during congestion.

**Three screens before anything is counted.** Each was added because a run
without it produced wrong numbers.

1. **Coverage:** keep only footprints that contain the whole AOI. The bay
   straddles two MGRS tiles, and partial tiles silently discarded 19 of 26 dates.
2. **Haze:** the summer marine layer passes cloud tests but washes out vessels.
   Clear dates measure `water_frac` 0.77–0.78 and `water_median` 7–12. Veiled
   dates measure 0.25–0.75 and 24–51.
3. **Surface roughness:** a broken water surface produced phantom counts (170
   ships vs a median of 58). `water_sd90` separates normal water (0.80–1.22)
   from degraded water (2.25–7.30).

About a third of dates are discarded, mostly in summer.

**Detection.** Water mask → bright-blob proposals → 80 px chips at 2× → CNN →
NMS. Proposals cut a ~6 minute sliding window to ~2 s per date.

## Domain gap

The aerial model (ShipsNet, ~3 m/px) does not transfer to Sentinel-2 (10 m/px).
It scored ships 0.449 and water 0.270, which leaves no usable threshold.
`finetune.py` retrains the head on 409 hand-labelled chips (98 ship / 311
not), **split by scene**. Held-out result, from two withheld scenes:

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.4 | 0.824 | 0.875 | 0.848 |
| **0.5** | **0.875** | **0.875** | **0.875** |
| 0.6 | 0.871 | 0.844 | 0.857 |

AUC 0.934. These are chip-level numbers conditional on the proposal stage.
