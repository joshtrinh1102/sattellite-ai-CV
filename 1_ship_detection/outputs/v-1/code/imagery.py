"""Component 1 input - acquire dated port scenes from Sentinel-2.

The binding constraint recorded in the README was that the vision side had a
single, manually downloaded scene. This module removes it: Sentinel-2 L2A is
open data on AWS, exposed through the Earth Search STAC API, so scenes for an
arbitrary list of dates can be pulled reproducibly.

Resolution note - the detector was trained on ShipsNet (~3 m/px, 80 px patches,
so a ship fills the patch). Sentinel-2 true colour is 10 m/px, so scenes are
upscaled ~3x before the sliding window runs, which puts a 250 m vessel at
roughly 75 px. That is the same apparent size the classifier was trained on.

    python -m scsai.imagery --port USLAX --start 2019-01-01 --end 2026-09-01

Writes one PNG per capture date to data/scenes/ plus a manifest CSV.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# GDAL needs these before rasterio touches a remote COG, otherwise it tries to
# list the whole bucket prefix on every open.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import numpy as np
import requests

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

# Ground sample distance of the Sentinel-2 true-colour (TCI) asset.
METRES_PER_PIXEL = 10.0


@dataclass(frozen=True)
class Port:
    """An area of interest. The bbox deliberately includes the outer
    anchorage, not just the berths - settings/requirements.md asks for ships on the
    water, and during congestion most of them are anchored offshore."""
    port_id: str
    name: str
    bbox: tuple      # (min_lon, min_lat, max_lon, max_lat)


PORTS = {
    # San Pedro Bay: Port of LA + Port of Long Beach berths, the main channel,
    # and the offshore anchorage where the 2021-22 backlog queued.
    "USLAX": Port(
        port_id="USLAX",
        name="Port of Los Angeles / Long Beach",
        # Widened after inspecting the 2021-10-10 scene: at the peak of the
        # backlog the anchorage ran past the east and south edges of a tighter
        # box, which would have clipped exactly the dates the study cares about.
        bbox=(-118.32, 33.58, -118.06, 33.78),
    ),
}


def covers(item_bbox, aoi):
    """Does this scene's footprint fully contain the AOI?"""
    return (item_bbox[0] <= aoi[0] and item_bbox[1] <= aoi[1]
            and item_bbox[2] >= aoi[2] and item_bbox[3] >= aoi[3])


def search_scenes(port, start, end, max_cloud=12.0, limit=1500, whole_aoi_only=True):
    """STAC search for low-cloud Sentinel-2 L2A scenes covering the AOI.

    `whole_aoi_only` matters more than it looks. A STAC bbox search returns
    every scene that *intersects* the AOI, and San Pedro Bay sits right on the
    MGRS 11SLT/11SMT tile boundary at -118.088. Scenes from 11SMT clip a
    thin eastern sliver of the AOI and read back 92% no-data - which is exactly
    what happened on the first full run: 19 of 26 candidate dates were
    discarded downstream, silently halving the sample. Filtering to footprints
    that fully contain the AOI removes the problem at the source.
    """
    body = {
        "collections": [COLLECTION],
        "bbox": list(port.bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 100,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    items, url, payload = [], STAC, body
    while url and len(items) < limit:
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        page = resp.json()
        items.extend(page.get("features", []))
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url, payload = nxt["href"], nxt.get("body", payload)
    if whole_aoi_only:
        items = [it for it in items if covers(it["bbox"], port.bbox)]
    return items[:limit]


# Fixed radiometric stretch. These constants are deliberately NOT per-scene
# percentiles: a per-scene stretch would rescale a busy date differently from a
# quiet one and destroy exactly the cross-date comparability the study needs.
# Measured on the AOI, open water sits near DN 13 and vessels saturate well
# above DN 90, so this maps water to ~DN 50 and hulls to ~DN 255.
STRETCH_LO, STRETCH_HI, STRETCH_GAMMA = 4.0, 90.0, 2.0


def stretch(image, lo=STRETCH_LO, hi=STRETCH_HI, gamma=STRETCH_GAMMA):
    """Raw Sentinel-2 TCI -> the brightness range the detector was trained on.

    ShipsNet patches are bright Planet chips; raw TCI is not. Without this the
    classifier sees a near-black image and fires on nothing.
    """
    x = (image.astype("float32") - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0) ** (1.0 / gamma)
    return (x * 255.0).astype("uint8")


def _visual_href(item):
    assets = item.get("assets", {})
    for key in ("visual", "visual-jp2", "tci"):
        if key in assets and assets[key].get("href"):
            return assets[key]["href"]
    return None


def fetch_window(href, bbox):
    """Read just the AOI out of a remote COG. Returns an RGB uint8 array."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(left, bottom, right, top, src.transform)
        data = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
    return np.transpose(data, (1, 2, 0))


def scene_quality(rgb):
    """Cheap per-AOI cloud/haze and no-data screen.

    Tile-level `eo:cloud_cover` describes a 110 km tile; a tile can pass at 8%
    while the 20 km of water we care about sits under the only cloud. Judge the
    window itself: bright near-neutral pixels are cloud, all-zero is off-swath.
    """
    grey = rgb.mean(axis=2)
    nodata = float((grey == 0).mean())
    spread = rgb.max(axis=2).astype("int16") - rgb.min(axis=2).astype("int16")
    cloud = float(((grey > 140) & (spread < 30)).mean())
    return {"nodata_frac": nodata, "cloud_frac": cloud, "mean_brightness": float(grey.mean())}


def water_quality(image_bgr):
    """Haze screen, measured on the water itself.

    `scene_quality` catches thick bright cloud, and it is not enough. San Pedro
    Bay spends much of May-September under a marine layer that reads as a thin
    milky veil: brightness around DN 120 with normal colour spread, so the
    cloud test passes it, while every vessel in the AOI is washed out. On the
    first batch of downloads, 2018-06-08, 2019-05-24, 2019-06-28 and 2019-09-21
    all passed the cloud test and all are unusable.

    Two measurements separate them cleanly, because the geography is fixed:

      water_frac  - the AOI is ~78% water on every clear date. Haze lifts the
                    water's brightness past the mask's DN<45 ceiling, so the
                    mask shrinks. Clear: 0.77-0.78. Hazy: 0.25-0.75.
      water_median - open water sits at DN 7-12 when clear, and DN 24-51 when
                    veiled. There is no overlap.
    """
    import cv2

    from .water_mask import build_water_mask_s2

    mask = build_water_mask_s2(image_bgr)
    grey = image_bgr.mean(axis=2).astype("float32")
    water = grey[mask] if mask.any() else grey

    # Local standard deviation over water: surface roughness. A calm sea is
    # almost featureless at 10 m/px, so anything the prescreen finds is an
    # object on it. A broken or veiled surface is not, and the prescreen
    # degenerates - see MAX_WATER_SD90.
    k = np.ones((9, 9), dtype="float32") / 81.0
    mu = cv2.filter2D(grey, -1, k)
    sq = cv2.filter2D(grey * grey, -1, k)
    sd = np.sqrt(np.maximum(0, sq - mu * mu))
    sd_water = sd[mask] if mask.any() else sd

    return {
        "water_frac": float(mask.mean()),
        "water_median": float(np.median(water)),
        "water_p99": float(np.percentile(water, 99)),
        "water_sd90": float(np.percentile(sd_water, 90)),
    }


MIN_WATER_FRAC = 0.70      # clear dates measure 0.77-0.78; hazy ones collapse
MAX_WATER_MEDIAN = 20.0    # clear water is DN 7-12, veiled water DN 24-51

# Surface roughness ceiling. This screen was added after the first full run
# produced 170 and 149 ships on 2025-03-18 and 2024-04-27, against a median of
# 58 and a genuine congestion-peak count of 118 on 2021-10-10. Inspection showed
# no fleet: both scenes are veiled, with a broken water surface that generated
# 1,437 and 2,309 proposals where a normal date generates ~200. The prescreen
# assumes vessels are the only bright compact objects on water, and on a broken
# surface that assumption fails, so the counts are not comparable with the rest.
# Measured over 87 dates the two populations do not overlap: normal scenes run
# sd90 0.80-1.22, degraded ones 2.25-7.30.
MAX_WATER_SD90 = 2.0


def usable(q):
    """Is this scene clear and calm enough over the water to count ships on?"""
    return (q["water_frac"] >= MIN_WATER_FRAC
            and q["water_median"] <= MAX_WATER_MEDIAN
            and q.get("water_sd90", 0.0) <= MAX_WATER_SD90)


def pick_dates(items, per_quarter=2):
    """Thin the STAC hits to a spread of capture dates.

    Sentinel-2 revisits every ~5 days, so a 7-year search returns hundreds of
    scenes clustered wherever the weather was clear. Ranking by cloud cover
    alone would bias the sample toward summer. Take the clearest few per
    calendar quarter instead, which keeps seasonality in the design.
    """
    by_quarter = {}
    for item in items:
        props = item["properties"]
        dt = datetime.fromisoformat(props["datetime"].replace("Z", "+00:00"))
        key = (dt.year, (dt.month - 1) // 3 + 1)
        by_quarter.setdefault(key, []).append((props.get("eo:cloud_cover", 100), dt, item))

    chosen, seen_dates = [], set()
    for key in sorted(by_quarter):
        for _, dt, item in sorted(by_quarter[key], key=lambda r: r[0])[: per_quarter * 3]:
            if dt.date() in seen_dates:
                continue
            seen_dates.add(dt.date())
            chosen.append((dt, item))
            if sum(1 for d, _ in chosen if (d.year, (d.month - 1) // 3 + 1) == key) >= per_quarter:
                break
    return sorted(chosen, key=lambda r: r[0])


def download(port, start, end, out_dir, per_quarter=2, max_cloud=12.0,
             max_scene_cloud=0.06, max_nodata=0.02, limit=None):
    """Fetch scenes and write one PNG per accepted capture date."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    items = search_scenes(port, start, end, max_cloud=max_cloud)
    print(f"STAC: {len(items)} scenes over {port.name} in {start}..{end} "
          f"(tile cloud < {max_cloud}%)")

    candidates = pick_dates(items, per_quarter=per_quarter)
    print(f"Thinned to {len(candidates)} candidate dates "
          f"({per_quarter} per quarter)\n")

    rows = []
    for dt, item in candidates:
        if limit and len(rows) >= limit:
            break
        date = dt.date().isoformat()
        existing = out_dir / f"{port.port_id}_{date}.png"
        if existing.exists():
            print(f"  {date}  already downloaded")
            continue
        href = _visual_href(item)
        if not href:
            print(f"  {date}  skip - no visual asset")
            continue
        try:
            rgb = fetch_window(href, port.bbox)
        except Exception as err:
            print(f"  {date}  skip - read failed ({type(err).__name__})")
            continue

        q = scene_quality(rgb)
        if q["nodata_frac"] > max_nodata:
            print(f"  {date}  skip - {q['nodata_frac']:.0%} no-data (off swath)")
            continue
        if q["cloud_frac"] > max_scene_cloud:
            print(f"  {date}  skip - {q['cloud_frac']:.0%} cloud over the AOI")
            continue

        wq = water_quality(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not usable(wq):
            print(f"  {date}  skip - haze (water_frac {wq['water_frac']:.2f}, "
                  f"water_median {wq['water_median']:.1f})")
            continue

        path = out_dir / f"{port.port_id}_{date}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rows.append({
            "port_id": port.port_id,
            "date": date,
            "path": path.name,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "metres_per_pixel": METRES_PER_PIXEL,
            "tile_cloud_pct": round(item["properties"].get("eo:cloud_cover", -1), 2),
            "aoi_cloud_frac": round(q["cloud_frac"], 4),
            "water_frac": round(wq["water_frac"], 4),
            "water_median": round(wq["water_median"], 2),
            "stac_id": item["id"],
        })
        print(f"  {date}  saved {rgb.shape[1]}x{rgb.shape[0]}  "
              f"aoi_cloud={q['cloud_frac']:.1%}  tile_cloud={item['properties'].get('eo:cloud_cover', -1):.1f}%")

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["date"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} scenes -> {out_dir}\nmanifest: {manifest}")
    return rows


def requalify(port, out_dir, drop=False):
    """Re-screen every scene on disk and rebuild the manifest.

    Kept separate from `download` so the haze gate can be tightened later
    without re-fetching 60 scenes over the network. With `drop`, scenes that
    fail are deleted; otherwise they stay on disk and are simply left out of
    the manifest, which is what the rest of the pipeline reads.
    """
    import cv2

    out_dir = Path(out_dir)
    rows, rejected = [], []
    for path in sorted(out_dir.glob(f"{port.port_id}_*.png")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        date = path.stem.split("_")[-1]
        wq = water_quality(img)
        if not usable(wq):
            rejected.append((date, wq))
            if drop:
                path.unlink()
            continue
        rows.append({
            "port_id": port.port_id,
            "date": date,
            "path": path.name,
            "width": img.shape[1],
            "height": img.shape[0],
            "metres_per_pixel": METRES_PER_PIXEL,
            "water_frac": round(wq["water_frac"], 4),
            "water_median": round(wq["water_median"], 2),
            "water_sd90": round(wq["water_sd90"], 3),
        })

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["date"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"usable: {len(rows)}   rejected for haze: {len(rejected)}")
    for date, wq in rejected:
        print(f"  reject {date}  water_frac {wq['water_frac']:.2f}  "
              f"water_median {wq['water_median']:.1f}  "
              f"water_sd90 {wq['water_sd90']:.2f}")
    print(f"manifest: {manifest}")
    return rows


def main(argv=None):
    from .config import DATA

    p = argparse.ArgumentParser(description="Fetch dated Sentinel-2 port scenes.")
    p.add_argument("--port", default="USLAX", choices=sorted(PORTS))
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-09-01")
    p.add_argument("--per-quarter", type=int, default=2)
    p.add_argument("--max-cloud", type=float, default=12.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--requalify", action="store_true",
                   help="re-screen scenes already on disk and rebuild the manifest")
    p.add_argument("--drop", action="store_true",
                   help="with --requalify, delete scenes that fail the haze gate")
    p.add_argument("--out-dir", type=Path, default=DATA / "scenes")
    args = p.parse_args(argv)

    if args.requalify:
        requalify(PORTS[args.port], args.out_dir, drop=args.drop)
        return

    download(PORTS[args.port], args.start, args.end, args.out_dir,
             per_quarter=args.per_quarter, max_cloud=args.max_cloud,
             limit=args.limit)


if __name__ == "__main__":
    main()
