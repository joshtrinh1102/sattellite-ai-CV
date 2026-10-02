"""Paths and settings for Component 2. Everything resolves from this folder.

The only path that leaves the folder is HANDOFF, where the daily factor series
is written for 3_ranking to read.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 2_factors/
HANDOFF = ROOT.parent / "handoff"

CACHE = ROOT / "cache"         # raw API responses, so rebuilds work offline
NEWS_RAW = CACHE / "news_raw"  # untouched GDELT responses
WEATHER_CACHE = CACHE / "weather_cache.json"
OUTPUTS = ROOT / "outputs"
ARTICLES = OUTPUTS / "articles.csv"   # every headline with its category

# Handoff contract - see handoff/README.md
FACTORS_DAILY = HANDOFF / "factors_daily.csv"

# Port areas of interest, as (min_lon, min_lat, max_lon, max_lat). Kept here
# rather than imported from 1_ship_detection so the folders stay independent;
# weather is fetched at the bbox centre.
PORT_BBOX = {
    "USLAX": (-118.32, 33.58, -118.06, 33.78),
}
