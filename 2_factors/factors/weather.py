"""Weather factors, from Open-Meteo's historical archive.

settings/requirements.md names weather as one of the candidate factors, and
3_ranking/plan-topics.md lists wind, visibility and precipitation as Stage B
features. Open-Meteo's archive API is free and needs no key, which keeps
Component 2's "no paid APIs" property.

Output is one row per day. Window features - "a week of gales backs up a
port, one windy morning does not" - are built from these days in 3_ranking,
which is the only place that knows the satellite capture dates.
"""

import json
import time
import urllib.parse
import urllib.request

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]


def fetch_daily(lat, lon, start, end, timeout=60, retries=3):
    """Daily weather for [start, end] at one point. Dates are `date` objects."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    url = f"{ARCHIVE}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:      # network flakiness, not logic
            last = err
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def daily(port_bbox, start, end, cache_path=None):
    """Daily weather rows for [start, end] at the bbox centre.

    One request covers the whole span. The cached response is reused when it
    already covers the span, so rebuilds work offline.
    """
    lon = (port_bbox[0] + port_bbox[2]) / 2
    lat = (port_bbox[1] + port_bbox[3]) / 2

    payload = None
    if cache_path and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        times = payload["daily"]["time"]
        if times[0] > start.isoformat() or times[-1] < end.isoformat():
            payload = None
    if payload is None:
        payload = fetch_daily(lat, lon, start, end)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

    d = payload["daily"]
    lo, hi = start.isoformat(), end.isoformat()
    return [{"date": t, **{v: d[v][i] for v in DAILY_VARS}}
            for i, t in enumerate(d["time"]) if lo <= t <= hi]
