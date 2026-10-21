"""Component 2 entry point: news + weather -> handoff/factors_daily.csv.

    python -m factors.build                  # from caches only (offline, seconds)
    python -m factors.build --fetch-news     # also fill news gaps from GDELT (hours)

One row per day. Columns:

    date
    temperature_2m_mean, precipitation_sum,      weather (Open-Meteo)
    wind_speed_10m_max, wind_gusts_10m_max
    news_covered                                 1 if GDELT was fetched for the day
    n_articles, n_relevant, tone_sum_relevant    news totals
    count_<factor>                               relevant headlines labelled with it
    dim_<dimension>                              relevant headlines in that dimension

Factors and dimensions come from factors.toml (multi-label, so count_ columns
can sum past n_relevant). Also writes handoff/factor_definitions.json - the
titles, dimensions and weights from that file - so 3_ranking never has to read
this folder.

A day with news_covered = 0 has MISSING news, not zero news.
"""

import argparse
import csv
import json
from datetime import date, timedelta

from . import news, topics, weather
from .config import ARTICLES, FACTOR_DEFS, FACTORS_DAILY, PORT_BBOX, WEATHER_CACHE

NEWS_COLS = (["n_articles", "n_relevant", "tone_sum_relevant"]
             + [f"count_{t}" for t in topics.RISK_LABELS]
             + [f"dim_{d}" for d in topics.DIMENSIONS])


def build(port, start, end, fetch_news=False):
    print(f"Weather: Open-Meteo, {start} -> {end}")
    wx = {r["date"]: r for r in weather.daily(PORT_BBOX[port], start, end,
                                               cache_path=WEATHER_CACHE)}

    print("News: GDELT" + ("" if fetch_news else " (cache only)"))
    articles, covered = news.collect(port, start, end, fetch=fetch_news)
    art_df, news_df = topics.daily_scores(articles)
    if not art_df.empty:
        ARTICLES.parent.mkdir(parents=True, exist_ok=True)
        art_df.to_csv(ARTICLES, index=False)
    by_day = {r["date"]: r for r in news_df.to_dict("records")}

    rows, d = [], start
    while d <= end:
        key = d.isoformat()
        rec = {"date": key, **{v: wx.get(key, {}).get(v) for v in weather.DAILY_VARS}}
        rec["news_covered"] = int(d in covered)
        for c in NEWS_COLS:
            # Covered days with no headlines are genuine zeros; uncovered stay blank.
            rec[c] = by_day.get(key, {}).get(c, 0) if d in covered else None
        rows.append(rec)
        d += timedelta(days=1)

    FACTORS_DAILY.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTORS_DAILY, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    FACTOR_DEFS.write_text(json.dumps(topics.TAXONOMY.definitions(), indent=2),
                           encoding="utf-8")
    print(f"\nWrote {len(rows)} days -> {FACTORS_DAILY}")
    if not art_df.empty:
        rel = art_df[art_df["label"] != "irrelevant"]
        multi = (rel["labels"].str.count(";") > 0).sum()
        print(f"Headlines: {len(art_df)} total, {len(rel)} relevant, "
              f"{multi} with more than one factor")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", default="USLAX", choices=sorted(PORT_BBOX))
    p.add_argument("--start", type=date.fromisoformat, default=date(2018, 9, 7))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 4, 7))
    p.add_argument("--fetch-news", action="store_true",
                   help="fetch uncovered days from GDELT (rate-limited; budget hours)")
    args = p.parse_args(argv)
    build(args.port, args.start, args.end, fetch_news=args.fetch_news)


if __name__ == "__main__":
    main()
