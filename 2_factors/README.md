# 2 — Factors affecting ships

**In:** GDELT news headlines and Open-Meteo weather. **Out:**
`handoff/factors_daily.csv`, one row per day (2018-09-07 → 2026-04-07).

Run from this folder.

```bash
python -m factors.build                 # from cache/ only: offline, seconds
python -m factors.build --fetch-news    # also fill missing news days from GDELT
python -m factors.news --probe          # is GDELT coverage dense enough?
```

| Path | What |
|---|---|
| `factors/weather.py` | daily wind, gusts, rain, temperature at the port |
| `factors/news.py` | GDELT ingest, dedupe, per-day binning, coverage tracking |
| `factors/topics.py` | headline → 1 of 12 risk categories (+ irrelevant), title tone |
| `factors/build.py` | joins them into the handoff file |
| `cache/` | raw API responses, versioned so rebuilds work offline |
| `outputs/articles.csv` | every headline with its category, for review |
| `plan-news.md` | design notes and query strategy |

**The 12 risk categories:** inflation/economy, material shortages, natural
disasters, global regulations, cybersecurity, logistics reliability, labour
shortages, demand volatility, operational risks, reputation risks,
health/pandemic, conflict/war.

## News coverage is the open gap

GDELT rate-limits hard: sustained use returns 429 for tens of minutes. Days
not yet fetched carry `news_covered = 0` and blank news columns, which means
**missing, not zero news**. Today 159 of 2,770 days are covered, which is 21
of the 76 capture windows, so 3_ranking leaves news out.
`--fetch-news` resumes where it stopped. Each 7-day chunk is cached, so budget
a few hours across runs.

Columns are daily **counts**, not shares, so they sum exactly over any window.
3_ranking turns them into 7-day shares.
