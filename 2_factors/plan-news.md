# Component 2 — News & factor ingestion

**Status:** implemented in `2_factors/factors/news.py`. Ports and date range settled for
USLAX; see "Implementation notes" at the end for what the live API forced to
change.

## Decisions taken (2026-09-22)

| # | Decision | Consequence |
|---|----------|-------------|
| Q1 | **GDELT 2.0 DOC API only.** Free, no key. | Covers COVID-era backfill at zero cost. |
| Q2 | **Headlines only, no article bodies.** | Component 3 classifies from titles. Accepted deliberately. |
| Q3 | **Manual imagery collection, a few dates.** | See "What this costs us" — the binding constraint on the whole study. |

## Goal

Produce a per-date, per-port record of the factors that plausibly disrupt port
traffic, so Component 3 can score them against observed ship activity.

Requirement, verbatim:

> see if disruptions, labor actions, vessel bunching, policy bunching, … using
> news api related to ports and activities; events that happens to the traffic

> factors like covid, weather, war, equipment, policy changes

## What this costs us: the sample-size problem

Manual imagery collection is the pragmatic choice, but it needs stating
plainly, because it determines which claims the project can honestly make.

With **~10 hand-collected dates**, the achievable result is a *before/after
comparison*: ship density during COVID vs during normal operations, with a
confidence interval. That is a real finding and it directly answers *"do ships
during covid have less than ships during normal times"*.

What ~10 dates **cannot** support is the second requirement — *"có 10 factors
khác nhau, nhưng chỉ có 5 factor thực sự ảnh hưởng"*. Ranking 12 factors means
estimating at least 12 relationships. With 10 observations there are more
parameters than data points; any ranking produced is noise with a number
attached.

**Cheapest fix, and it is genuinely cheap.** Google Earth Pro has a historical
imagery timeline for major ports — step through available capture dates over
the same fixed viewport and export each one. Getting from 10 to **50–60 dates**
is an afternoon of clicking, not a new pipeline, and it moves the analysis from
"before/after only" to "a correlation study with enough power to rank factors".

| Dates | What becomes possible |
|-------|----------------------|
| ~10 | COVID vs normal comparison. Nothing about individual factors. |
| ~30 | Add 2–3 pre-specified factor hypotheses, chosen in advance. |
| ~60 | Rank the top factors with regularised regression; "which 5 matter" becomes answerable. |

Sampling matters as much as count: spread dates across seasons *and* across
both COVID and normal periods. Clustering all the "normal" captures in one
season confounds the COVID effect with seasonality, and there is no way to
separate them after the fact.

## Output contract

`data/articles.parquet`, one row per deduplicated article:

| column | type | notes |
|--------|------|-------|
| `article_id` | str | stable hash of normalised URL |
| `published_at` | timestamp (UTC) | source timestamp, normalised |
| `date` | date | UTC calendar date, the join key |
| `port_id` | str | which port query matched; `GLOBAL` if none |
| `title` | str | **the model input** — GDELT gives no body |
| `url`, `domain` | str | |
| `language` | str | ISO 639-1 |
| `tone` | float | computed by us, not from GDELT — see below |
| `query_tag` | str | which query matched, for auditing recall |

Plus `data/news_raw/<date>.jsonl` holding untouched responses, so the pipeline
can be rebuilt without re-hitting the API.

## Sentiment: NOT free after all (corrected 2026-09-22)

An earlier draft of this plan said GDELT returns a per-article V2Tone score and
that sentiment therefore needed no work. **A live probe disproved that.** The
DOC API in `mode=ArtList` returns exactly these fields:

```
domain, language, seendate, socialimage, sourcecountry, title, url, url_mobile
```

No tone. Per-article tone lives in the GDELT GKG files, not this endpoint.

Options for the sentiment feature, which the impact model still wants:

| Option | Cost | Notes |
|--------|------|-------|
| **Score titles ourselves (VADER or a small transformer)** | Free, minutes | Title-only sentiment is weaker than document-level, but it is the same text the classifier sees, so it is at least consistent. **Recommended.** |
| GDELT GKG raw files | Free, heavy | Per-article tone over full text, but means downloading 15-minute CSV dumps and filtering — a real pipeline, not an API call. |
| `mode=ToneChart` | Free | Returns an aggregate tone histogram per query, not per article. Possibly enough, since the impact model consumes daily aggregates anyway. Being tested. |

This does not change the source decision — GDELT is still the only free option
covering the COVID window — but it does add work that was previously assumed
away.

## Source comparison (for the record)

| Source | Historical reach | Full text | Cost | Outcome |
|--------|-----------------|-----------|------|---------|
| **GDELT 2.0 DOC API** | 2017→ (GKG to 2015) | No — title + metadata, no tone | Free, no key | **Chosen** |
| NewsAPI.org | ~1 month on free tier | Yes | Archive is paid | Rejected: no COVID backfill |
| Event Registry | 2014→ | Yes | Paid; academic tiers | Rejected: budget |
| Common Crawl CC-NEWS | 2016→ | Yes | Free but TB-scale | Rejected: processing cost |
| Open-Meteo Archive | 1940→ | n/a | Free (non-commercial) | **Also used**, for the weather factor |

Rate limits and tier details change; verify against current docs before
building.

## Query design

Two query families, both tagged so recall can be audited later.

**Port-specific.** Each port's name plus aliases and operating authority:

```
"Port of Los Angeles" OR "POLA" OR "San Pedro Bay" OR "Los Angeles port"
```

**Factor-specific**, one query per risk category, scoped to shipping
vocabulary so "labor shortage" does not pull in unrelated sectors:

```
("labor shortage" OR "dockworker" OR "ILWU" OR "strike") AND (port OR terminal OR longshore)
```

Queries live in `scsai/queries.yaml`, version-controlled and dated, so a query
change does not silently alter an already-built history.

**GDELT-specific note:** the DOC API caps results per query (250 at the time of
writing) and has no cursor. A busy day for a broad query will silently
truncate. Chunk queries by day — or by 6-hour window for high-volume periods —
and record the returned count so truncation is visible rather than invisible.

## Design decisions worth pinning now

**Backfill is chunked and resumable.** Fetch day by day, write each day's raw
JSONL immediately, skip days already on disk. A multi-year backfill will be
interrupted; it must not restart from zero.

**Deduplication is mandatory, not cosmetic.** Wire stories are republished
across dozens of domains. Without dedupe, one AP story about a strike becomes
40 articles and that day's topic score is inflated by syndication volume rather
than real signal. Canonicalise URLs (strip UTM, resolve redirects), then
cluster near-identical titles within a 3-day window and keep the earliest.
This matters more than usual here: titles are the only text available, so
duplicate titles are also duplicate model inputs.

**Timezone.** Normalise everything to UTC dates. Satellite captures have a
local overpass time; mixing local and UTC dates introduces a spurious ±1 day
lag that would corrupt any lag analysis.

**Article volume is a confound, not a signal.** Total news volume rises over
time and spikes on unrelated major events. Daily topic scores must be
normalised against total port-related volume that day, otherwise Component 3
learns the news cycle rather than port conditions.

## Milestones

1. **Feasibility probe** — one week of GDELT for LA Port, inspected by hand.
   Confirms coverage is dense enough (target: ≥5 relevant articles/day) and
   that titles are informative enough to classify.
2. GDELT client + cache layer with fixture-based tests.
3. `queries.yaml` v1, precision spot-checked on 50 sampled articles.
4. Backfill over the agreed window, resumable.
5. Normalise + dedupe, with dedupe rate reported.
6. Weather ingestion via Open-Meteo, for the dates imagery exists.
7. Daily aggregation (counts + tone), handed to Component 3.

## Testing

Record real GDELT responses as fixtures and test the client, normalise and
dedupe steps against those — no live calls in the test suite. Assert
specifically on: date normalisation across timezones, dedupe collapsing a known
syndicated story, truncation being detected when a day hits the result cap, and
empty-result days producing a zero row rather than a missing row.

## Open questions

- **Q4 — Ports and date range.** LA/Long Beach only, or multiple ports?
  Multiple ports multiply the sample size, which matters a great deal given the
  manual-imagery constraint — 3 ports × 20 dates is 60 observations for the
  same per-port collection effort. What date window: 2019–2026?
- **Q5 — Non-English news.** Include and translate, or English only? Low stakes
  for US West Coast ports; high stakes for Asian or European ports.


---

## Implementation notes (added when the module was built)

### The sample-size problem is solved, by a different route

This plan assumed manual collection and budgeted an afternoon in Google Earth
Pro to reach 50-60 dates. That is no longer the cheapest path. Sentinel-2 L2A
is open data on AWS behind the Earth Search STAC API, so `1_ship_detection/vision/imagery.py`
fetches dated scenes with no key and no quota, and the AOI can be windowed out
of each COG for a few MB. The run behind `results/results.md` used **76 usable
capture dates** spanning 2018-09 to 2026-04, which clears the 60-date row of
the table above.

What replaced the constraint is weather. Roughly a third of candidate dates are
discarded because San Pedro Bay is under a marine layer, and those discards
cluster from March to August — so the usable sample is not missing at random,
and the 2020 demand-collapse window has **no usable dates at all**.

### What the live API forced to change

| Issue | Response |
|---|---|
| ArtList caps at 250 with no cursor, and the busy weeks are exactly the congestion episodes | `search_window` halves the window recursively until it stops truncating — 2-4 extra requests, against 7 for a day-by-day refetch |
| `RemoteDisconnected` is an `HTTPException`, not a `URLError` | Widened the retry clause; the narrow one let a 40-date run die after one date |
| Sustained use trips a block that 429s *every* request for tens of minutes | `_cooldown` stands down in escalating minutes and probes with `api_available` before resuming, rather than burning the run in retries |
| One bad date should not discard an hour of successful fetches | `collect` records the failure, carries on, and reports the gaps; the per-date cache means a re-run fetches only what is missing |

**The rate limiting is the practical bottleneck**, not coverage. Budget hours,
not minutes, for a first collection over 60+ dates, and run it well before the
analysis is needed. `results/results.md` was generated while the API was blocked,
which is why its ranking covers weather and seasonality only.
