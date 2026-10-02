# Prediction of Supply Chain Health Using Satellite Imagery

Measuring port congestion from satellite imagery, and working out which
real-world factors actually move it.

The research question, from [settings/requirements.md](settings/requirements.md):
given ~10 candidate factors that plausibly disrupt port traffic (COVID, weather,
war, labour actions, policy changes, ...), which ones genuinely correlate with
observed ship activity, and by how much?

## Three components, one point of contact

```
1_ship_detection ──> handoff/ship_counts.csv   ──┐
2_factors        ──> handoff/factors_daily.csv ──┴──> 3_ranking ──> factor_ranking.csv
                                                                  + Streamlit dashboard
```

| Folder | Does | Output (in [handoff/](handoff/README.md)) |
|---|---|---|
| [1_ship_detection/](1_ship_detection/README.md) | Sentinel-2 scenes → ships counted on water | `ship_counts.csv` — one row per capture date |
| [2_factors/](2_factors/README.md) | News (GDELT) + weather (Open-Meteo) → daily numbers | `factors_daily.csv` — one row per day |
| [3_ranking/](3_ranking/README.md) | Joins both, ranks factors, dashboard | `outputs/factor_ranking.csv`, `app.py` |

Components never import each other. 1 and 2 only write to `handoff/`; 3 only
reads from it. The column contract is in [handoff/README.md](handoff/README.md).

## Quickstart

```bash
pip install -r settings/requirements.txt

cd 1_ship_detection && python -m vision.pipeline && cd ..   # ship counts (needs scenes + model)
cd 2_factors        && python -m factors.build   && cd ..   # daily factors (offline, from cache)
cd 3_ranking        && python -m ranking.rank    && cd ..       # ranking -> 3_ranking/outputs/
streamlit run 3_ranking/app.py                              # dashboard, from the repo root
```

Each step only needs the previous handoff files, so you can rerun any one alone.
Tests: `pytest -c settings/pytest.ini`.

## Layout

```
1_ship_detection/   vision/ code · data/ · models/ · outputs/ · tests/ · notebook
2_factors/          factors/ code · cache/ · outputs/ · tests/ · plan-news.md
3_ranking/          ranking/ code · app.py · outputs/ · reports/ · tests/ · plan-topics.md
handoff/            the only files that cross folders
settings/           requirements, .gitignore, .env.example, pytest.ini, chat logs
.venv_tf/           local virtualenv
```

`.gitignore` lives in `settings/`. Git only reads a root `.gitignore`, so after
`git init` run `git config core.excludesFile settings/.gitignore`.

## Current result, and its limits

76 capture dates, 10 weather and season factors. **No factor survives
multiple-comparison correction.** News factors are not in the ranking yet:
GDELT coverage is 21 of 76 capture windows (see
[2_factors](2_factors/README.md)). Full write-up:
[3_ranking/outputs/results.md](3_ranking/outputs/results.md).

- **Association, not causation.** One port, a handful of disruption episodes.
- **Topic labels are keyword rules**, not a trained classifier, so category
  precision is unmeasured.
- **Detector error is reported but not propagated** into the factor intervals.
- **Clear-sky dates only**, which thins summer, so the sample is not missing at random.

## Author

**Long Trinh** — MS Artificial Intelligence and Business Analytics
University of South Florida (USF)
