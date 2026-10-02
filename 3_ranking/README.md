# 3 — Factor ranking

**In:** `handoff/ship_counts.csv` and `handoff/factors_daily.csv` (nothing else).
**Out:** `outputs/factor_ranking.csv` and a Streamlit dashboard.

Run from this folder.

```bash
python -m ranking.rank --explain 2021-10-10   # -> outputs/
streamlit run app.py                          # dashboard (from the repo root: streamlit run 3_ranking/app.py)
```

`--news auto|on|off` controls news factors. `auto`, the default, uses them only
when every capture date has full 7-day coverage. `on` drops the dates that
don't have it.

| Path | What |
|---|---|
| `ranking/rank.py` | 7-day window features → design matrix → method gate → outputs |
| `ranking/analysis.py` | elastic net, bootstrap, BH correction, attribution (numpy + scipy only) |
| `app.py` | dashboard: ranking · ship counts over time · factor explorer · detection images |
| `outputs/factor_ranking.csv` | per factor: coefficient, bootstrap CI, selection rate, sign consistency, Spearman, BH |
| `outputs/results.md` | the write-up, including what the numbers cannot establish |
| `outputs/study_features.csv` | the exact design matrix modelled |
| `outputs/study_result.json` | the same findings as structured data |
| `plan-topics.md`, `reports/` | method plan; research and progress reports |

## The sample-size gate

`analysis.choose_method` reads the number of usable dates and refuses methods
the sample can't support:

| Dates | Method | Claim available |
|---|---|---|
| ~10 | two-group | COVID vs normal only |
| ~30 | pre-specified | 2–3 hypotheses chosen in advance |
| ~60 | elastic net | ranking of which factors matter |

Running the 60-date method on 10 dates still prints a confident ranked list,
so the gate is enforced in code.

## Reading the ranking

Read `selection_rate` first. A large coefficient that is selected half the time
is noise. In the current run (76 dates, 10 weather and season factors), the
penalty came out tiny, the coefficients are near zero, and **no factor survives
BH correction**. The honest reading is that there is no detectable association
yet. News factors are absent until 2_factors fills its coverage.
