# handoff/ — the only point of contact between components

```
1_ship_detection ──> ship_counts.csv, detections/ ──┐
2_factors        ──> factors_daily.csv            ──┴──> 3_ranking
```

Components 1 and 2 only **write** here; 3 only **reads** here. No component
imports another's code. If a column changes, change it here first.

## ship_counts.csv — one row per satellite capture

| column | meaning |
|---|---|
| `port_id`, `date` | port and capture date (YYYY-MM-DD) |
| `n_ships` | ships detected on water |
| `water_area_km2`, `density` | area measured over; ships per km² (**the target**) |
| `occupancy` | low / medium / high vs the pre-COVID baseline |
| `n_proposals`, `mean_conf`, `aoi_cloud_frac`, `scene` | detector diagnostics |

## factors_daily.csv — one row per calendar day

| column | meaning |
|---|---|
| `date` | YYYY-MM-DD |
| `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `wind_gusts_10m_max` | Open-Meteo daily weather |
| `news_covered` | 1 if GDELT was fetched for that day. **0 means missing, not zero news** |
| `n_articles`, `n_relevant`, `tone_sum_relevant` | headline totals (blank when not covered) |
| `count_<topic>` × 12 | relevant headlines per risk category |

Daily counts, not shares: 3_ranking builds shares and weather summaries over
the 7 days before each capture date.

## detections/

Annotated scenes for the busiest and quietest dates,
`<date>_<count>ships_<occupancy>.jpg`. Shown in the dashboard.
