"""Component 3 entry point: handoff CSVs -> which factors move ship counts.

    python -m ranking.rank --explain 2021-10-10

Reads ONLY handoff/ship_counts.csv and handoff/factors_daily.csv. Turns the
daily factors into features over the window before each capture date, picks an
analysis method from the number of dates, and writes to outputs/:

    factor_ranking.csv    the ranking, when the sample supports one
    study_features.csv    the design matrix actually modelled
    study_result.json     the same findings as structured data (feeds app.py)
    results.md            the write-up, including what could not be claimed
"""

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import analysis

# Seasonality has to be in the model before any topic is credited with an
# effect: peak season and Lunar New Year move port traffic hard and correlate
# with news volume (3_ranking/plan-topics.md, "Control for seasonality").
LUNAR_NEW_YEAR = {
    2018: date(2018, 2, 16), 2019: date(2019, 2, 5), 2020: date(2020, 1, 25),
    2021: date(2021, 2, 12), 2022: date(2022, 2, 1), 2023: date(2023, 1, 22),
    2024: date(2024, 2, 10), 2025: date(2025, 1, 29), 2026: date(2026, 2, 17),
}


def days_to_lny(d):
    """Absolute days to the nearest Lunar New Year, capped at 90."""
    best = 999
    for year in (d.year - 1, d.year, d.year + 1):
        if year in LUNAR_NEW_YEAR:
            best = min(best, abs((d - LUNAR_NEW_YEAR[year]).days))
    return min(best, 90)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_ship_counts(path):
    rows = read_csv(path)
    for r in rows:
        r["date_obj"] = date.fromisoformat(r["date"])
        r["n_ships"] = int(r["n_ships"])
        r["density"] = float(r["density"])
    return sorted(rows, key=lambda r: r["date_obj"])


def _num(v):
    return None if v in (None, "") else float(v)


def window_features(daily, d, window=7, groups=None):
    """Factor values over the `window` days ending on capture date `d`.

    Weather is summarised two ways - the day itself, and the run-up, because a
    week of gales backs up a port where one windy morning does not. News
    becomes topic SHARES of relevant headlines over the window, so the feature
    tracks what the news was about rather than how much there was. News is
    None unless every day in the window was actually fetched.
    """
    days = [daily.get((d - timedelta(days=k)).isoformat()) for k in range(window)]
    days = [x for x in days if x]
    wind = [v for x in days if (v := _num(x["wind_speed_10m_max"])) is not None]
    gust = [v for x in days if (v := _num(x["wind_gusts_10m_max"])) is not None]
    rain = [v for x in days if (v := _num(x["precipitation_sum"])) is not None]
    today = daily.get(d.isoformat(), {})

    wx = {
        "wind_speed_10m_max_win_mean": sum(wind) / len(wind) if wind else 0.0,
        "wind_gusts_10m_max_win_max": max(gust) if gust else 0.0,
        "precipitation_sum_win_mean": sum(rain) / len(rain) if rain else 0.0,
        "windy_days_win": float(sum(g >= 12.0 for g in gust)),
        "wet_days_win": float(sum(r >= 1.0 for r in rain)),
        "wind_speed_10m_max_day": _num(today.get("wind_speed_10m_max")) or 0.0,
    }

    if len(days) < window or not all(x["news_covered"] == "1" for x in days):
        return wx, None
    topics = [c[len("count_"):] for c in days[0] if c.startswith("count_")]
    if groups:   # fold topics into dimensions; unlisted topics keep their own
        grouped = {t for ts in groups.values() for t in ts}
        groups = {**groups, **{t: [t] for t in topics if t not in grouped}}
    else:
        groups = {t: [t] for t in topics}
    n_rel = sum(float(x["n_relevant"]) for x in days)
    news = {f"share_{g}": (sum(float(x[f"count_{t}"]) for x in days
                               for t in ts if f"count_{t}" in x) / n_rel
                           if n_rel else 0.0) for g, ts in groups.items()}
    news["tone_mean_relevant"] = (sum(float(x["tone_sum_relevant"]) for x in days)
                                  / n_rel if n_rel else 0.0)
    news["news_volume"] = n_rel
    return wx, news


def build_features(counts, daily, use_news, window=7, groups=None):
    """One row per capture date: target density plus every candidate factor.

    With `use_news`, dates whose window lacks full news coverage are dropped
    rather than filled - a gap in GDELT is not a quiet news week.
    """
    rows, y, dates = [], [], []
    t0 = counts[0]["date_obj"]
    for r in counts:
        d = r["date_obj"]
        wx, news = window_features(daily, d, window, groups)
        if use_news and news is None:
            continue
        month_angle = 2 * np.pi * (d.month - 1) / 12.0
        rec = {**(news if use_news else {}), **wx,
               "days_to_lny": float(days_to_lny(d)),
               "month_sin": float(np.sin(month_angle)),
               "month_cos": float(np.cos(month_angle)),
               "time_trend": (d - t0).days / 365.25}
        rows.append(rec)
        y.append(r["density"])
        dates.append(r["date"])

    names = list(rows[0]) if rows else []
    X = np.array([[r[c] for c in names] for r in rows], dtype="float64")
    return X, np.array(y, dtype="float64"), dates, names, rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main(argv=None):
    from .config import (FACTOR_RANKING, FACTORS_DAILY, RESULTS_MD, SHIP_COUNTS,
                         STUDY_FEATURES, STUDY_RESULT, TOPIC_GROUPS)

    p = argparse.ArgumentParser(description="Rank the factors behind ship counts.")
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--explain", default=None,
                   help="capture date (YYYY-MM-DD) to decompose")
    p.add_argument("--boot", type=int, default=400)
    p.add_argument("--news", choices=["auto", "on", "off"], default="auto",
                   help="auto: use news factors only if every capture date has "
                        "full GDELT coverage; on: drop dates that do not")
    p.add_argument("--topics", choices=["grouped", "individual"], default="grouped",
                   help="grouped: fold the 12 topics into the dimensions in "
                        "config.TOPIC_GROUPS; individual: one factor per topic")
    args = p.parse_args(argv)
    groups = TOPIC_GROUPS if args.topics == "grouped" else None

    counts = read_ship_counts(SHIP_COUNTS)
    daily = {r["date"]: r for r in read_csv(FACTORS_DAILY)}
    print(f"Ship counts: {len(counts)} capture dates, "
          f"{counts[0]['date']} -> {counts[-1]['date']}")

    n_cov = sum(window_features(daily, r["date_obj"], args.window, groups)[1] is not None
                for r in counts)
    have_news = args.news == "on" or (args.news == "auto" and n_cov == len(counts))
    print(f"News coverage: {n_cov}/{len(counts)} capture windows -> news factors "
          + ("ON" if have_news else "OFF (run 2_factors with --fetch-news to fill)"))

    X, y, dts, names, rows = build_features(counts, daily, have_news, args.window, groups)
    kept = set(dts)
    occ = [r for r in counts if r["date"] in kept]
    print(f"\nDesign matrix: {X.shape[0]} dates x {X.shape[1]} features")

    joined = [{"date": d, "density": float(v), **r} for d, v, r in zip(dts, y, rows)]
    write_csv(STUDY_FEATURES, joined)

    gate = analysis.choose_method(len(y))
    print(f"\nSample-size gate: {len(y)} dates -> {gate['method']}")
    print(f"  {gate['description']}")

    out = {"gate": gate, "n_dates": len(y), "dates": dts,
           "has_news_factors": have_news, "features": names}

    # COVID is asked for by name in requirements.md and is valid at every sample
    # size, so this runs regardless of which gate was selected.
    #
    # It has to be split into two phases, because at San Pedro Bay the pandemic
    # moved ship counts in OPPOSITE directions. Spring 2020 was a demand
    # collapse - blank sailings, fewer vessels. From late 2020 the import surge
    # produced the anchorage backlog, the single busiest period on record. A
    # single "COVID window" averages a drop against a spike and reports roughly
    # nothing, which would answer the requirement's question wrongly rather
    # than answering it weakly.
    periods = {
        "pre_covid": lambda d: d < "2020-03-01",
        "covid_collapse": lambda d: "2020-03-01" <= d <= "2020-08-31",
        "covid_backlog": lambda d: "2020-09-01" <= d <= "2022-06-30",
        "post": lambda d: d > "2022-06-30",
    }
    grouped = {k: [v for d, v in zip(dts, y) if f(d)] for k, f in periods.items()}
    out["periods"] = {k: {"n": len(v), "mean": float(np.mean(v)) if v else None}
                      for k, v in grouped.items()}
    print("\nShip density by period (ships/km2):")
    for k, v in out["periods"].items():
        if v["n"]:
            print(f"  {k:16s} n={v['n']:3d}  mean {v['mean']:.3f}")

    baseline = grouped["pre_covid"] + grouped["post"]
    out["covid"] = {}
    for phase in ("covid_collapse", "covid_backlog"):
        if len(grouped[phase]) >= 3 and len(baseline) >= 3:
            c = analysis.two_group_comparison(
                grouped[phase], baseline, phase, "pre-COVID + post-2022")
            out["covid"][phase] = c
            print(f"  {phase} vs baseline: {c['mean_a']:.3f} vs {c['mean_b']:.3f} "
                  f"({c['pct_change']:+.1f}%), 95% CI "
                  f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}], "
                  f"MW p={c['mannwhitney_p']:.3f}  (n={c['n_a']} vs {c['n_b']})")

    if gate["method"] == "elastic_net":
        print(f"\nRanking {X.shape[1]} factors (elastic net + {args.boot} bootstraps)...")
        res = analysis.rank_factors(X, y, names, n_boot=args.boot)
        out["ranking"] = res
        write_csv(FACTOR_RANKING, res["ranking"])
        print(f"  lambda={res['lambda']:.4f}\n")
        print(f"  {'factor':32s} {'beta':>8s} {'sel':>5s} {'sign':>5s} "
              f"{'rho':>6s} {'p_adj':>7s}")
        for r in res["ranking"][:15]:
            print(f"  {r['factor']:32s} {r['enet_beta']:+8.3f} "
                  f"{r['selection_rate']:5.2f} {r['sign_consistency']:5.2f} "
                  f"{r['spearman_rho']:+6.2f} {r['spearman_p_adj']:7.3f}"
                  + ("  *" if r["bh_survives"] else ""))

        if args.explain:
            if args.explain not in dts:
                print(f"\n{args.explain} is not a capture date; "
                      f"nearest available: {min(dts, key=lambda d: abs((datetime.strptime(d,'%Y-%m-%d').date() - datetime.strptime(args.explain,'%Y-%m-%d').date()).days))}")
            else:
                i = dts.index(args.explain)
                ex = analysis.explain_date(X, y, names, i, lam=res["lambda"])
                out["explain"] = {"date": args.explain, **ex}
                print(f"\nFactor attribution for {args.explain}")
                print(f"  actual {ex['actual']:.3f} ships/km2 | "
                      f"model {ex['predicted']:.3f} | baseline {ex['baseline']:.3f} | "
                      f"residual {ex['residual']:+.3f}")
                for c in ex["contributions"][:10]:
                    bar = "+" if c["contribution"] > 0 else "-"
                    print(f"    {bar} {c['factor']:30s} {c['contribution']:+.4f}  "
                          f"(value {c['value']:.3f}, z {c['z']:+.2f})")

    STUDY_RESULT.write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    write_report(RESULTS_MD, out, occ, names)
    print(f"\nWrote {STUDY_FEATURES.name}, {STUDY_RESULT.name} and "
          f"{RESULTS_MD.name} to {RESULTS_MD.parent}")
    return out


def write_report(path, out, occ, feature_names):
    """Render the findings, including the parts the sample cannot support."""
    gate = out["gate"]
    L = []
    L.append("# Results\n")
    L.append("Generated by `python -m ranking.rank`. Every number here comes from "
             "the pipeline in this repo; nothing is hand-entered.\n")

    L.append("## Sample\n")
    L.append(f"- Capture dates analysed: **{out['n_dates']}** "
             f"({occ[0]['date']} to {occ[-1]['date']})")
    L.append(f"- Ship counts: min {min(r['n_ships'] for r in occ)}, "
             f"median {int(np.median([r['n_ships'] for r in occ]))}, "
             f"max {max(r['n_ships'] for r in occ)}")
    L.append(f"- Method selected by the sample-size gate: **{gate['method']}** "
             f"(threshold {gate['threshold']} dates)")
    L.append(f"- {gate['description']}\n")

    if out.get("periods"):
        L.append("## COVID vs normal\n")
        L.append("Answers *\"do ships during covid have less than ships during "
                 "normal times\"* from requirements.md — but the pandemic has to "
                 "be split in two, because at San Pedro Bay it moved ship counts "
                 "in opposite directions. Spring 2020 was a demand collapse; from "
                 "late 2020 the import surge produced the anchorage backlog. "
                 "Averaging them together answers the question wrongly rather "
                 "than weakly.\n")
        L.append("| period | dates | mean ships/km2 |")
        L.append("|---|---:|---:|")
        for k, v in out["periods"].items():
            mean = f"{v['mean']:.3f}" if v["n"] else "-"
            L.append(f"| {k} | {v['n']} | {mean} |")
        L.append("")
        empty = [k for k, v in out["periods"].items() if not v["n"]]
        if empty:
            L.append(f"No usable dates fall in: **{', '.join(empty)}**. This is the "
                     "haze screen biting, not an absence of satellite passes — "
                     "March to August over San Pedro Bay is marine-layer season, "
                     "and the 2020 demand-collapse window sits inside it. The "
                     "collapse phase therefore cannot be measured here at all, "
                     "which is a gap in the answer to the requirement's COVID "
                     "question rather than a null result for it.\n")
        for phase, c in out.get("covid", {}).items():
            crosses = c["ci_low"] <= 0 <= c["ci_high"]
            L.append(f"**{phase}** vs pre-COVID + post-2022 baseline: "
                     f"{c['mean_a']:.3f} vs {c['mean_b']:.3f} ships/km2, "
                     f"**{c['pct_change']:+.1f}%** "
                     f"(bootstrap 95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}], "
                     f"Mann-Whitney p = {c['mannwhitney_p']:.4f}, "
                     f"n={c['n_a']} vs {c['n_b']}). The interval "
                     f"{'includes' if crosses else 'excludes'} zero, so the shift is "
                     f"{'not distinguishable from noise' if crosses else 'detectable'} "
                     f"at this sample size.\n")

    if "ranking" in out:
        r = out["ranking"]
        L.append("## Factor ranking\n")
        L.append(f"Elastic net (lambda={r['lambda']:.4f}, chosen by blocked "
                 f"time-series CV) over {r['n_features']} candidate factors on "
                 f"{r['n']} dates, with bootstrap resampling for stability.\n")
        L.append("Read `selection_rate` first: it is how often the factor survives "
                 "regularisation under resampling. A large coefficient selected "
                 "half the time is noise with a number attached.\n")
        L.append("| factor | coef | selection rate | sign consistency | Spearman rho | p (BH-adj) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for row in r["ranking"]:
            L.append(f"| {row['factor']} | {row['enet_beta']:+.3f} | "
                     f"{row['selection_rate']:.2f} | {row['sign_consistency']:.2f} | "
                     f"{row['spearman_rho']:+.2f} | {row['spearman_p_adj']:.3f}"
                     f"{' **' if row['bh_survives'] else ''} |")
        L.append("")
        stable = [x for x in r["ranking"]
                  if x["selection_rate"] >= 0.8 and abs(x["enet_beta"]) > 1e-6]
        significant = [x for x in r["ranking"] if x["bh_survives"]]
        L.append(f"**{len(stable)} of {r['n_features']} candidate factors** are "
                 f"selected in at least 80% of bootstrap resamples with a "
                 f"non-zero coefficient, and **{len(significant)}** survive "
                 f"Benjamini-Hochberg correction on the univariate test.\n")
        if not significant:
            L.append("Selection rate alone is not a finding here. The "
                     f"cross-validated penalty came out very small "
                     f"(lambda={r['lambda']:.4f}), which is what happens when no "
                     "feature predicts well: the CV has nothing to gain by "
                     "shrinking, so almost every coefficient stays non-zero and "
                     "selection rates run high across the board. The "
                     "coefficients themselves are near zero and no factor "
                     "survives multiple-comparison correction. The honest "
                     "reading is that **none of these factors shows a "
                     "detectable association with ship density** at this "
                     "sample size.\n")

    if "explain" in out:
        e = out["explain"]
        L.append(f"## Factor attribution for {e['date']}\n")
        L.append(f"- Observed: **{e['actual']:.3f}** ships/km2")
        L.append(f"- Model: {e['predicted']:.3f} (baseline {e['baseline']:.3f}, "
                 f"residual {e['residual']:+.3f})\n")
        L.append("| factor | value | z | coef | contribution |")
        L.append("|---|---:|---:|---:|---:|")
        for c in e["contributions"][:12]:
            L.append(f"| {c['factor']} | {c['value']:.3f} | {c['z']:+.2f} | "
                     f"{c['beta']:+.3f} | {c['contribution']:+.4f} |")
        L.append("")
        L.append("Contributions are `coef x standardised value`, so they sum with the "
                 "baseline to the model's prediction. They are attribution under the "
                 "fitted model, not causes. A large residual means the day was driven "
                 "by something outside the feature set.\n")

    if not out.get("has_news_factors", True):
        L.append("## News factors are absent from this run\n")
        L.append("News coverage from GDELT is incomplete, so the ranking above "
                 "covers weather and seasonality only. The topic factors from "
                 "requirements.md — labour, policy, conflict, pandemic and the "
                 "rest — are **not** in it, and their absence from the table is "
                 "not evidence that they do not matter. Re-run "
                 "`python -m factors.build --fetch-news` in 2_factors, then this "
                 "step; the response cache means only missing days are "
                 "fetched.\n")

    L.append("## What this does not establish\n")
    L.append("- **Association, not causation.** One port, a handful of disruption "
             "episodes, and no counterfactual.")
    L.append("- **Topic labels are keyword rules**, not the trained classifier "
             "3_ranking/plan-topics.md specifies, and the 300-title gold set does not "
             "exist yet. Category precision is unmeasured and bounds everything "
             "downstream.")
    L.append("- **Ship counts carry detector error.** Held-out precision and recall "
             "are reported in `ship_detector_s2.report.json`; that uncertainty "
             "is not propagated into the intervals above.")
    L.append("- **Clear-sky dates only.** Scenes under the marine layer are "
             "discarded, which thins summer months and is not a random "
             "missingness pattern.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
