"""Component 3 - window features, the impact model and seasonality control."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranking import analysis, config
from ranking.rank import days_to_lny, window_features


def test_inputs_come_only_from_handoff():
    for p in (config.SHIP_COUNTS, config.FACTORS_DAILY, config.DETECTIONS):
        assert p.parent == config.HANDOFF


# --------------------------------------------------------------------------
# Daily factors -> window features
# --------------------------------------------------------------------------

def _day(d, covered=True, n_rel=0, labour=0, tone=0.0, wind=5.0, gust=10.0, rain=0.0):
    return {"date": d.isoformat(), "wind_speed_10m_max": str(wind),
            "wind_gusts_10m_max": str(gust), "precipitation_sum": str(rain),
            "news_covered": "1" if covered else "0",
            "n_relevant": str(n_rel) if covered else "",
            "tone_sum_relevant": str(tone) if covered else "",
            "count_labor_shortages": str(labour) if covered else "",
            "count_conflict_war": "0" if covered else ""}


def _week(end, overrides=None):
    """Seven daily rows ending on `end`; `overrides` maps days-before-end -> fields."""
    days = {}
    for k in range(7):
        d = end - timedelta(days=k)
        days[d.isoformat()] = _day(d, **(overrides or {}).get(k, {}))
    return days


def test_news_shares_pool_counts_over_the_window():
    end = date(2021, 10, 10)
    daily = _week(end, {0: {"n_rel": 3, "labour": 3, "tone": -1.5},
                          3: {"n_rel": 1, "labour": 0, "tone": 0.5}})
    _, news = window_features(daily, end)
    # 3 of 4 relevant headlines in the week were labour, not mean(1.0, 0.0)
    assert news["share_labor_shortages"] == pytest.approx(0.75)
    assert news["tone_mean_relevant"] == pytest.approx(-0.25)
    assert news["news_volume"] == 4


def test_topic_groups_sum_their_members_and_keep_unlisted_topics():
    end = date(2021, 10, 10)
    daily = _week(end, {0: {"n_rel": 4, "labour": 1}})
    for row in daily.values():
        row["count_conflict_war"] = "1" if row["n_relevant"] == "4" else "0"
    _, news = window_features(daily, end, groups={"both": ["labor_shortages", "conflict_war"]})
    assert news["share_both"] == pytest.approx(0.5)      # (1 + 1) / 4
    assert "share_labor_shortages" not in news
    _, news = window_features(daily, end, groups={"supply": ["labor_shortages"]})
    assert "share_conflict_war" in news                  # unlisted topic survives


def test_one_uncovered_day_drops_news_for_the_window():
    end = date(2021, 10, 10)
    daily = _week(end, {4: {"covered": False}})
    wx, news = window_features(daily, end)
    assert news is None
    assert wx["wind_speed_10m_max_win_mean"] == pytest.approx(5.0)


def test_weather_window_counts_rough_days():
    end = date(2021, 10, 10)
    daily = _week(end, {1: {"gust": 15.0, "rain": 2.0}, 2: {"gust": 12.0}})
    wx, _ = window_features(daily, end)
    assert wx["windy_days_win"] == 2 and wx["wet_days_win"] == 1
    assert wx["wind_gusts_10m_max_win_max"] == 15.0


# --------------------------------------------------------------------------
# The impact model
# --------------------------------------------------------------------------

def test_choose_method_follows_the_sample_size_table():
    assert analysis.choose_method(66)["method"] == "elastic_net"
    assert analysis.choose_method(30)["method"] == "prespecified"
    assert analysis.choose_method(9)["method"] == "two_group"


def test_rank_factors_recovers_planted_signal_and_shrinks_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(66, 14))
    y = 1.5 * X[:, 2] - 2.0 * X[:, 7] + rng.normal(scale=0.6, size=66) + 10
    names = [f"f{i}" for i in range(14)]
    res = analysis.rank_factors(X, y, names, n_boot=120)

    top2 = {r["factor"] for r in res["ranking"][:2]}
    assert top2 == {"f2", "f7"}
    for r in res["ranking"][:2]:
        assert r["selection_rate"] == 1.0
        assert r["bh_survives"]
    # noise features must not walk away with a large coefficient
    noise = [r for r in res["ranking"] if r["factor"] not in top2]
    assert max(abs(r["enet_beta"]) for r in noise) < 0.3


def test_explain_date_decomposition_reconstructs_the_prediction():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    y = X[:, 1] * 2 + rng.normal(scale=0.3, size=40)
    names = [f"f{i}" for i in range(6)]
    ex = analysis.explain_date(X, y, names, target_index=7, lam=0.05)
    total = ex["baseline"] + sum(c["contribution"] for c in ex["contributions"])
    assert total == pytest.approx(ex["predicted"])
    assert ex["actual"] - ex["predicted"] == pytest.approx(ex["residual"])


def test_benjamini_hochberg_is_less_strict_than_bonferroni():
    p = np.array([0.001, 0.008, 0.02, 0.6, 0.9])
    survives, adj = analysis.benjamini_hochberg(p, q=0.05)
    assert survives[0] and survives[1]
    assert not survives[3] and not survives[4]
    assert np.all(adj >= p)


def test_two_group_comparison_detects_a_real_shift():
    rng = np.random.default_rng(5)
    low = rng.normal(0.8, 0.1, 15)
    high = rng.normal(1.2, 0.1, 20)
    res = analysis.two_group_comparison(low, high, "low", "high", n_boot=2000)
    assert res["diff"] < 0
    assert res["ci_high"] < 0            # interval excludes zero
    assert res["mannwhitney_p"] < 0.01


def test_blocked_folds_keep_time_contiguous():
    folds = analysis.blocked_folds(20, k=4)
    assert [len(f) for f in folds] == [5, 5, 5, 5]
    for f in folds:
        assert list(f) == list(range(f[0], f[0] + len(f)))


def test_days_to_lny_is_zero_on_the_day_and_capped_far_away():
    assert days_to_lny(date(2021, 2, 12)) == 0
    assert days_to_lny(date(2021, 2, 15)) == 3
    assert days_to_lny(date(2021, 7, 1)) == 90
