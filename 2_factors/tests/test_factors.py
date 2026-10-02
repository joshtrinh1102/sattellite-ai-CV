"""Component 2 - dedupe, topic classification and the daily handoff rows."""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors import config, news, topics
from factors.news import _article_id, _normalise_title


def test_output_goes_to_handoff():
    assert config.FACTORS_DAILY.parent == config.HANDOFF


# --------------------------------------------------------------------------
# News - dedupe and coverage
# --------------------------------------------------------------------------

def test_article_id_normalises_urls():
    a = _article_id("https://www.Reuters.com/a/b/?utm_source=x")
    b = _article_id("http://reuters.com/a/b")
    assert a == b


def test_normalise_title_strips_outlet_suffix():
    assert (_normalise_title("Port of LA backlog hits record - Reuters")
            == _normalise_title("Port of LA backlog hits record | Splash247"))


def _cache(tmp_path, end, articles, window=7):
    (tmp_path / f"USLAX_{end}_{window}d.json").write_text(json.dumps({
        "capture_date": end, "window_days": window, "truncated": False,
        "articles": articles}), encoding="utf-8")


def test_collect_bins_by_day_and_reports_coverage(tmp_path):
    art = {"title": "Container backlog grows off Los Angeles", "url": "http://x/1",
           "language": "English", "seendate": "20211008T120000Z", "domain": "x"}
    _cache(tmp_path, "2021-10-10", [art])
    _cache(tmp_path, "2021-10-12", [art])   # overlapping window, same article
    rows, covered = news.collect("USLAX", date(2021, 10, 1), date(2021, 10, 31),
                                 cache_dir=tmp_path, fetch=False)
    assert len(rows) == 1 and rows[0]["date"] == "2021-10-08"
    assert min(covered) == date(2021, 10, 4) and max(covered) == date(2021, 10, 12)
    assert len(covered) == 9


# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("ILWU dockworkers begin strike at Port of Los Angeles", "labor_shortages"),
    ("Record backlog of container ships waiting off Los Angeles", "logistics_reliability"),
    ("COVID-19 outbreak forces Ningbo terminal closure", "health_pandemic"),
    ("Red Sea attacks push war risk premiums higher", "conflict_war"),
    ("Ransomware attack hits terminal operating system", "cybersecurity"),
    ("New tariffs on Chinese imports take effect Monday", "global_regulations"),
    ("Chassis shortage leaves empty containers stacked at terminals", "material_shortages"),
])
def test_classify_assigns_the_expected_category(title, expected):
    assert topics.classify(title)["label"] == expected


def test_classify_routes_non_operational_news_to_irrelevant():
    """The escape hatch is the point of the 13th class - without it these get
    forced into a risk bucket and become noise in the impact model."""
    for title in ("Port of LA names new executive director",
                  "Cruise terminal opens new passenger lounge"):
        assert topics.classify(title)["label"] == "irrelevant"


def test_title_tone_is_signed():
    assert topics.title_tone("Port congestion crisis delays cargo") < 0
    assert topics.title_tone("Port backlog clears as volumes improve") > 0
    assert topics.title_tone("Port of Los Angeles") == 0.0


def test_daily_scores_counts_exclude_irrelevant():
    articles = [
        {"date": "2021-10-10", "title": "ILWU strike halts terminal", "url": "u1"},
        {"date": "2021-10-10", "title": "Congestion backlog grows", "url": "u2"},
        {"date": "2021-10-10", "title": "Port names new director", "url": "u3"},
    ]
    _, daily = topics.daily_scores(articles)
    row = daily.iloc[0]
    assert row["n_articles"] == 3 and row["n_relevant"] == 2
    assert row["count_labor_shortages"] == 1
    assert row["count_logistics_reliability"] == 1
