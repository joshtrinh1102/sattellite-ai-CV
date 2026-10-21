"""Component 3 Stage A - article title -> risk factors (multi-label).

The taxonomy is NOT in this file. It lives in 2_factors/factors.toml, which is
meant to be edited by hand: factors, the dimensions they roll up into, their
keywords and thresholds, and the model weights 3_ranking applies. Changing that
file and rebuilding re-labels every cached headline without re-fetching.

WHICH RUNG OF THE LADDER THIS IS
--------------------------------
3_ranking/plan-topics.md lists three label sources (Q7). This module implements
option 2, **keyword rules** - free, fully reproducible, noisier than the LLM
labelling the plan recommends. The TF-IDF and transformer rungs above it are
still unbuilt, and so is the 300-title gold set the plan calls non-negotiable.
The honest reading of any ranking downstream is therefore "ranked using
rule-assigned topics", and category precision is the limiting factor, not the
impact model.

MULTI-LABEL
-----------
A headline gets EVERY factor whose score reaches that factor's threshold
("ILWU strike worsens port congestion" is both labor and logistics). The
strongest one is kept as `label` so the single-label view is still available.
A headline that reaches no threshold is `irrelevant` - the escape hatch (Q11)
that keeps port-authority appointments and cruise-terminal puff pieces out of
the operational buckets.

Scoring is deliberately not a bag of equal keyword hits:

  * keywords are tiered - "ilwu" is decisive for Labor, "workers" is not;
  * phrases are listed separately from the words in them, so "empty container"
    can be decisive for Material Shortages without the bare word "container"
    firing in every logistics story;
  * a minimum score is required before a factor is assigned, which is what
    routes genuinely off-topic articles to `irrelevant`.
"""

import re
import tomllib
from dataclasses import dataclass, field

from .config import FACTORS_FILE

TIER_POINTS = {"decisive": 3.0, "strong": 2.0, "supporting": 1.0}
WEIGHT_FAMILIES = ("weather", "season", "trend", "news_summary")
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class Factor:
    key: str
    title: str
    dimension: str
    weight: float
    threshold: float
    cues: list = field(default_factory=list)      # [(points, compiled pattern)]


@dataclass
class Taxonomy:
    factors: dict          # key -> Factor, in file order
    dimensions: dict       # key -> {"title", "weight"}
    weights: dict          # non-news feature families -> weight

    def definitions(self):
        """The part 3_ranking needs, written to the handoff as JSON."""
        return {
            "factors": {k: {"title": f.title, "dimension": f.dimension,
                            "weight": f.weight, "threshold": f.threshold}
                        for k, f in self.factors.items()},
            "dimensions": self.dimensions,
            "weights": self.weights,
        }


def compile_keyword(entry):
    """Keyword syntax from factors.toml -> regex over the normalised title."""
    entry = entry.strip()
    if entry.startswith("re:"):
        return re.compile(entry[3:])
    text = " ".join(re.sub(r"[^a-z0-9* ]+", " ", entry.lower()).split())
    prefix = text.endswith("*")
    body = re.escape(text.rstrip("*").rstrip())
    return re.compile(r"\b" + body + ("" if prefix else r"\b"))


def load_taxonomy(path=FACTORS_FILE):
    """Read and validate factors.toml. Errors name the offending entry."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    def positive(value, where):
        value = float(value)
        if value <= 0:
            raise ValueError(f"{path.name}: {where} must be > 0, got {value}")
        return value

    default_threshold = float(raw.get("settings", {}).get("default_threshold", 2.0))
    weights = {fam: positive(raw.get("weights", {}).get(fam, 1.0), f"weights.{fam}")
               for fam in WEIGHT_FAMILIES}

    dimensions = {}
    for key, d in raw.get("dimensions", {}).items():
        if not _KEY.match(key):
            raise ValueError(f"{path.name}: dimension key {key!r} must be lowercase_with_underscores")
        dimensions[key] = {"title": d.get("title", key),
                           "weight": positive(d.get("weight", 1.0), f"dimensions.{key}.weight")}

    factors = {}
    for key, f in raw.get("factors", {}).items():
        if not _KEY.match(key) or key == "irrelevant":
            raise ValueError(f"{path.name}: factor key {key!r} must be lowercase_with_underscores "
                             "and not 'irrelevant'")
        dim = f.get("dimension")
        if dim not in dimensions:
            raise ValueError(f"{path.name}: factors.{key}.dimension = {dim!r} is not one of "
                             f"{sorted(dimensions)}")
        cues = []
        for tier, points in TIER_POINTS.items():
            for entry in f.get(tier, []):
                try:
                    cues.append((points, compile_keyword(entry)))
                except re.error as exc:
                    raise ValueError(f"{path.name}: factors.{key}.{tier}: bad pattern "
                                     f"{entry!r} ({exc})") from None
        if not cues:
            raise ValueError(f"{path.name}: factors.{key} has no keywords")
        factors[key] = Factor(
            key=key, title=f.get("title", key), dimension=dim,
            weight=positive(f.get("weight", 1.0), f"factors.{key}.weight"),
            threshold=positive(f.get("threshold", default_threshold),
                               f"factors.{key}.threshold"),
            cues=cues)
    if not factors:
        raise ValueError(f"{path.name}: no [factors.*] defined")
    return Taxonomy(factors, dimensions, weights)


TAXONOMY = load_taxonomy()
RISK_LABELS = tuple(TAXONOMY.factors)
DIMENSIONS = tuple(TAXONOMY.dimensions)
LABELS = RISK_LABELS + ("irrelevant",)
LABEL_TITLES = {**{k: f.title for k, f in TAXONOMY.factors.items()},
                "irrelevant": "Irrelevant"}


def _normalise(title):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", title.lower()).split())


def score_title(title, taxonomy=None):
    """Return {factor: points} - each matching keyword adds its tier's points."""
    tax = taxonomy or TAXONOMY
    text = _normalise(title)
    return {key: sum(pts for pts, pat in f.cues if pat.search(text))
            for key, f in tax.factors.items()}


def classify(title, taxonomy=None):
    """Every factor at or above its threshold, strongest first.

    `label` is the strongest, kept so single-label consumers and the old
    articles.csv column still work; `labels` is the full multi-label set.
    """
    tax = taxonomy or TAXONOMY
    scores = score_title(title, tax)
    hits = sorted(((k, s) for k, s in scores.items() if s >= tax.factors[k].threshold),
                  key=lambda kv: (-kv[1], kv[0]))
    if not hits:
        return {"label": "irrelevant", "labels": [], "score": float(max(scores.values())),
                "dimensions": []}
    labels = [k for k, _ in hits]
    dims = list(dict.fromkeys(tax.factors[k].dimension for k in labels))
    return {"label": labels[0], "labels": labels, "score": float(hits[0][1]),
            "dimensions": dims}


# ---------------------------------------------------------------------------
# Stage A output - daily topic scores
# ---------------------------------------------------------------------------

# Tone lexicon. GDELT's ArtList gives no V2Tone (see factors/news.py), so tone is
# computed here from the titles, which is the fallback 2_factors/plan-news.md
# settles on. Deliberately small and disruption-oriented rather than a general
# sentiment model: what matters is whether the port news is bad news.
NEGATIVE = re.compile(
    r"\b(crisis|delay|backlog|congestion|strike|shortage|halt|closure|closed|"
    r"disrupt|collapse|plunge|slump|drop|fall|fell|warn|risk|threat|fear|"
    r"cancel|blocked|stuck|stranded|worst|record high|surge|snarl|chaos|"
    r"protest|attack|damage|accident|fire|spill|outage|failure|lawsuit)\w*"
)
POSITIVE = re.compile(
    r"\b(record|growth|grow|rise|improve|recovery|recover|ease|easing|clear|"
    r"cleared|resolve|resolved|agreement|deal|expand|invest|upgrade|boost|"
    r"efficient|reopen|resume|gain|strong)\w*"
)


def title_tone(title):
    """Crude polarity in [-1, 1] from a title. Negative = disruption-flavoured."""
    text = title.lower()
    neg = len(NEGATIVE.findall(text))
    pos = len(POSITIVE.findall(text))
    if neg + pos == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def daily_scores(articles, taxonomy=None):
    """Classify every headline and count them per day, factor and dimension.

    `articles` is the row list from `factors.news.collect`. Returns
    (labelled_articles_df, daily_df) with one daily row per date:

        n_articles, n_relevant         totals for the day
        count_<factor>                 relevant articles labelled with that factor
        dim_<dimension>                relevant articles with ANY factor in it
        tone_sum_relevant              summed title tone over relevant articles

    Multi-label means count_ columns can add up to more than n_relevant. The
    dim_ columns count each article once per dimension, so two labels in the
    same dimension are not double-counted - which is why they are computed
    here and not by summing count_ columns downstream.

    Counts and sums rather than shares and means, because they aggregate
    exactly: a 7-day share is sum(count) / sum(n_relevant) over the window,
    which 3_ranking computes. Articles labelled `irrelevant` stay out of
    n_relevant - they are the escape hatch, not another risk.
    """
    import pandas as pd

    tax = taxonomy or TAXONOMY
    rows = []
    for a in articles:
        c = classify(a["title"], tax)
        rows.append({
            **a,
            "label": c["label"],
            "labels": ";".join(c["labels"]),
            "dimensions": ";".join(c["dimensions"]),
            "label_score": c["score"],
            "tone": title_tone(a["title"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    out = []
    for day, grp in df.groupby("date"):
        relevant = grp[grp["label"] != "irrelevant"]
        labels = [set(s.split(";")) for s in relevant["labels"]]
        dims = [set(s.split(";")) for s in relevant["dimensions"]]
        rec = {
            "date": day,
            "n_articles": len(grp),
            "n_relevant": len(relevant),
            "tone_sum_relevant": float(relevant["tone"].sum()),
        }
        for key in tax.factors:
            rec[f"count_{key}"] = sum(key in ls for ls in labels)
        for key in tax.dimensions:
            rec[f"dim_{key}"] = sum(key in ds for ds in dims)
        out.append(rec)

    return df, pd.DataFrame(out).sort_values("date").reset_index(drop=True)
