"""Component 3 Stage A - article title -> 1 of 13 risk categories.

Taxonomy and structure follow 3_ranking/plan-topics.md: the 10 NetSuite supply
chain risk categories from settings/requirements.md, plus Health/Pandemic and
Conflict/War (decision Q8), plus an explicit `irrelevant` escape hatch (Q11) so
port-authority appointments and cruise-terminal puff pieces do not get forced
into an operational bucket.

WHICH RUNG OF THE LADDER THIS IS
--------------------------------
The plan lists three label sources (Q7). This module implements option 2,
**keyword rules** - free, fully reproducible, noisier than the LLM labelling
the plan recommends. It is the baseline the plan says to build first, not the
finished Stage A classifier: the TF-IDF and transformer rungs above it are
still unbuilt, and so is the 300-title gold set the plan calls non-negotiable.
The honest reading of any ranking downstream is therefore "ranked using
rule-assigned topics", and category precision is the limiting factor, not the
impact model.

Scoring is deliberately not a bag of independent keyword hits:

  * cues are weighted - "ilwu" is decisive for Labor, "workers" is not;
  * phrases beat single words, so "empty container" does not also fire
    Material Shortages via the bare word "container" in every logistics story;
  * a minimum score is required before any operational label is assigned,
    which is what routes genuinely off-topic articles to `irrelevant` rather
    than to whichever category happened to have the loosest keyword.
"""

import re

# The 12 operational categories, in the plan's order, plus the escape hatch.
LABELS = (
    "inflation_econ",
    "material_shortages",
    "natural_disasters",
    "global_regulations",
    "cybersecurity",
    "logistics_reliability",
    "labor_shortages",
    "demand_volatility",
    "operational_risks",
    "reputation_risks",
    "health_pandemic",
    "conflict_war",
    "irrelevant",
)

RISK_LABELS = LABELS[:-1]

LABEL_TITLES = {
    "inflation_econ": "Inflation & economic concerns",
    "material_shortages": "Material shortages",
    "natural_disasters": "Natural disasters",
    "global_regulations": "Global regulations",
    "cybersecurity": "Cybersecurity threats",
    "logistics_reliability": "Logistics reliability",
    "labor_shortages": "Labor shortages",
    "demand_volatility": "Demand volatility",
    "operational_risks": "Operational risks",
    "reputation_risks": "Reputation risks",
    "health_pandemic": "Health / pandemic",
    "conflict_war": "Conflict / war",
    "irrelevant": "Irrelevant",
}

# weight 3 = decisive on its own, 2 = strong, 1 = supporting evidence only.
CUES = {
    "inflation_econ": [
        (3, r"\bfreight rate|\bspot rate|\bcharter rate|\bbunker (?:price|cost)"),
        (2, r"\binflation|\brecession|\bfuel (?:price|cost|surcharge)|\bdiesel price"),
        (2, r"\beconomic downturn|\bconsumer price|\bcost of living|\binterest rate"),
        (1, r"\beconomy|\bcosts? (?:rise|soar|surge|climb)|\bprofit|\bearnings"),
    ],
    "material_shortages": [
        (3, r"\bchassis shortage|\bcontainer shortage|\bequipment shortage"),
        (3, r"\bempty container|\bempties\b|\bbox shortage"),
        (2, r"\bshortage of|\bsemiconductor|\bchip shortage|\braw material"),
        (1, r"\bshortage|\bscarcity"),
    ],
    "natural_disasters": [
        (3, r"\bhurricane|\btyphoon|\bearthquake|\btsunami|\bwildfire\b"),
        (3, r"\bdense fog|\bfog (?:closure|delay)|\bstorm (?:closes|shuts|halts)"),
        (2, r"\bflood|\bstorm\b|\bhigh winds?\b|\bgale|\bheavy rain|\bdrought"),
        (1, r"\bweather (?:delay|disrupt|close)|\bsevere weather"),
    ],
    "global_regulations": [
        (3, r"\btariff|\bsanction|\bcustoms rule|\bimport dut|\btrade war"),
        (3, r"\bemissions? (?:rule|regulation|standard)|\bimo 2020|\bcarbon tax"),
        (2, r"\bregulat|\blegislation|\bpolicy change|\bexecutive order|\bban on"),
        (1, r"\bcompliance|\bmandate|\bfederal maritime"),
    ],
    "cybersecurity": [
        (3, r"\bransomware|\bcyber ?attack|\bcybersecurity|\bdata breach"),
        (2, r"\bhackers?\b|\bhacked|\bmalware|\bphishing|\bit outage"),
        (1, r"\bcyber\b|\bsecurity breach"),
    ],
    "logistics_reliability": [
        (3, r"\bcongestion|\bbacklog|\bbottleneck|\bdwell time|\bberth (?:delay|wait)"),
        (3, r"\bblank sailing|\bschedule reliability|\bat anchor\b|\bqueue of ships"),
        (2, r"\bdelay|\bgridlock|\bpile ?up|\bturn time|\bwait time|\blogjam"),
        (2, r"\bsupply chain (?:crisis|disruption|snarl|mess|woes)"),
        (1, r"\bthroughput|\bvessel traffic|\bport traffic|\bqueue"),
    ],
    "labor_shortages": [
        (3, r"\bilwu\b|\bstrike\b|\bwalkout|\bwork stoppage|\bpicket|\block ?out"),
        (3, r"\blabou?r (?:dispute|action|contract|negotiation|shortage|unrest)"),
        (2, r"\bunion\b|\bdockworker|\blongshore|\bstaffing|\bworker shortage"),
        (2, r"\bdriver shortage|\btrucker (?:shortage|protest)"),
        (1, r"\bworkers?\b|\bhiring|\bwages?\b"),
    ],
    "demand_volatility": [
        (3, r"\bimport surge|\bcargo surge|\bpeak season|\brecord (?:imports|volume|cargo)"),
        (3, r"\border cancellation|\bdemand (?:slump|collapse|drop|surge)"),
        (2, r"\bvolumes? (?:rise|fall|drop|plunge|jump|surge|slide)|\brestocking|\bdestocking"),
        (1, r"\bdemand\b|\bconsumer spending|\bretail sales"),
    ],
    "operational_risks": [
        (3, r"\bcrane (?:failure|collapse|accident)|\bterminal (?:outage|shutdown|closure)"),
        (3, r"\ballision|\bgrounding|\bcapsiz|\bcollision|\bvessel fire|\boil spill"),
        (2, r"\bbreakdown|\bmalfunction|\bequipment failure|\bpower outage|\bderail"),
        (2, r"\baccident|\bexplosion|\bcontainers? lost|\bran aground"),
        (1, r"\bmaintenance|\brepair|\bdredging|\bincident"),
    ],
    "reputation_risks": [
        (3, r"\bboycott|\besg\b|\bgreenwash|\bforced labou?r|\bhuman rights"),
        (2, r"\blawsuit|\bprotest against|\bscandal|\bpublic backlash|\bfined for"),
        (1, r"\bcriticism|\breputation|\bcommunity (?:opposition|concern)|\bpollution"),
    ],
    "health_pandemic": [
        (3, r"\bcovid|\bcoronavirus|\bpandemic|\bquarantine|\bomicron|\bdelta variant"),
        (3, r"\bcrew change crisis|\bcrew (?:stranded|change)"),
        (2, r"\block ?down|\boutbreak|\bvaccine|\bvaccination|\bepidemic|\binfection"),
        (1, r"\bhealth (?:order|protocol)|\bmask mandate"),
    ],
    "conflict_war": [
        (3, r"\bred sea|\bhouthi|\bblockade|\bwar risk|\bwar ?zone|\bmissile"),
        (2, r"\bwar\b|\binvasion|\bconflict|\bmilitary|\bpiracy|\bhijack"),
        (2, r"\bukraine|\brussia|\bgaza|\bisrael|\bsuez"),
        (1, r"\bgeopolitic|\btension"),
    ],
}

COMPILED = {
    label: [(w, re.compile(pat)) for w, pat in cues] for label, cues in CUES.items()
}

# Below this, nothing operational was actually said -> irrelevant.
MIN_SCORE = 2.0


def score_title(title):
    """Return {label: score} over the 12 operational categories."""
    text = " " + re.sub(r"[^a-z0-9 ]+", " ", title.lower()) + " "
    text = " ".join(text.split())
    scores = {}
    for label, cues in COMPILED.items():
        total = 0.0
        for weight, pattern in cues:
            if pattern.search(text):
                total += weight
        scores[label] = total
    return scores


def classify(title):
    """Single dominant label per article (decision Q6), plus the runner-up.

    The plan asks for the runner-up to be stored even though nothing consumes
    it today - it is free, and it is what makes the single-label decision
    revisitable without re-labelling.
    """
    scores = score_title(title)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_label, top_score = ranked[0]
    if top_score < MIN_SCORE:
        return {"label": "irrelevant", "score": float(top_score),
                "runner_up": None, "runner_up_score": 0.0}
    runner_up, runner_score = ranked[1]
    return {
        "label": top_label,
        "score": float(top_score),
        "runner_up": runner_up if runner_score > 0 else None,
        "runner_up_score": float(runner_score),
    }


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


def daily_scores(articles):
    """Classify every headline and count them per day and category.

    `articles` is the row list from `factors.news.collect`. Returns
    (labelled_articles_df, daily_df) with one daily row per date:

        n_articles, n_relevant         totals for the day
        count_<topic>                  relevant articles with that dominant topic
        tone_sum_relevant              summed title tone over relevant articles

    Counts and sums rather than shares and means, because they aggregate
    exactly: a 7-day share is sum(count) / sum(n_relevant) over the window,
    which 3_ranking computes. Articles labelled `irrelevant` stay out of
    n_relevant - they are the escape hatch, not a thirteenth risk.
    """
    import pandas as pd

    rows = []
    for a in articles:
        c = classify(a["title"])
        rows.append({
            **a,
            "label": c["label"],
            "label_score": c["score"],
            "runner_up": c["runner_up"],
            "tone": title_tone(a["title"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    out = []
    for day, grp in df.groupby("date"):
        relevant = grp[grp["label"] != "irrelevant"]
        rec = {
            "date": day,
            "n_articles": len(grp),
            "n_relevant": len(relevant),
            "tone_sum_relevant": float(relevant["tone"].sum()),
        }
        for label in RISK_LABELS:
            rec[f"count_{label}"] = int((relevant["label"] == label).sum())
        out.append(rec)

    return df, pd.DataFrame(out).sort_values("date").reset_index(drop=True)
