# Component 3 — Topic classifier and impact model

**Status:** Stage A implemented as the *keyword-rules* rung only
(`2_factors/factors/topics.py`); Stage B implemented in full (`3_ranking/ranking/analysis.py`,
`3_ranking/ranking/rank.py`). Q7 is answered for now by option 2, and the gold set is
still missing. See "Implementation notes".

Requirement, verbatim:

> A multi-class model that ranks the top 10 topics and their relevance to the
> supply chain.

## Decisions taken (2026-09-22)

| # | Decision | Consequence |
|---|----------|-------------|
| Q6 | **Single-label: one of 12 categories per article.** | Softmax, not sigmoids. See "Why single-label holds up". |
| Q8 | **12 labels** — the 10 NetSuite categories plus Health/Pandemic and Conflict/War. | COVID and war get their own buckets instead of being scattered. |
| Q9 | **Fixed taxonomy.** No unsupervised topic discovery. | The model ranks within a known factor list, which is what the research question needs. |
| — | **Input is the article title only.** | Follows from the GDELT decision in Component 2. |
| — | **Target variable is ships detected.** | Occupancy density from Component 1. |

## Two models, not one

Confirming the structure: **yes, this needs two separate models**, and they are
separate for a reason that is not just convenience.

```
  Stage A — Topic classifier              Stage B — Impact model
  ------------------------------          ------------------------------
  input:   article title (text)           input:  per-date topic shares,
  output:  1 of 12 risk categories                tone, weather, season
  data:    labelled titles (thousands)    output: predicted ship density
  metric:  macro-F1 on a gold set         data:   dates with imagery (tens)
                                          metric: effect size + CI
```

They cannot be one model because **they train on different things**. Stage A
learns from text and needs thousands of labelled titles, which are cheap.
Stage B learns from satellite observations and has only as many rows as there
are collected image dates — tens, not thousands. Fusing them would force the
whole system down to the smaller sample size and throw away the text data.

Keeping them separate also means Stage A can be built, evaluated and finished
**now**, while imagery collection for Stage B is still in progress.

### Why single-label holds up here

Multi-label is the more faithful representation — a strike article really is
Labor Shortages *and* Logistics Reliability at once. But given the sample-size
constraint on Stage B, single-label is the better engineering call, for a
concrete reason: with a few dozen image dates there is no statistical power to
distinguish the effect of *Labor Shortages* from the effect of *Logistics
Reliability* when the two co-occur on most of the same articles. Multi-label
would add correlated features the impact model cannot separate.

Assign each article its **dominant** risk category. Record the runner-up
category and its probability alongside — free to store, and it allows
revisiting the decision later without re-labelling.

## Label taxonomy

| # | Label | Title cues |
|---|-------|-----------|
| 1 | Inflation & economic concerns | fuel cost, freight rates, recession |
| 2 | Material shortages | chassis, container, empty equipment |
| 3 | Natural disasters | storm, typhoon, flood, earthquake, fog closure |
| 4 | Global regulations | tariff, sanction, emissions rule, customs |
| 5 | Cybersecurity threats | ransomware on terminal operating systems |
| 6 | Logistics reliability | berth delay, dwell time, blank sailing, congestion |
| 7 | Labor shortages | ILWU, strike, walkout, staffing |
| 8 | Demand volatility | import surge, peak season, order cancellations |
| 9 | Operational risks | crane failure, terminal outage, accident |
| 10 | Reputation risks | boycott, sanction avoidance, ESG pressure |
| 11 | Health / pandemic | COVID, quarantine, crew change crisis |
| 12 | Conflict / war | Red Sea, blockade, war-risk premium |

Plus an implicit 13th outcome, **`irrelevant`**. Port queries return plenty of
articles that are genuinely about nothing operational — a port authority
executive appointment, a cruise terminal opening. Without an explicit escape
hatch the classifier is forced to assign one of the 12 to every one of them,
injecting noise straight into the impact model. Include it.

Every label definition goes in `docs/label-guide.md` with three positive and
three near-miss negative examples, written **before** labelling starts.
Without it, labels drift between annotation sessions and the resulting F1 is
unreproducible.

## Labels: what "use the titles" does and does not settle

Titles are the classifier's **input** — settled by the GDELT decision. But
something still has to decide which of the 12 categories each title belongs to,
and a title does not carry its own label. A label source is still needed:

1. **LLM labelling of titles (recommended).** Titles are short — roughly 15
   tokens each — so labelling 10,000 of them is a genuinely small API spend,
   far cheaper than the full-article version originally planned. Claude with
   the label guide in the prompt, structured output, three samples per title
   with majority vote to surface ambiguity.
2. **Keyword rules.** Zero cost. The cue column above is most of the work
   already. Noisier, needs hand-tuning per category, but fully reproducible.
3. **Hand-label everything.** Highest quality, unrealistic at 12 categories.

Whichever is chosen, **300 hand-labelled titles are non-negotiable** as a gold
set — stratified so each label has ≥20 positives, never trained on, used for
every reported number. Without it there is no way to know whether the labels
are any good, and a model trained on bad labels still produces confident,
plausible, wrong rankings.

**Gate:** score the label source against the gold set before training anything.
If macro-F1 is below ~0.70, fix the labelling before proceeding.

## Stage A — model ladder

1. **TF-IDF + logistic regression (multinomial).** Hours to build. Sets the
   number everything else must beat. On short keyword-dense text like headlines
   this is often within a few points of a transformer — worth knowing before
   spending a week fine-tuning one.
2. **Fine-tuned DistilBERT / DeBERTa-v3-small**, 13-way softmax, cross-entropy
   with class weights. Real gains expected on titles where the category depends
   on phrasing rather than keywords.
3. **LLM few-shot** as a comparison point only — it is the thing being
   distilled in step 2.

### Evaluation

Reported on the gold set only:

- **Per-class precision / recall / F1**, macro-F1 as the headline. Plain
  accuracy is misleading — `irrelevant` will dominate the class distribution.
- **Full 13×13 confusion matrix.** Expect the main error mode between Logistics
  Reliability and Operational Risks, which overlap heavily.
- **Calibration.** Daily topic scores aggregate predicted probabilities, so
  overconfident scores distort the ranking even when the argmax is right.

**Split by time, never randomly.** A random split puts syndicated copies of the
same story on both sides and inflates every metric — a real risk when titles
are near-identical across outlets.

## Stage A output — daily topic scores

Per port, per day, for each of the 12 categories:

```
raw_count(topic, day) = deduplicated articles with that dominant topic
share(topic, day)     = raw_count / total_relevant_articles(day)
tone(topic, day)      = mean GDELT V2Tone over those articles
```

`share` rather than `raw_count` is what enters the impact model — the
normalisation that stops the score tracking overall news volume rather than
port conditions. Smooth over a 7-day window; single-day spikes are usually one
wire story.

Output: `data/risk_daily.parquet`, one row per (port, date).

## Stage B — impact model

This is the stage that answers the project's headline question, and the one
constrained by imagery. Read the sample-size section of the news plan first —
it determines which version below is honest to run.

**Target:** ship density (ships per km² of water) on each date with imagery.

**Features**, per date, from the 7-day window preceding the capture:

- `share` for each topic category
- mean tone overall, and mean tone within the dominant topics
- weather on the capture date (Open-Meteo): wind, visibility, precipitation
- seasonality: month, and a Lunar New Year proximity flag
- a linear time trend

### Scale the method to the sample size

| Dates | Honest method | What can be claimed |
|-------|--------------|---------------------|
| ~10 | Two-group comparison: COVID vs normal, with a CI | "Ship density was X% lower during COVID (CI a–b)." Nothing about individual factors. |
| ~30 | Pre-registered test of 2–3 factor hypotheses chosen **before** looking | "Of the three factors tested, only labour actions showed a detectable association." |
| ~60 | Elastic-net regression over lagged topic shares | "Of 12 candidate factors, N survive regularisation; ranked by effect size." |

Running the 60-date method on 10 dates will still produce a ranked list of 12
factors with coefficients. That list would be fabrication dressed as a result —
the model has more parameters than observations, so the ranking is determined
by noise. This table exists to prevent that.

### Analysis notes that apply at every sample size

- **Lags matter.** A strike affects berth occupancy days later, not the same
  afternoon. Test the preceding 7-, 14- and 30-day windows. Given the sample
  size, pick the lag structure **in advance** rather than scanning all of them
  and reporting the best — scanning is how noise becomes a finding.
- **Control for seasonality before attributing anything.** Lunar New Year and
  peak season move port traffic hard and correlate with news volume. An
  uncontrolled model credits whatever topic happens to spike in February.
- **Correct for multiple comparisons** (Benjamini-Hochberg) whenever more than
  a couple of hypotheses are tested.
- **Report effect sizes with confidence intervals.** "Labour actions are
  associated with a 14% (CI 6–22%) drop in occupancy" is a result; "p = 0.03"
  is not.
- **Detector error propagates.** Ship counts come from a model with its own
  false positives and negatives. Report the detector's precision/recall
  alongside the occupancy numbers so the uncertainty is visible rather than
  silently absorbed into the correlation.
- **This is correlational.** With one port and a handful of disruption
  episodes, the honest claim is association, not causation. State it.

## Milestones

**Stage A — can start immediately, does not wait on imagery**

1. `docs/label-guide.md` with examples per label, including `irrelevant`.
2. Gold set: 300 hand-labelled titles.
3. Label source built and scored against gold. **Gate: macro-F1 ≥ 0.70.**
4. Silver set generated over the backfilled titles.
5. TF-IDF baseline plus evaluation harness.
6. Transformer fine-tune, compared against the baseline on the same gold set.
7. `risk_daily.parquet` produced.

**Stage B — gated on imagery collection**

8. Occupancy table built from the collected image dates.
9. Method selected from the sample-size table, based on the actual date count.
10. Impact analysis and the final factor ranking, with effect sizes and CIs.

## Open questions

- **Q7 — Label source.** LLM labelling of titles (recommended, small cost
  because titles are short), or keyword rules (free, noisier)? Either way the
  300-title gold set is required.
- **Q11 — Does `irrelevant` get included as a 13th class?** Recommended yes;
  confirm.


---

## Implementation notes (added when the modules were built)

### Stage A is on the bottom rung, and that bounds everything

`2_factors/factors/topics.py` implements **option 2 of Q7, keyword rules** — the 13-class
taxonomy above, with weighted cues so that `ilwu` is decisive and `workers` is
only supporting evidence, phrase cues beating single words, and a minimum score
routing genuinely off-topic articles to `irrelevant`.

What is **not** built, and what the plan says is non-negotiable:

- the **300-title gold set**, so category precision is unmeasured;
- the **macro-F1 ≥ 0.70 gate**, which therefore has not been applied;
- the TF-IDF baseline and the transformer above it.

Every downstream number should be read as "ranked using rule-assigned topics".
Category precision, not the impact model, is the limiting factor.

Tone is computed here too, from a small disruption-oriented lexicon over the
titles, since the GDELT probe confirmed no V2Tone is returned.

### Stage B: the sample-size table is enforced in code

`analysis.choose_method` reads the number of usable dates and returns the
method this plan's table allows, and `3_ranking/ranking/rank.py` runs only that method.
The 60-date row was reached: the run behind `results/results.md` used **76 dates**,
so the elastic net is the honest choice rather than an aspiration.

Implemented as specified: elastic net over standardised features, lambda by
**blocked** time-series CV (a shuffled fold puts neighbouring captures on both
sides and picks too small a penalty), bootstrap resampling for a selection rate
per factor, Spearman with Benjamini-Hochberg across the factor set, seasonality
and a linear time trend as controls, and effect sizes with intervals rather
than bare p-values.

`explain_date` was added beyond the plan: it decomposes a **single** date's
prediction into `coef x standardised value` per factor. It answers "which
factors drove the count on this date" rather than "on average", and its
residual is the honest signal when a date is driven by something outside the
feature set — as 2021-10-10 is.

Everything is numpy and scipy. scikit-learn and statsmodels are installed but
their compiled extensions are blocked by this machine's Application Control
policy, so ridge, the elastic net, the bootstrap and the AUC are implemented
directly.

### Still open

- **Q7** remains open in substance: the LLM labelling the plan recommends is
  not built, and the keyword rules are the stopgap it warned would be noisier.
- **Q11** is answered: `irrelevant` is included as the 13th class.
- **Lag structure** is fixed at the pre-specified 7-day window. The 14- and
  30-day variants in the plan have not been tested, deliberately — scanning
  lags and reporting the best is how noise becomes a finding.
