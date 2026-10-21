"""Dashboard for the factor study.

    streamlit run app.py          (from 3_ranking/)

Reads outputs/ (written by `python -m ranking.rank`) and handoff/. Read-only:
it never recomputes the model, so what it shows is exactly what the CSV says.
"""

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ranking.config import (DETECTIONS, FACTOR_DEFS, FACTOR_RANKING, SHIP_COUNTS,
                            STUDY_FEATURES, STUDY_RESULT)
from ranking.updater import STEPS, UpdateJob

st.set_page_config(page_title="Ship Factor Study", layout="wide")

# Palette (validated: ordinal ramp per mode; diverging blue <-> red poles).
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
OCC = ({"low": "#9ec5f4", "medium": "#3987e5", "high": "#184f95"} if DARK else
       {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"})
POS, NEG = ("#3987e5", "#e66767") if DARK else ("#2a78d6", "#e34948")
MUTED = "#898781"
LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
              legend=dict(orientation="h", y=1.08, x=0))


@st.cache_data
def load(version):   # `version` (file mtime) keys the cache so CLI re-runs show up
    ships = pd.read_csv(SHIP_COUNTS, parse_dates=["date"])
    rank = pd.read_csv(FACTOR_RANKING)
    feats = pd.read_csv(STUDY_FEATURES, parse_dates=["date"])
    result = json.loads(STUDY_RESULT.read_text(encoding="utf-8"))
    # Show only the news dimensions from 2_factors/factors.toml. Weather, season,
    # trend and tone stay in the model as controls but are not displayed.
    defs = (json.loads(FACTOR_DEFS.read_text(encoding="utf-8"))
            if FACTOR_DEFS.exists() else {})
    titles = {f"share_{k}": v["title"] for k, v in defs.get("dimensions", {}).items()}
    rank = rank[rank.factor.isin(titles)].copy()
    rank.insert(0, "dimension", rank.factor.map(titles))
    return ships, rank, feats, result, titles


@st.cache_resource
def update_job():
    return UpdateJob()      # one shared refresh, however many sessions are open


def _update_password():
    try:
        return st.secrets.get("UPDATE_PASSWORD")
    except Exception:       # no secrets file configured
        return None


def update_panel(job):
    """Sidebar: refresh news + weather, re-rank, reload. Hours when GDELT is slow."""
    updated = datetime.fromtimestamp(STUDY_RESULT.stat().st_mtime)
    st.caption(f"Data last updated {updated:%Y-%m-%d %H:%M}")
    if job.running:
        label = dict((k, t) for k, t, *_ in STEPS).get(job.step, "Starting")
        st.info(f"Running: {label}…")
        st.button("Stop fetching", on_click=job.stop_fetch,
                  help="Keeps what was fetched, then rebuilds and re-ranks.")
    else:
        locked = _update_password()
        allowed = not locked or st.text_input(
            "Update password", type="password") == locked
        st.button("Update data", on_click=job.start, disabled=not allowed,
                  help="Fetches missing news and weather, then re-ranks. "
                       "Only gaps are fetched; GDELT rate limits can make a full "
                       "run take hours.")
        if job.state == "done":
            st.success("Update finished.")
        elif job.state == "failed":
            st.error("Update failed - see the log.")
    if job.state != "idle":
        with st.expander("Log", expanded=job.running):
            st.code(job.tail() or "…", language=None)
    if job.generation != st.session_state.get("seen_gen"):
        st.session_state["seen_gen"] = job.generation
        load.clear()
        st.rerun()


job = update_job()
st.session_state.setdefault("seen_gen", job.generation)
if job.generation != st.session_state["seen_gen"]:
    load.clear()
    st.session_state["seen_gen"] = job.generation
with st.sidebar:
    st.subheader("Update data")
    # Poll only while a job is running; an idle page costs nothing.
    st.fragment(run_every=3 if job.running else None)(update_panel)(job)

ships, rank, feats, result, titles = load(STUDY_RESULT.stat().st_mtime)

st.title("Which factors move ship counts at San Pedro Bay?")
method = result.get("ranking", {}).get("method", result["gate"]["method"])
EXPLORATORY = method == "partial_spearman"
dropped = result.get("n_capture_dates", result["n_dates"]) - result["n_dates"]
st.caption(f"{result['n_dates']} Sentinel-2 capture dates with news"
           + (f" ({dropped} without news coverage left out)" if dropped else "")
           + " · 6 news dimensions · "
           + ("exploratory partial correlation, controlling for season and trend"
              if EXPLORATORY else "elastic net, controlling for weather and season"))
if EXPLORATORY:
    st.warning(f"**Exploratory view.** Only {result['n_dates']} dates have news, "
               "fewer than the 60 the ranking model needs. Each bar is that "
               "dimension's correlation with ship density on its own, so wide "
               "intervals are expected and two dimensions that rise together "
               "cannot be told apart. The full model takes over automatically "
               "once more news is fetched.")

# One score per method, so the chart and explorer read the same columns.
if EXPLORATORY:
    rank = rank.assign(score=rank.partial_rho, lo=rank.rho_ci_low, hi=rank.rho_ci_high)
    stable_label, stable = "CI excludes zero", rank[(rank.lo > 0) | (rank.hi < 0)]
    x_title = "Partial Spearman ρ (season + trend removed) · whiskers = bootstrap 95% CI"
else:
    rank = rank.assign(score=rank.get("enet_beta"), lo=rank.get("boot_ci_low"),
                       hi=rank.get("boot_ci_high"))
    stable_label = "Stable (≥80% selected)"
    stable = (rank[(rank.selection_rate >= 0.8) & (rank.enet_beta.abs() > 1e-6)]
              if not rank.empty else rank)
    x_title = "Elastic-net coefficient (standardised) · whiskers = bootstrap 95% CI"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Capture dates", result["n_dates"])
c2.metric("Dimensions ranked", len(rank))
c3.metric(stable_label, len(stable))
c4.metric("Significant after BH correction", int(rank.bh_survives.sum()) if not rank.empty else 0)

tab_rank, tab_ships, tab_explore, tab_img = st.tabs(
    ["Factor ranking", "Ship counts over time", "Factor explorer", "Detection images"])

NO_NEWS = (f"No news dimensions in this run: fewer than 10 capture dates have "
           "news coverage. Press **Update data** in the sidebar to fetch the "
           "missing weeks.")

# --- 1. Ranking: diverging bars, CI whiskers, significance in the label ------
with tab_rank:
    if rank.empty:
        st.info(NO_NEWS)
    else:
        r = rank.sort_values("score", key=abs)
        labels = [f"{f} ★" if s else f for f, s in zip(r.dimension, r.bh_survives)]
        fig = go.Figure(go.Bar(
            x=r.score, y=labels, orientation="h",
            marker=dict(color=[POS if b >= 0 else NEG for b in r.score],
                        cornerradius=4),
            error_x=dict(type="data", symmetric=False, color=MUTED, thickness=1.5,
                         array=r.hi - r.score, arrayminus=r.score - r.lo),
            customdata=r[["lo", "hi", "spearman_p_adj"]],
            hovertemplate="<b>%{y}</b><br>%{x:+.3f} [%{customdata[0]:+.3f}, "
                          "%{customdata[1]:+.3f}]<br>p adj %{customdata[2]:.3f}"
                          "<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        fig.update_layout(**LAYOUT, height=60 + 40 * len(r), bargap=0.35,
                          xaxis_title=x_title)
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue: more of this news, more ships. Red: fewer. ★ = survives "
                   "Benjamini-Hochberg correction. "
                   + ("A bar whose whisker crosses zero is not distinguishable from noise."
                      if EXPLORATORY else "Read selection rate before size."))
        with st.expander("Table view"):
            st.dataframe(rank.drop(columns=["score", "lo", "hi"]), hide_index=True,
                         width="stretch")

# --- 2. Ship counts: one line, markers coloured by occupancy (ordinal) -------
with tab_ships:
    metric = st.radio("Measure", ["n_ships", "density"], horizontal=True,
                      format_func={"n_ships": "Ships", "density": "Ships per km²"}.get)
    fig = go.Figure(go.Scatter(x=ships.date, y=ships[metric], mode="lines",
                               line=dict(color=MUTED, width=2), hoverinfo="skip",
                               showlegend=False))
    for occ, color in OCC.items():
        s = ships[ships.occupancy == occ]
        fig.add_trace(go.Scatter(
            x=s.date, y=s[metric], mode="markers", name=occ,
            marker=dict(color=color, size=9, line=dict(width=2, color="rgba(0,0,0,0)")),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y}<extra>" + occ + "</extra>"))
    fig.add_vrect(x0="2020-09-01", x1="2022-06-30", fillcolor=MUTED, opacity=0.12,
                  line_width=0, annotation_text="COVID backlog",
                  annotation_position="top left")
    fig.update_layout(**LAYOUT, height=420, yaxis_title=metric)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Table view"):
        st.dataframe(ships, hide_index=True, width="stretch")

# --- 3. Explorer: one factor vs density (scatter), no dual axis --------------
with tab_explore:
    if rank.empty:
        st.info(NO_NEWS)
    else:
        factor = st.selectbox("Dimension", rank.factor.tolist(), format_func=titles.get)
        row = rank.set_index("factor").loc[factor]
        if EXPLORATORY:
            st.caption(f"Partial Spearman ρ {row.score:+.2f} [{row.lo:+.2f}, {row.hi:+.2f}] "
                       f"· p (BH-adj) {row.spearman_p_adj:.3f} · the scatter shows raw "
                       "values, before season and trend are removed")
        else:
            st.caption(f"Spearman ρ {row.spearman_rho:+.2f} · p (BH-adj) "
                       f"{row.spearman_p_adj:.3f} · selected in {row.selection_rate:.0%} "
                       "of bootstrap resamples")
        df = feats.merge(ships[["date", "occupancy"]], on="date", how="left")
        fig = px.scatter(df, x=factor, y="density", color="occupancy",
                         color_discrete_map=OCC, hover_data={"date": "|%Y-%m-%d"},
                         category_orders={"occupancy": list(OCC)},
                         labels={factor: f"{titles[factor]} - share of relevant "
                                         "headlines in the week before"})
        fig.update_traces(marker=dict(size=9))
        fig.update_layout(**LAYOUT, height=420, yaxis_title="Ships per km²")
        st.plotly_chart(fig, width="stretch")

# --- 4. Detection images ----------------------------------------------------
with tab_img:
    imgs = sorted(DETECTIONS.glob("*.jpg"))
    st.caption("Busiest and quietest dates, as annotated by the detector.")
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        d, n, occ = p.stem.split("_")
        cols[i % 2].image(str(p), caption=f"{d} · {n.replace('ships', '')} ships · {occ}",
                          width="stretch")
