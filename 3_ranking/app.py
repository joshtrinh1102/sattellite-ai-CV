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

from ranking.config import (DETECTIONS, FACTOR_RANKING, SHIP_COUNTS,
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
def load():
    ships = pd.read_csv(SHIP_COUNTS, parse_dates=["date"])
    rank = pd.read_csv(FACTOR_RANKING)
    feats = pd.read_csv(STUDY_FEATURES, parse_dates=["date"])
    result = json.loads(STUDY_RESULT.read_text(encoding="utf-8"))
    return ships, rank, feats, result


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

ships, rank, feats, result = load()

st.title("Which factors move ship counts at San Pedro Bay?")
st.caption(f"{result['n_dates']} Sentinel-2 capture dates · method: "
           f"{result['gate']['method']} · "
           + ("news + weather + season factors" if result["has_news_factors"]
              else "weather + season factors only (news coverage incomplete)"))

stable = rank[(rank.selection_rate >= 0.8) & (rank.enet_beta.abs() > 1e-6)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Capture dates", result["n_dates"])
c2.metric("Factors ranked", len(rank))
c3.metric("Stable (≥80% selected)", len(stable))
c4.metric("Significant after BH correction", int(rank.bh_survives.sum()))

tab_rank, tab_ships, tab_explore, tab_img = st.tabs(
    ["Factor ranking", "Ship counts over time", "Factor explorer", "Detection images"])

# --- 1. Ranking: diverging bars, CI whiskers, significance in the label ------
with tab_rank:
    r = rank.sort_values("enet_beta", key=abs)
    labels = [f"{f} ★" if s else f for f, s in zip(r.factor, r.bh_survives)]
    fig = go.Figure(go.Bar(
        x=r.enet_beta, y=labels, orientation="h",
        marker=dict(color=[POS if b >= 0 else NEG for b in r.enet_beta],
                    cornerradius=4),
        error_x=dict(type="data", symmetric=False, color=MUTED, thickness=1.5,
                     array=r.boot_ci_high - r.enet_beta,
                     arrayminus=r.enet_beta - r.boot_ci_low),
        customdata=r[["selection_rate", "spearman_rho", "spearman_p_adj"]],
        hovertemplate="<b>%{y}</b><br>coef %{x:+.4f}<br>selected %{customdata[0]:.0%}"
                      "<br>Spearman ρ %{customdata[1]:+.2f} (p adj %{customdata[2]:.3f})"
                      "<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(**LAYOUT, height=60 + 32 * len(r), bargap=0.35,
                      xaxis_title="Elastic-net coefficient (standardised) · whiskers = bootstrap 95% CI")
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue raises ship density, red lowers it. ★ = survives "
               "Benjamini-Hochberg correction. Read selection rate before size.")
    with st.expander("Table view"):
        st.dataframe(rank, hide_index=True, width="stretch")

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
    factor = st.selectbox("Factor", rank.factor.tolist())
    row = rank.set_index("factor").loc[factor]
    st.caption(f"Spearman ρ {row.spearman_rho:+.2f} · p (BH-adj) "
               f"{row.spearman_p_adj:.3f} · selected in {row.selection_rate:.0%} "
               "of bootstrap resamples")
    df = feats.merge(ships[["date", "occupancy"]], on="date", how="left")
    fig = px.scatter(df, x=factor, y="density", color="occupancy",
                     color_discrete_map=OCC, hover_data={"date": "|%Y-%m-%d"},
                     category_orders={"occupancy": list(OCC)})
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
