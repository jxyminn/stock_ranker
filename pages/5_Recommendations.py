"""
Recommendations: the one page in this app that explicitly says "these
stocks look most interesting right now" instead of just describing what
changed. Reuses the exact same selection logic as the Ranker page's "Best
Picks" tab (dual-horizon confirmation for short-term, consistency across
every horizon for long-term), computed automatically every night against
the whole watchlist by nightly_briefing.py -- see compute_recommendations()
there.

Reasoning per pick is computed fresh when this page loads (not baked into
the nightly JSON): SHAP-based drivers (same as Stock Research's "Why the
model says this"), a plain-English technical read, fundamentals context,
and the model's own documented backtest reliability for that horizon --
so the "why" is grounded in what the model actually weighted, not a
generic blurb, and is honest about how (un)reliable that horizon has been
historically.

This is a higher-stakes page than the rest of the app precisely because it
IS opinionated -- see the disclaimer, which is deliberately repeated at
both the top and bottom rather than buried once at the end.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIEFING_DIR
from data_fetcher import fetch_fundamentals
from explain import explain_prediction
from features import FEATURE_COLUMNS
from insights import generate_why_bullets
from predict import latest_features_for_ticker
from reliability import load_backtest_results, reliability_ranking
from stock_info import get_models

st.set_page_config(
    page_title="Recommendations · Stock Research",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.main-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 28px 36px; margin-bottom: 20px;
}
.main-header h1 { font-size: 2.1rem; font-weight: 700; color: #58a6ff; margin: 0; }
.main-header p  { color: #8b949e; margin: 6px 0 0; font-size: 0.95rem; }

.disclaimer-box {
    background: #2d0b0b; border: 2px solid #f85149; border-radius: 10px;
    padding: 18px 22px; color: #ffb3ad; font-size: 0.92rem; line-height: 1.6;
    margin-bottom: 20px;
}
.disclaimer-box b { color: #ff8a80; }

.info-box {
    background: #0c2d45; border-left: 4px solid #58a6ff;
    border-radius: 6px; padding: 12px 16px; color: #a5c8f0;
    font-size: 0.88rem; margin-bottom: 16px;
}
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.metric-card .label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; margin-top: 4px; }
.metric-card .sub   { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }

.pick-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 22px; margin-bottom: 16px;
}
.pick-rank { font-size: 1.3rem; margin-right: 8px; }
.pick-ticker { font-size: 1.4rem; font-weight: 800; color: #e6edf3; }
.pick-name { color: #8b949e; font-size: .9rem; margin-left: 8px; }
.pick-price { color: #8b949e; font-size: .85rem; margin-top: 2px; }
.section-heading {
    font-size: 1.2rem; font-weight: 700; color: #e6edf3;
    margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

HORIZON_LABELS = {5: "1 week", 10: "2 weeks", 15: "3 weeks", 30: "1 month",
                   50: "~2.5 months", 200: "~9.5 months"}
MEDALS = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

DISCLAIMER_HTML = """<div class="disclaimer-box">
    ⚠️ <b>Read this before anything else on this page.</b><br><br>
    These are <b>model-generated candidates, not investment advice</b>. They come from a
    statistical model trained on historical price patterns -- the same model whose honest,
    documented track record you can see on the Daily Briefing page's "Historical Backtest"
    and "Live Track Record" tabs. That track record shows real, measured uncertainty, not a
    confident expert's judgment.<br><br>
    <b>The model can be, and will sometimes be, wrong.</b> A stock appearing here is not a
    guarantee it will go up, and a stock NOT appearing here is not a sign to avoid it. Treat
    this as one input to think about, not a conclusion to act on. Always do your own research
    and consider talking to a financial advisor before making investment decisions.
</div>"""


def load_briefing() -> dict | None:
    path = BRIEFING_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def render_pick_reasoning(ticker: str, horizons: list[int], models) -> None:
    """Fresh (not cached from last night) fundamentals + technical read +
    SHAP drivers for the horizon(s) this pick was selected on, plus that
    horizon's own documented backtest reliability."""
    feat = latest_features_for_ticker(ticker)
    if feat is None:
        st.caption("Couldn't load fresh data for this ticker right now.")
        return
    fund = fetch_fundamentals(ticker)
    X = feat[FEATURE_COLUMNS].astype(float).to_frame().T

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Technical read**")
        for bullet in generate_why_bullets(feat, 0.0, 0.5):
            st.markdown(f"- {bullet}")

        st.markdown("**Fundamentals**")
        if fund:
            st.markdown(f"Sector: {fund.get('sector', 'n/a')} / {fund.get('industry', 'n/a')}")
            target = fund.get("target_mean_price")
            if target:
                upside = (target / feat["_last_close"] - 1) * 100
                st.markdown(f"Analyst mean target: ${target:.2f} ({upside:+.1f}% vs. current)")
        else:
            st.markdown("_Unavailable right now._")

    with col2:
        st.markdown("**Why the model picked this (top drivers)**")
        if models is None:
            st.caption("Models unavailable.")
        else:
            h = horizons[-1]
            reg = models[h]["regressor"]
            for d in explain_prediction(reg, X, top_n=4):
                color = "#3fb950" if d["direction"] == "up" else "#f85149"
                st.markdown(
                    f"**{d['label']}** (value {d['value']:+.3f}) pushed the {HORIZON_LABELS[h]} "
                    f"forecast <span style='color:{color}'>{d['direction']} "
                    f"{abs(d['contribution_pp']):.2f}pp</span>", unsafe_allow_html=True,
                )

        st.markdown("**How reliable has this horizon actually been?**")
        ranked = reliability_ranking(load_backtest_results())
        if ranked:
            stat = next((r for r in ranked if r["horizon"] == horizons[-1]), None)
            if stat:
                st.markdown(f"Historical backtest for {HORIZON_LABELS[horizons[-1]]}: "
                            f"mean IC **{stat['mean_ic']:+.3f}**, AUC **{stat['mean_auc']:.3f}** "
                            f"(0.5 = coin flip). See the Daily Briefing page for full context.")
            else:
                st.caption("No backtest data for this horizon yet.")
        else:
            st.caption("No backtest on record yet.")


st.markdown("""<div class="main-header">
    <h1>🎯 Recommendations</h1>
    <p>Model-generated candidates from across your entire watchlist, recomputed every night.
    Not a buy list -- read the disclaimer below before anything else.</p>
</div>""", unsafe_allow_html=True)

st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

briefing = load_briefing()
if briefing is None:
    st.markdown("""<div class="info-box">
        No briefing has been generated yet. It runs automatically every night via GitHub
        Actions, or run <code>python nightly_briefing.py</code> locally to generate one now.
    </div>""", unsafe_allow_html=True)
    st.stop()

rec = briefing.get("recommendations") or {}
short_term = rec.get("short_term", [])
long_term = rec.get("long_term", [])

if not short_term and not long_term:
    st.info("Not enough agreement across the watchlist to surface any candidates tonight -- "
            "or models weren't available when this briefing ran. Try again tomorrow.")
    st.stop()

generated_at = datetime.fromisoformat(briefing["generated_at"])
age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="metric-card">
    <div class="label">Computed</div>
    <div class="value" style="font-size:1.1rem">{generated_at.strftime('%b %d, %Y %H:%M UTC')}</div>
    <div class="sub">{age_hours:.0f} hours ago</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card">
    <div class="label">Watchlist Considered</div>
    <div class="value">{rec.get('universe_size', '—')}</div>
    <div class="sub">stocks screened</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card">
    <div class="label">Candidates Tonight</div>
    <div class="value">{len(short_term) + len(long_term)}</div>
    <div class="sub">{len(short_term)} short-term, {len(long_term)} long-term</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

models, missing = get_models()
if missing:
    st.caption(f"⚠️ No trained model for horizon(s) {missing} -- some reasoning may be incomplete.")

if short_term:
    st.markdown('<div class="section-heading">⚡ Short-Term Candidates (1-2 weeks, dual-confirmed)</div>',
                unsafe_allow_html=True)
    st.caption("Ranked in the top 20% for BOTH of the two shortest horizons -- two independent "
               "confirmations of a near-term signal, not just one noisy read.")
    for i, pick in enumerate(short_term):
        with st.expander(
            f"{MEDALS[i]} {pick['ticker']} — {pick['name']} — ${pick['price']:.2f}",
            expanded=(i == 0),
        ):
            h1, h2 = pick["horizons"]
            r1, r2 = pick["pred_return_pct"]
            p1, p2 = pick["prob_up"]
            st.markdown(
                f"**{HORIZON_LABELS[h1]}**: {r1:+.2f}% predicted, {p1*100:.0f}% chance of a gain "
                f"&nbsp;·&nbsp; **{HORIZON_LABELS[h2]}**: {r2:+.2f}% predicted, {p2*100:.0f}% chance "
                f"of a gain &nbsp;·&nbsp; average rank **#{pick['avg_rank']:.1f}** across both"
            )
            st.markdown("---")
            render_pick_reasoning(pick["ticker"], pick["horizons"], models)

if long_term:
    st.markdown('<div class="section-heading">🏦 Longer-Term Candidates (consistent across all horizons)</div>',
                unsafe_allow_html=True)
    st.caption("Ranked in the top 50% across EVERY horizon the model forecasts, from 1 week out to "
               "~9.5 months -- consistency across the whole curve, not a one-horizon fluke.")
    for i, pick in enumerate(long_term):
        with st.expander(
            f"{MEDALS[i]} {pick['ticker']} — {pick['name']} — ${pick['price']:.2f}",
            expanded=(i == 0),
        ):
            h = pick["horizon"]
            st.markdown(
                f"**{HORIZON_LABELS[h]}**: {pick['pred_return_pct']:+.2f}% predicted, "
                f"{pick['prob_up']*100:.0f}% chance of a gain &nbsp;·&nbsp; "
                f"average rank **#{pick['avg_rank']:.1f}** across all horizons"
            )
            # JSON object keys are always strings -- rank_by_horizon's keys
            # come back as "5"/"10"/etc. after the briefing/latest.json
            # round-trip, even though they started as ints in nightly_briefing.py.
            rank_str = " · ".join(f"{HORIZON_LABELS[int(hz)]}: #{r}"
                                   for hz, r in pick["rank_by_horizon"].items())
            st.caption(f"Per-horizon rank — {rank_str}")
            st.markdown("---")
            render_pick_reasoning(pick["ticker"], [h], models)

st.markdown("---")
st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
st.caption(
    "For the full, un-opinionated picture on any stock — including ones that DIDN'T make this "
    "list — use Stock Research or My Watchlist. Those pages describe; this page is the only "
    "one that picks."
)
