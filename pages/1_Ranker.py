"""
Streamlit UI for the multi-stock ranker. Secondary page — the main search
experience is app.py (single-stock Deep Dive). This page is for comparing
many stocks against each other at once.
Reached via the page picker in the sidebar.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_TRAIN_UNIVERSE, HORIZONS, MODELS_DIR
from watchlist_store import get_effective_sections
from data_fetcher import search_ticker, validate_tickers
from predict import latest_features_for_ticker, load_models, score_universe
from insights import generate_why_bullets
from reliability import load_backtest_results, reliability_ranking

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ranker · Stock Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.main-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 28px 36px; margin-bottom: 24px;
}
.main-header h1 { font-size: 2.1rem; font-weight: 700; color: #58a6ff; margin: 0; }
.main-header p  { color: #8b949e; margin: 6px 0 0; font-size: 0.95rem; }

.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.metric-card .label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; margin-top: 4px; }
.metric-card .sub   { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }

.step-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px; height: 100%;
}
.step-num { font-size: 1.8rem; font-weight: 800; color: #58a6ff; }
.step-title { font-weight: 700; color: #e6edf3; font-size: 1rem; margin: 6px 0 4px; }
.step-body  { color: #8b949e; font-size: .87rem; line-height: 1.5; }

.glossary-row { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 14px; }
.glossary-term { color: #58a6ff; font-weight: 600; min-width: 160px; font-size:.9rem; }
.glossary-def  { color: #8b949e; font-size:.87rem; line-height: 1.5; }

[data-testid="stDataFrame"] table { background: #161b22 !important; }
[data-testid="stDataFrame"] th    { background: #21262d !important; color: #8b949e !important; }
[data-testid="stDataFrame"] td    { color: #e6edf3 !important; }

[data-baseweb="tab-list"]   { background: #161b22 !important; border-radius: 8px; gap: 4px; }
[data-baseweb="tab"]        { color: #8b949e !important; border-radius: 6px !important; }
[aria-selected="true"]      { background: #21262d !important; color: #58a6ff !important; }

.stButton > button {
    background: #238636; color: #fff; border: none;
    border-radius: 8px; font-weight: 600; padding: 10px 28px;
    font-size: 0.95rem; transition: background .2s;
}
.stButton > button:hover { background: #2ea043; }

.stTextArea textarea, .stTextInput input, .stMultiSelect div {
    background: #0d1117 !important; color: #e6edf3 !important;
    border-color: #30363d !important;
}

.info-box {
    background: #0c2d45; border-left: 4px solid #58a6ff;
    border-radius: 6px; padding: 12px 16px; color: #a5c8f0;
    font-size: 0.88rem; margin-bottom: 16px;
}
.warn-box {
    background: #2d2007; border-left: 4px solid #d29922;
    border-radius: 6px; padding: 14px 18px; color: #d29922;
    font-size: 0.88rem; margin-bottom: 16px; line-height: 1.6;
}
.search-result {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
}
.search-result .sym  { color: #58a6ff; font-weight: 700; font-size: .95rem; }
.search-result .name { color: #e6edf3; font-size: .85rem; margin-top: 2px; }
.search-result .exch { color: #8b949e; font-size: .78rem; }

/* ── Hover-tooltip table ── */
.sr-table-wrap { width: 100%; overflow: visible; }
.sr-table {
    width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem;
}
.sr-table th {
    background: #21262d; color: #8b949e; font-weight: 600; padding: 10px 14px;
    text-align: left; font-size: .77rem; text-transform: uppercase; letter-spacing: .05em;
    border-bottom: 1px solid #30363d; white-space: nowrap;
}
.sr-table td { padding: 10px 14px; color: #e6edf3; border-bottom: 1px solid #21262d; vertical-align: middle; }
.sr-table tr:hover td { background: #161b2288; }
.sr-rank { color: #8b949e; font-size: .85rem; }

/* Ticker cell with popup */
.tt-cell { position: relative; font-weight: 700; font-size: .95rem; cursor: help; }
.tt-cell::after { content: " ⓘ"; font-size: .7rem; color: #8b949e; font-weight: 400; }
.tt-popup {
    display: none;
    position: absolute;
    left: 0; top: calc(100% + 6px);
    z-index: 9999; width: 310px;
    background: #1c2128;
    border: 1px solid #388bfd66;
    border-radius: 10px; padding: 13px 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,.75);
    pointer-events: none;
}
.tt-cell:hover .tt-popup { display: block; }
.tt-head { font-weight: 600; color: #58a6ff; font-size: .81rem; margin-bottom: 9px; }
.tt-list { margin: 0; padding: 0; list-style: none; }
.tt-list li {
    color: #c9d1d9; font-size: .80rem; line-height: 1.55;
    padding: 3px 0 3px 14px; position: relative; border-bottom: 1px solid #21262d22;
}
.tt-list li:last-child { border-bottom: none; }
.tt-list li::before { content: "·"; position: absolute; left: 4px; color: #58a6ff; font-weight: 700; }

/* ---- Investment pick cards ---- */
.picks-section {
    border-radius: 12px; padding: 22px 24px; margin-bottom: 24px;
}
.picks-section.short { border: 1px solid #3fb95066; background: linear-gradient(135deg,#0d1f12,#0d1117); }
.picks-section.long  { border: 1px solid #58a6ff66; background: linear-gradient(135deg,#0c1a2e,#0d1117); }
.picks-section-title { font-size: 1.15rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px; }
.picks-section-sub   { color: #8b949e; font-size: .87rem; margin-bottom: 18px; line-height: 1.5; }

.pick-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 18px 18px 14px; height: 100%; display: flex; flex-direction: column;
}
.pick-medal  { font-size: 1.4rem; margin-bottom: 6px; }
.pick-ticker { font-size: 1.5rem; font-weight: 800; color: #e6edf3; line-height: 1.1; }
.pick-price  { color: #8b949e; font-size: .83rem; margin: 3px 0 14px; }
.pick-row    { display: flex; justify-content: space-between; align-items: center;
               margin-bottom: 7px; }
.pick-lbl    { color: #8b949e; font-size: .82rem; }
.pick-val    { font-weight: 600; font-size: .92rem; color: #e6edf3; }
.pick-why    { margin-top: auto; padding-top: 12px; border-top: 1px solid #21262d;
               color: #8b949e; font-size: .80rem; line-height: 1.45; }
.pick-why b  { color: #a5c8f0; }
.no-picks    { color: #8b949e; font-size: .9rem; padding: 16px 0; }

.badge-short { background:#1a4731; color:#3fb950; border-radius:4px;
               padding:2px 7px; font-size:.72rem; font-weight:700; margin-left:8px; }
.badge-long  { background:#0c2d45; color:#58a6ff; border-radius:4px;
               padding:2px 7px; font-size:.72rem; font-weight:700; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

# ── Tab labels ────────────────────────────────────────────────────────────────
HORIZON_LABELS = {
    5:   ("Next Week",       "5 trading days"),
    10:  ("2 Weeks Out",     "10 trading days"),
    15:  ("3 Weeks Out",     "15 trading days"),
    30:  ("Next Month",      "30 trading days"),
    50:  ("~2.5 Months Out", "50 trading days"),
    200: ("~9.5 Months Out", "200 trading days"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    missing = [h for h in HORIZONS if not (MODELS_DIR / f"horizon_{h}.joblib").exists()]
    if missing:
        return None, missing
    return load_models(), []


def models_trained() -> bool:
    return all((MODELS_DIR / f"horizon_{h}.joblib").exists() for h in HORIZONS)


def build_bar_chart(df: pd.DataFrame, horizon: int) -> go.Figure:
    label, _ = HORIZON_LABELS[horizon]
    top = df.head(15).copy()
    scores = top["composite_score"]
    lo, hi = scores.min(), scores.max()
    mid = (lo + hi) / 2
    colors = []
    for v in scores:
        if v >= mid + (hi - mid) * 0.25:
            colors.append("#3fb950")
        elif v <= mid - (mid - lo) * 0.25:
            colors.append("#f85149")
        else:
            colors.append("#d29922")

    fig = go.Figure(go.Bar(
        x=top["ticker"],
        y=scores,
        marker_color=colors,
        text=[f"{v:.2f}" for v in scores],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=11),
        hovertemplate="<b>%{x}</b><br>AI Score: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        title=dict(text=f"AI Ranking — {label}", font=dict(size=14, color="#8b949e")),
        xaxis=dict(gridcolor="#21262d", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#21262d", title="AI Score"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
    )
    return fig


def render_hover_table(df: pd.DataFrame, horizon: int, top_n: int, feat_df: pd.DataFrame | None) -> None:
    col_ret = f"pred_return_{horizon}d_pct"
    _, days_desc = HORIZON_LABELS[horizon]

    rows_html: list[str] = []
    for rank_i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        ticker = str(row["ticker"])
        pred   = float(row[col_ret])
        prob   = float(row["prob_up"])

        # Generate tooltip bullets
        if feat_df is not None and ticker in feat_df.index:
            bullets = generate_why_bullets(feat_df.loc[ticker], pred, prob)
        else:
            bullets = ["Run a fresh analysis to see per-stock reasoning."]

        direction = "bullish 📈" if pred >= 0 else "bearish 📉"
        li_items  = "".join(f"<li>{b}</li>" for b in bullets)
        popup_html = (
            f'<div class="tt-popup">'
            f'<div class="tt-head">Why the AI is {direction} on {ticker}:</div>'
            f'<ul class="tt-list">{li_items}</ul>'
            f'</div>'
        )

        gain_color = "#3fb950" if pred >= 0 else "#f85149"
        prob_pct   = prob * 100
        prob_color = "#3fb950" if prob_pct >= 55 else ("#f85149" if prob_pct < 45 else "#d29922")

        rows_html.append(
            f"<tr>"
            f'<td class="sr-rank">{rank_i}</td>'
            f'<td class="tt-cell">{ticker}{popup_html}</td>'
            f"<td>${float(row['last_close']):.2f}</td>"
            f'<td style="color:{gain_color};font-weight:600">{pred:+.2f}%</td>'
            f'<td style="color:{prob_color};font-weight:600">{prob_pct:.0f}%</td>'
            f"<td>{float(row['risk_adjusted']):.2f}</td>"
            f'<td style="color:#58a6ff;font-weight:700">{float(row["composite_score"]):.2f}</td>'
            f"</tr>"
        )

    table_html = (
        '<div class="sr-table-wrap">'
        '<table class="sr-table"><thead><tr>'
        "<th>#</th>"
        '<th>Company <span style="color:#8b949e;font-size:.72rem;font-weight:400;text-transform:none">— hover for details</span></th>'
        "<th>Current Price</th>"
        f"<th>Expected Gain ({days_desc})</th>"
        "<th>Chance of Rising</th>"
        "<th>Consistency</th>"
        "<th>AI Score</th>"
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_horizon_tab(df: pd.DataFrame, horizon: int, top_n: int):
    col_ret = f"pred_return_{horizon}d_pct"
    _, days_desc = HORIZON_LABELS[horizon]
    top1 = df.iloc[0]

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Top Pick</div>
            <div class="value">{top1['ticker']}</div>
            <div class="sub">Highest ranked stock</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        gain = top1[col_ret]
        gain_color = "#3fb950" if gain >= 0 else "#f85149"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Expected Price Change</div>
            <div class="value" style="color:{gain_color}">{gain:+.2f}%</div>
            <div class="sub">Over the next {days_desc}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        prob = top1['prob_up'] * 100
        prob_color = "#3fb950" if prob >= 55 else ("#f85149" if prob < 45 else "#d29922")
        st.markdown(f"""<div class="metric-card">
            <div class="label">Chance of Going Up</div>
            <div class="value" style="color:{prob_color}">{prob:.1f}%</div>
            <div class="sub">For the top pick</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_prob = df.head(top_n)["prob_up"].mean() * 100
        st.markdown(f"""<div class="metric-card">
            <div class="label">Average Chance Up</div>
            <div class="value">{avg_prob:.1f}%</div>
            <div class="sub">Across your top {top_n} picks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([5, 4], gap="large")

    with left:
        st.plotly_chart(build_bar_chart(df, horizon), use_container_width=True)

    with right:
        feat_df = st.session_state.get("feat_df")
        render_hover_table(df, horizon, top_n, feat_df)

    # Column glossary
    with st.expander("What do these columns mean?"):
        st.markdown("""
<div class="glossary-row">
  <div class="glossary-term">Company</div>
  <div class="glossary-def">The stock symbol (short code) for the company — e.g. AAPL = Apple, MSFT = Microsoft.</div>
</div>
<div class="glossary-row">
  <div class="glossary-term">Current Price</div>
  <div class="glossary-def">The most recent closing price of the stock, in US dollars.</div>
</div>
<div class="glossary-row">
  <div class="glossary-term">Expected Gain</div>
  <div class="glossary-def">The AI's best estimate of how much the stock price will change (up or down) over this time period. A positive number means the model thinks the price will rise.</div>
</div>
<div class="glossary-row">
  <div class="glossary-term">Chance of Rising</div>
  <div class="glossary-def">The probability (out of 100%) that the stock will be higher than today at the end of this period. Above 50% means the model leans toward a price increase.</div>
</div>
<div class="glossary-row">
  <div class="glossary-term">Consistency</div>
  <div class="glossary-def">How stable the expected gain looks relative to how volatile (up-and-down) the stock has been recently. Higher is better — it means the expected gain comes without wild price swings.</div>
</div>
<div class="glossary-row">
  <div class="glossary-term">AI Score</div>
  <div class="glossary-def">The overall ranking score combining all three signals above. Higher = ranked better. Use this to compare stocks against each other, not as an absolute measure.</div>
</div>
""", unsafe_allow_html=True)


# ── Investment ideas helpers ──────────────────────────────────────────────────
def get_short_long_picks(results: dict, n_scored: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Short-term: stocks that rank in the top 20% (min 5) for BOTH the 5-day
                and 10-day horizons — dual confirmation of a near-term signal.
    Long-term:  stocks that rank in the top 50% across EVERY horizon in
                config.HORIZONS, sorted by their average rank (most
                consistent overall). "Expected gain" uses the longest
                configured horizon.
    """
    cutoff_s = max(5, n_scored // 5)   # top 20%
    cutoff_l = max(5, n_scored // 2)   # top 50%
    longest_h = HORIZONS[-1]

    short_tickers = (
        set(results[5].head(cutoff_s)["ticker"]) &
        set(results[10].head(cutoff_s)["ticker"])
    )
    long_tickers = set(results[HORIZONS[0]].head(cutoff_l)["ticker"])
    for h in HORIZONS[1:]:
        long_tickers &= set(results[h].head(cutoff_l)["ticker"])

    idx = {h: results[h].set_index("ticker") for h in HORIZONS}

    # ── Short-term dataframe ──
    short_rows = []
    for t in short_tickers:
        short_rows.append({
            "ticker":      t,
            "last_close":  idx[5].loc[t, "last_close"],
            "pred_5d":     idx[5].loc[t, "pred_return_5d_pct"],
            "pred_10d":    idx[10].loc[t, "pred_return_10d_pct"],
            "prob_up_5d":  idx[5].loc[t, "prob_up"],
            "prob_up_10d": idx[10].loc[t, "prob_up"],
            "risk_adj":    (idx[5].loc[t, "risk_adjusted"] + idx[10].loc[t, "risk_adjusted"]) / 2,
            "score_5d":    idx[5].loc[t, "composite_score"],
            "score_10d":   idx[10].loc[t, "composite_score"],
        })
    short_df = pd.DataFrame(short_rows)
    if not short_df.empty:
        short_df["avg_score"] = (short_df["score_5d"] + short_df["score_10d"]) / 2
        short_df = short_df.sort_values("avg_score", ascending=False).reset_index(drop=True)

    # ── Long-term dataframe ──
    long_rows = []
    ticker_lists = {h: results[h]["ticker"].tolist() for h in HORIZONS}
    for t in long_tickers:
        ra_avg = sum(idx[h].loc[t, "risk_adjusted"] for h in HORIZONS) / len(HORIZONS)
        avg_rank = sum(ticker_lists[h].index(t) + 1 for h in HORIZONS) / len(HORIZONS)
        row = {
            "ticker":     t,
            "last_close": idx[longest_h].loc[t, "last_close"],
            "pred_long":  idx[longest_h].loc[t, f"pred_return_{longest_h}d_pct"],
            "prob_up":    idx[longest_h].loc[t, "prob_up"],
            "risk_adj":   ra_avg,
            "avg_rank":   avg_rank,
        }
        for h in HORIZONS:
            row[f"rank_{h}d"] = ticker_lists[h].index(t) + 1
        long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values("avg_rank", ascending=True).reset_index(drop=True)

    return short_df, long_df


def _pick_why_short(row: pd.Series) -> str:
    parts = []
    prob = max(float(row["prob_up_5d"]), float(row["prob_up_10d"])) * 100
    if prob >= 65:
        parts.append(f"<b>{prob:.0f}% chance of rising</b> over the next two weeks")
    if row["pred_5d"] > 1.0:
        parts.append(f"AI expects <b>{row['pred_5d']:+.1f}%</b> in just the first week")
    if row["pred_10d"] > 1.5:
        parts.append(f"<b>{row['pred_10d']:+.1f}%</b> predicted over two weeks")
    if row["risk_adj"] > 0.5:
        parts.append("gains look <b>steady, not erratic</b>")
    if not parts:
        parts.append("ranks in the <b>top picks for both</b> the 1-week and 2-week outlooks")
    return " · ".join(parts[:2])


def _pick_why_long(row: pd.Series) -> str:
    n_horizons = len(HORIZONS)
    parts = []
    if row["avg_rank"] <= 5:
        parts.append(f"<b>average rank #{row['avg_rank']:.1f}</b> across all {n_horizons} time frames")
    elif row["avg_rank"] <= 10:
        parts.append(f"consistently ranks around <b>#{row['avg_rank']:.0f}</b> across every time frame")
    if row["risk_adj"] > 0.4:
        parts.append("signal is <b>stable and consistent</b>, not just a one-week blip")
    if row["prob_up"] >= 0.6:
        parts.append(f"<b>{row['prob_up']*100:.0f}% chance</b> of still being higher by the end of the window")
    if not parts:
        parts.append(f"appears in the <b>top half across all {n_horizons} time frames</b> — a durable signal")
    return " · ".join(parts[:2])


def render_investment_ideas(results: dict, n_scored: int):
    short_df, long_df = get_short_long_picks(results, n_scored)

    MEDALS = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    # ── Short-term section ────────────────────────────────────────────────────
    st.markdown("""<div class="picks-section short">
        <div class="picks-section-title">⚡ Short-Term Opportunities <span class="badge-short">NEXT 1–2 WEEKS</span></div>
        <div class="picks-section-sub">
            These stocks ranked highly for <em>both</em> the 1-week and 2-week outlooks —
            meaning the AI sees a near-term opportunity, not just a random spike.
            Best for people comfortable checking in and potentially selling within a couple of weeks.
        </div>
    </div>""", unsafe_allow_html=True)

    if short_df.empty:
        st.markdown('<div class="no-picks">Not enough agreement across the 1-week and 2-week rankings to highlight any stocks right now. Try expanding to a larger stock list.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(min(len(short_df), 3))
        for i, col in enumerate(cols):
            if i >= len(short_df):
                break
            row = short_df.iloc[i]
            gain_color = "#3fb950" if row["pred_10d"] >= 0 else "#f85149"
            prob = max(float(row["prob_up_5d"]), float(row["prob_up_10d"])) * 100
            prob_color = "#3fb950" if prob >= 60 else ("#d29922" if prob >= 50 else "#f85149")
            with col:
                st.markdown(f"""<div class="pick-card">
                    <div class="pick-medal">{MEDALS[i]}</div>
                    <div class="pick-ticker">{row['ticker']}</div>
                    <div class="pick-price">Current price: ${row['last_close']:.2f}</div>
                    <div class="pick-row">
                        <span class="pick-lbl">Expected gain (1 week)</span>
                        <span class="pick-val" style="color:{gain_color}">{row['pred_5d']:+.2f}%</span>
                    </div>
                    <div class="pick-row">
                        <span class="pick-lbl">Expected gain (2 weeks)</span>
                        <span class="pick-val" style="color:{gain_color}">{row['pred_10d']:+.2f}%</span>
                    </div>
                    <div class="pick-row">
                        <span class="pick-lbl">Chance of going up</span>
                        <span class="pick-val" style="color:{prob_color}">{prob:.0f}%</span>
                    </div>
                    <div class="pick-why">💡 {_pick_why_short(row)}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Long-term section ─────────────────────────────────────────────────────
    st.markdown("""<div class="picks-section long">
        <div class="picks-section-title">🏦 Steadier, Longer-Term Picks <span class="badge-long">UP TO ~9.5 MONTHS</span></div>
        <div class="picks-section-sub">
            These stocks rank well across <em>every</em> time frame we analyze — not just next week,
            but also 2 weeks, 3 weeks, a month, ~2.5 months, and ~9.5 months out. That consistency
            is a stronger signal than a stock that looks great for one week and then fades.<br><br>
            <b>Note:</b> Our AI looks up to 200 trading days (~9.5 months) ahead. For multi-year
            investing, these rankings are a starting point — not a substitute for deeper research.
        </div>
    </div>""", unsafe_allow_html=True)

    longest_h = HORIZONS[-1]
    longest_label, _ = HORIZON_LABELS[longest_h]

    if long_df.empty:
        st.markdown(f'<div class="no-picks">No stocks ranked in the top half across all {len(HORIZONS)} time frames right now. Try using a larger stock list.</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(min(len(long_df), 3))
        for i, col in enumerate(cols):
            if i >= len(long_df):
                break
            row = long_df.iloc[i]
            gain_color = "#3fb950" if row["pred_long"] >= 0 else "#f85149"
            prob_color = "#3fb950" if row["prob_up"] >= 0.6 else ("#d29922" if row["prob_up"] >= 0.5 else "#f85149")
            with col:
                st.markdown(f"""<div class="pick-card">
                    <div class="pick-medal">{MEDALS[i]}</div>
                    <div class="pick-ticker">{row['ticker']}</div>
                    <div class="pick-price">Current price: ${row['last_close']:.2f}</div>
                    <div class="pick-row">
                        <span class="pick-lbl">Expected gain ({longest_label})</span>
                        <span class="pick-val" style="color:{gain_color}">{row['pred_long']:+.2f}%</span>
                    </div>
                    <div class="pick-row">
                        <span class="pick-lbl">Chance of going up</span>
                        <span class="pick-val" style="color:{prob_color}">{row['prob_up']*100:.0f}%</span>
                    </div>
                    <div class="pick-row">
                        <span class="pick-lbl">Average rank across all {len(HORIZONS)} time frames</span>
                        <span class="pick-val">#{row['avg_rank']:.1f}</span>
                    </div>
                    <div class="pick-why">💡 {_pick_why_long(row)}</div>
                </div>""", unsafe_allow_html=True)

        if len(long_df) > 3:
            with st.expander(f"See all {len(long_df)} consistent picks"):
                rank_cols = [f"rank_{h}d" for h in HORIZONS]
                disp = long_df[["ticker", "last_close", "pred_long", "prob_up", "risk_adj", "avg_rank",
                                *rank_cols]].copy()
                disp.index = disp.index + 1
                disp.index.name = "Rank"
                rank_headers = [f"Rank {HORIZON_LABELS[h][0]}" for h in HORIZONS]
                disp.columns = pd.Index([
                    "Company", "Price", f"{longest_label} Gain", "Chance Up",
                    "Consistency", "Avg Rank", *rank_headers,
                ])
                disp["Price"]              = disp["Price"].map(lambda x: f"${x:.2f}")
                disp[f"{longest_label} Gain"] = disp[f"{longest_label} Gain"].map(lambda x: f"{x:+.2f}%")
                disp["Chance Up"]          = disp["Chance Up"].map(lambda x: f"{x*100:.0f}%")
                disp["Consistency"]        = disp["Consistency"].map(lambda x: f"{x:.2f}")
                disp["Avg Rank"]           = disp["Avg Rank"].map(lambda x: f"#{x:.1f}")
                st.dataframe(disp, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    st.markdown("#### Which stocks do you want to analyze?")
    source = st.radio(
        "Stock source",
        ["Built-in list of 50 popular US stocks", "My watchlist", "Type my own symbols"],
        index=1,
        label_visibility="collapsed",
    )

    if source == "Built-in list of 50 popular US stocks":
        chosen = DEFAULT_TRAIN_UNIVERSE.copy()
        st.caption(f"Analyzing {len(chosen)} large, well-known US companies.")
    elif source == "My watchlist":
        st.caption("Grouped from your saved stock images. Uncheck a section to leave it out.")
        watchlist_sections = get_effective_sections()
        chosen = []
        for section_name, tickers in watchlist_sections.items():
            with st.expander(f"{section_name} ({len(tickers)})", expanded=False):
                include = st.checkbox("Include this section", value=True, key=f"wl_{section_name}")
                st.caption(", ".join(tickers))
            if include:
                chosen.extend(tickers)
        chosen = list(dict.fromkeys(chosen))  # dedupe, keep order
        st.caption(f"{len(chosen)} stock{'s' if len(chosen) != 1 else ''} selected across {sum(1 for s in watchlist_sections if st.session_state.get(f'wl_{s}', True))} sections.")
    else:
        st.markdown("""<div style="background:#0c2d45;border-radius:6px;padding:10px 12px;
            color:#a5c8f0;font-size:.82rem;margin-bottom:8px;line-height:1.5;">
            💡 <b>What is a stock symbol?</b><br>
            Every publicly traded company has a short code called a <i>symbol</i> or <i>ticker</i>.
            For example, Apple is <b>AAPL</b>, Tesla is <b>TSLA</b>, and Google is <b>GOOGL</b>.
            Use the search box below if you don't know the symbol for a company.
        </div>""", unsafe_allow_html=True)
        raw = st.text_area(
            "Type stock symbols, one per line",
            placeholder="AAPL\nMSFT\nNVDA\nTSLA",
            height=140,
            help="Each line should be a stock symbol. Not sure of the symbol? Use the search below.",
        )
        chosen = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
        if chosen:
            st.caption(f"{len(chosen)} stock{'s' if len(chosen) != 1 else ''} entered.")

    st.markdown("---")
    st.markdown("#### How many results to show?")
    top_n = st.slider(
        "Show top ___ stocks per time period",
        min_value=3, max_value=25, value=10, step=1,
        help="For example, set to 5 to see only the top 5 ranked stocks, or 15 to see a longer list.",
    )

    st.markdown("---")
    st.markdown("#### Don't know a company's symbol?")
    st.caption("Search by company name to find the right symbol to use above.")
    query = st.text_input(
        "Search by company name",
        placeholder="e.g. Apple, Tesla, Nvidia…",
    )
    if query:
        with st.spinner("Searching…"):
            hits = search_ticker(query, max_results=6)
        if hits:
            for h in hits:
                if h.get("symbol"):
                    st.markdown(f"""<div class="search-result">
                        <div class="sym">{h['symbol']}</div>
                        <div class="name">{h.get('name','')}</div>
                        <div class="exch">{h.get('exchange','')}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.caption("No results found — try a different spelling.")

    st.markdown("---")
    ready = models_trained()
    st.caption("System status: " + ("✅ Ready to analyze" if ready else "❌ Models not trained — run train.py first"))
    run_btn = st.button("Analyze Stocks Now", use_container_width=True, disabled=not ready)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
    <h1>📈 Stock Ranker</h1>
    <p>An AI tool that ranks stocks by their potential over the next week, 2 weeks, 3 weeks, and month —
    using price patterns and trading data, updated live.</p>
</div>""", unsafe_allow_html=True)

if not models_trained():
    st.markdown("""<div class="warn-box">
        ⚠️ <b>Setup needed:</b> The AI models haven't been trained yet.
        Open a terminal in this folder and run <code>python train.py</code> to get started.
        This only needs to be done once (and periodically refreshed).
    </div>""", unsafe_allow_html=True)
    st.stop()

if not run_btn and "results" not in st.session_state:
    # Landing page
    st.markdown("""<div class="info-box">
        👈 &nbsp;Ready to go! Choose your stocks in the sidebar on the left, then click
        <strong>Analyze Stocks Now</strong> to see the rankings.
    </div>""", unsafe_allow_html=True)

    st.markdown("### How it works — in plain English")
    st.caption("The AI looks at recent price and trading data for each stock, then answers three questions:")

    c1, c2, c3 = st.columns(3)
    for col, emoji, num, title, body in [
        (c1, "📈", "1", "Will the price go up?",
         "The AI is trained on years of historical stock data to estimate how much a stock's price is likely to change over the chosen time period. Think of it like a weather forecast — it gives a best guess, not a guarantee."),
        (c2, "🎯", "2", "How confident is it?",
         "A second AI model separately estimates the odds that the price will be higher (not just by how much). When both models agree a stock looks good, the ranking gets a bigger boost."),
        (c3, "⚖️", "3", "Is the gain worth the risk?",
         "A stock that bounces wildly up and down every day is riskier than one that moves steadily. The AI adjusts rankings so that calmer, more predictable gains rank higher than the same gain on a volatile stock."),
    ]:
        col.markdown(f"""<div class="step-card">
            <div class="step-num">{emoji}</div>
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### What you'll see after clicking Analyze")

    cols = st.columns(len(HORIZONS) + 1)
    for col, label, sub in zip(cols, [
        "💡 Best Picks", *[HORIZON_LABELS[h][0] for h in HORIZONS]
    ], [
        "Short-term & longer-term suggestions, explained in plain English",
        *[HORIZON_LABELS[h][1] for h in HORIZONS],
    ]):
        col.markdown(f"""<div class="metric-card">
            <div class="value" style="font-size:1rem">{label}</div>
            <div class="sub" style="margin-top:6px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="warn-box">
        ⚠️ <b>Important reminder:</b> This tool is for research and learning only.
        Stock prices are unpredictable — no AI can reliably predict them.
        The AI only looks at price data, not news, earnings, or world events.
        Never make investment decisions based solely on any automated tool.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ── Run analysis ──────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:

    if run_btn:
        if not chosen:
            st.error("Please enter at least one stock symbol in the sidebar, or keep the default list checked.")
            st.stop()

        progress_bar = st.progress(0, text="Starting up…")
        models, missing = get_models()
        if models is None:
            st.error(f"AI models are missing for these time periods: {missing}. Please run train.py.")
            st.stop()

        progress_bar.progress(15, text=f"Checking that your {len(chosen)} stocks exist…")
        valid = validate_tickers(chosen)
        if not valid:
            st.error("None of those stock symbols returned any data. Double-check the symbols and try again.")
            progress_bar.empty()
            st.stop()

        skipped = len(chosen) - len(valid)
        if skipped:
            st.warning(f"{skipped} symbol(s) couldn't be found and were skipped.")

        progress_bar.progress(30, text=f"Downloading the latest prices for {len(valid)} stocks…")

        rows = {}
        for i, t in enumerate(valid):
            pct = 30 + int(60 * (i + 1) / len(valid))
            progress_bar.progress(pct, text=f"Getting data for {t}… ({i+1} of {len(valid)})")
            feat = latest_features_for_ticker(t)
            if feat is not None:
                rows[t] = feat

        if not rows:
            st.error("Couldn't retrieve enough recent price history for any of those stocks.")
            progress_bar.empty()
            st.stop()

        from features import FEATURE_COLUMNS
        feat_df = pd.DataFrame(rows).T
        st.session_state["feat_df"] = feat_df[FEATURE_COLUMNS].astype(float)

        progress_bar.progress(95, text="Running the AI models and ranking stocks…")
        results = score_universe(list(rows.keys()), models)

        st.session_state["results"]  = results
        st.session_state["n_valid"]  = len(valid)
        st.session_state["n_scored"] = len(rows)
        st.session_state["run_time"] = time.strftime("%I:%M %p")
        progress_bar.progress(100, text="All done!")
        time.sleep(0.4)
        progress_bar.empty()

    results  = st.session_state["results"]
    n_valid  = st.session_state["n_valid"]
    n_scored = st.session_state["n_scored"]
    run_time = st.session_state["run_time"]

    # Summary strip
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"""<div class="metric-card">
        <div class="label">Stocks Analyzed</div>
        <div class="value">{n_scored}</div>
        <div class="sub">out of {n_valid} found</div>
    </div>""", unsafe_allow_html=True)
    s2.markdown(f"""<div class="metric-card">
        <div class="label">Time Periods</div>
        <div class="value">{len(HORIZONS)}</div>
        <div class="sub">{' · '.join(HORIZON_LABELS[h][0] for h in HORIZONS)}</div>
    </div>""", unsafe_allow_html=True)

    longest_h = HORIZONS[-1]
    top_long = results[longest_h].iloc[0]
    top_long_gain = top_long[f"pred_return_{longest_h}d_pct"]
    gain_color = "#3fb950" if top_long_gain >= 0 else "#f85149"
    s3.markdown(f"""<div class="metric-card">
        <div class="label">Best {HORIZON_LABELS[longest_h][0]} Pick</div>
        <div class="value">{top_long['ticker']}</div>
        <div class="sub" style="color:{gain_color}">Expected {top_long_gain:+.2f}%</div>
    </div>""", unsafe_allow_html=True)
    s4.markdown(f"""<div class="metric-card">
        <div class="label">Last Updated</div>
        <div class="value" style="font-size:1.2rem">{run_time}</div>
        <div class="sub">using live market data</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Re-run button inline
    rcol, _ = st.columns([1, 4])
    with rcol:
        if st.button("Re-run with fresh data"):
            for k in ("results", "feat_df"):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    ranked = reliability_ranking(load_backtest_results())
    if ranked:
        best, worst = ranked[0], ranked[-1]
        st.caption(
            f"📊 From the last walk-forward backtest: **{best['horizon']}-day** picks have been "
            f"the most reliable (mean IC {best['mean_ic']:+.3f}), **{worst['horizon']}-day** the "
            f"least (mean IC {worst['mean_ic']:+.3f}). Run `python backtest.py` to refresh this."
        )
    else:
        st.caption("📊 Run `python backtest.py` to see which time period's picks have historically "
                   "been most reliable.")

    tab_labels = ["💡 Best Picks"] + [f"{'📅' if h == 5 else '🗓'} {HORIZON_LABELS[h][0]}" for h in HORIZONS]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.caption("Stocks the AI highlights based on strength and consistency of their signals.")
        st.markdown("<br>", unsafe_allow_html=True)
        render_investment_ideas(results, n_scored)

    for tab, h in zip(tabs[1:], HORIZONS):
        with tab:
            _, days_desc = HORIZON_LABELS[h]
            st.caption(f"Rankings based on the next {days_desc} of trading")
            render_horizon_tab(results[h], h, top_n)

    st.markdown("---")
    st.markdown("""<div class="warn-box">
        ⚠️ <b>This is not financial advice.</b><br>
        The AI only analyzes historical price patterns — it cannot see news, company earnings,
        economic events, or anything else that moves markets. Stock prices are inherently
        unpredictable. A high ranking today does not mean a stock will go up.
        Always do your own research and consult a financial advisor before investing.
    </div>""", unsafe_allow_html=True)
