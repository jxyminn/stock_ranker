"""
Stock Research: search one stock and see everything the system knows about
it -- model predictions, technical read, earnings timing, market regime, and
fundamentals. This is the main entry point.
Run with: streamlit run app.py

Reuses the exact same data/logic as `python analyze.py <ticker>` (analyze.py
is the single source of truth for what goes into the report; this file only
adds Streamlit rendering on top).

The multi-stock ranker is still available as a secondary page -- see the
page picker in the sidebar ("Ranker").
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from config import HORIZONS, MODELS_DIR
from analyze import build_report, resolve_ticker, fmt_money, fmt
from insights import generate_why_bullets
from reliability import reliability_ranking

st.set_page_config(
    page_title="Stock Research",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

HORIZON_LABELS = {5: "1 week", 10: "2 weeks", 15: "3 weeks", 30: "1 month", 50: "~2.5 months", 200: "~9.5 months"}


def models_trained() -> bool:
    return all((MODELS_DIR / f"horizon_{h}.joblib").exists() for h in HORIZONS)


def build_price_chart(price_df: pd.DataFrame, earnings_df: pd.DataFrame | None, ticker: str) -> go.Figure:
    """Price + SMA20/50 + Bollinger Bands on top, volume below, earnings dates marked."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03,
    )

    # Bollinger Band shaded envelope
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["bb_upper"], line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["bb_lower"], line=dict(width=0),
        fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
        name="Bollinger Band (20d)", hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["sma_50"], line=dict(color="#d29922", width=1, dash="dot"),
        name="50d average",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["sma_20"], line=dict(color="#8b949e", width=1, dash="dot"),
        name="20d average",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["close"], line=dict(color="#58a6ff", width=2),
        name="Close price",
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(price_df["close"], price_df["open"])]
    fig.add_trace(go.Bar(
        x=price_df.index, y=price_df["volume"], marker_color=vol_colors,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    if earnings_df is not None and not earnings_df.empty:
        idx = earnings_df.index.tz_localize(None) if earnings_df.index.tz is not None else earnings_df.index
        chart_start = price_df.index.min()
        chart_end = price_df.index.max()
        chart_start_naive = chart_start.tz_localize(None) if chart_start.tz is not None else chart_start
        chart_end_naive = chart_end.tz_localize(None) if chart_end.tz is not None else chart_end
        for date, row in zip(idx, earnings_df["surprise_pct"]):
            if date < chart_start_naive or date > chart_end_naive:
                continue
            color = "#3fb950" if pd.notna(row) and row > 0 else "#f85149" if pd.notna(row) and row < 0 else "#8b949e"
            fig.add_vline(x=date, line=dict(color=color, width=1, dash="dash"),
                          opacity=0.6, row=1, col=1)

    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=80, b=0),
        height=480,
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                y=1.28,
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#21262d", activecolor="#388bfd", font=dict(color="#e6edf3", size=11),
            ),
        ),
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(gridcolor="#21262d", title_text="Volume", row=2, col=1)
    return fig


st.markdown("## 🔍 Stock Research")
st.caption("Search a stock to see everything the system knows about it — model predictions, "
           "technicals, earnings timing, market regime, and fundamentals, all in one place. "
           "Looking to compare many stocks at once? See **Ranker** in the sidebar.")

if not models_trained():
    st.warning("The AI models haven't been trained yet. Run `python train.py` first.")
    st.stop()

query = st.text_input("Stock symbol or company name", placeholder="e.g. AAPL or 'Nvidia'")
look_up_clicked = st.button("Look it up", type="primary", disabled=not query)

if look_up_clicked and query:
    with st.spinner(f"Looking up '{query}'…"):
        ticker = resolve_ticker(query)
    if ticker is None:
        st.error(f"No matching stock found for '{query}'. Try the exact ticker symbol.")
        st.stop()
    with st.spinner(f"Pulling live data and running the models for {ticker}…"):
        report = build_report(ticker)
    st.session_state["deep_dive_report"] = report

report = st.session_state.get("deep_dive_report")
if report is None:
    st.info("👆 Enter a stock above and click **Look it up**.")
    st.stop()

if "error" in report:
    st.error(f"Couldn't build a report for {report['ticker']}: {report['error']}")
    st.stop()

ticker = report["ticker"]
feat = report["features"]
fund = report["fundamentals"]
name = fund.get("long_name") or ticker

st.markdown(f"### {name} ({ticker})")
st.caption(f"As of {pd.Timestamp(report['as_of']).date()}  ·  Last close **${report['last_close']:.2f}**")

# ── Price chart ──────────────────────────────────────────────────────────
price_history = report.get("price_history")
if price_history is not None and not price_history.empty:
    st.plotly_chart(
        build_price_chart(price_history, report.get("earnings_history"), ticker),
        use_container_width=True,
        theme=None,
    )
    st.caption("Dashed vertical lines mark earnings reports — green if that report beat estimates, "
               "red if it missed, gray for an upcoming (estimated) date.")

# ── Model predictions ──────────────────────────────────────────────────────
st.markdown("#### Model predictions")
st.caption("From technical + earnings-timing + macro features only — not fundamentals or news.")
rows = []
for h in HORIZONS:
    p = report["predictions"][h]
    rows.append({
        "Horizon": HORIZON_LABELS[h],
        "Predicted return": f"{p['pred_return_pct']:+.2f}%",
        "Chance of a gain": f"{p['prob_up'] * 100:.0f}%",
        "Risk-adjusted": f"{p['risk_adjusted']:+.2f}",
        "Model reliability (MAE / AUC)": f"{p['holdout_mae']:.4f} / {p['holdout_auc']:.3f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("AUC 0.5 = no better than a coin flip. This is a single holdout split — run "
           "`python backtest.py` for a fuller, multi-fold reliability check before trusting it.")

ranked = reliability_ranking(report.get("backtest"))
if ranked:
    bt = report["backtest"]
    run_at = bt["run_at"][:16].replace("T", " ")
    best, worst = ranked[0], ranked[-1]
    st.caption(
        f"From the last `backtest.py` run ({run_at} UTC, {bt['folds']} folds, "
        f"{bt['universe_size']} tickers): **{best['horizon']}d** has been the most reliable "
        f"horizon (mean IC {best['mean_ic']:+.3f}), **{worst['horizon']}d** the least "
        f"(mean IC {worst['mean_ic']:+.3f})."
    )
else:
    st.caption("No backtest results found yet — run `python backtest.py` to rank horizons by "
               "actual walk-forward reliability instead of the noisy single-holdout numbers above.")

with st.expander("🔍 Why the model says this"):
    st.caption("Grounded in this model's own learned splits for this specific prediction — "
               "unlike the Technical Read below, which is a general plain-English summary, not "
               "literally derived from what the model weighted.")
    explain_horizon = st.selectbox(
        "Horizon", HORIZONS, index=HORIZONS.index(30) if 30 in HORIZONS else 0,
        format_func=lambda h: HORIZON_LABELS[h], key="explain_horizon",
    )
    drivers = report["predictions"][explain_horizon].get("explain", [])
    if not drivers:
        st.caption("Unavailable for this horizon.")
    else:
        for d in drivers:
            color = "#3fb950" if d["direction"] == "up" else "#f85149"
            st.markdown(
                f"**{d['label']}** (value {d['value']:+.3f}) pushed the forecast "
                f"<span style='color:{color}'>{d['direction']} {abs(d['contribution_pp']):.2f}pp</span>",
                unsafe_allow_html=True,
            )

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Technical read")
    avg_pred = sum(report["predictions"][h]["pred_return_pct"] for h in HORIZONS) / len(HORIZONS)
    avg_prob = sum(report["predictions"][h]["prob_up"] for h in HORIZONS) / len(HORIZONS)
    for bullet in generate_why_bullets(feat, avg_pred / 100, avg_prob):
        st.markdown(f"- {bullet}")
    st.markdown(f"- Volatility: **{feat['vol_20d'] * (252 ** 0.5) * 100:.1f}%** annualized (20d)")
    st.markdown(f"- 52-week position: **{feat['dist_from_52w_high'] * 100:+.1f}%** from high, "
                f"**{feat['dist_from_52w_low'] * 100:+.1f}%** above low")

    st.markdown("#### Earnings")
    days_until = feat.get("days_until_earnings")
    days_since = feat.get("days_since_earnings")
    surprise = feat.get("last_earnings_surprise_pct")
    if pd.notna(days_since):
        line = f"Last report **{days_since:.0f} days ago**"
        if pd.notna(surprise):
            line += f", surprise **{surprise:+.2f}%**"
        st.markdown(line)
    else:
        st.markdown("No earnings history on record.")
    if pd.notna(days_until):
        note = " ⚠️ within a week — expect higher volatility than usual" if days_until <= 7 else ""
        st.markdown(f"Next report in **{days_until:.0f} days**{note}")
    else:
        st.markdown("Next report date unknown.")

    st.markdown("#### Market regime")
    st.caption("Same market-wide context the model sees — not specific to this stock.")
    vix = feat.get("vix_level", float("nan"))
    st.markdown(f"S&P 500: **{feat.get('mkt_ret_5d', float('nan')) * 100:+.2f}%** (5d), "
                f"**{feat.get('mkt_ret_20d', float('nan')) * 100:+.2f}%** (20d)")
    st.markdown(f"VIX: **{fmt(vix, 1)}**")

with col2:
    st.markdown("#### Fundamentals")
    st.caption("Informational only — not seen by the model.")
    if not fund:
        st.markdown("_Unavailable_")
    else:
        st.markdown(f"**Sector / industry:** {fund.get('sector', 'n/a')} / {fund.get('industry', 'n/a')}")
        st.markdown(f"**Market cap:** {fmt_money(fund.get('market_cap'))}")
        st.markdown(f"**P/E:** trailing {fmt(fund.get('trailing_pe'))}, forward {fmt(fund.get('forward_pe'))} "
                    f"&nbsp;·&nbsp; **P/B:** {fmt(fund.get('price_to_book'))} "
                    f"&nbsp;·&nbsp; **Beta:** {fmt(fund.get('beta'))}")
        div = fund.get("dividend_yield")
        st.markdown(f"**Dividend yield:** {fmt(div, 2, '%') if div else 'none'}")
        lo, hi = fund.get("fifty_two_week_low"), fund.get("fifty_two_week_high")
        if lo and hi:
            pos = (report["last_close"] - lo) / (hi - lo) * 100
            # Two literal "$" on one line would otherwise get parsed as inline LaTeX math
            # (Streamlit markdown treats matching $...$ pairs as math mode) -- escape both.
            st.markdown(f"**52-week range:** \\${lo:.2f} – \\${hi:.2f} ({pos:.0f}% up the range)")
        target = fund.get("target_mean_price")
        if target:
            upside = (target / report["last_close"] - 1) * 100
            n_analysts = fund.get("num_analyst_opinions")
            st.markdown(f"**Analyst mean target:** ${target:.2f} ({upside:+.1f}% vs. current), "
                        f"consensus **{fund.get('recommendation_key', 'n/a')}**"
                        + (f" ({n_analysts} analysts)" if n_analysts else ""))

st.markdown("#### Recent headlines")
st.caption("Informational only — not seen by the model. yfinance's news feed mixes "
           "stock-specific stories with broader market news, and sentiment is scored "
           "automatically — read headlines yourself too.")
news = report.get("news", {})
articles = news.get("articles", [])
if not articles:
    st.markdown("_No recent headlines found._")
else:
    mean_score = news["mean_score"]
    st.markdown(f"Overall tone across {len(articles)} headlines: **{news['overall_label']}** "
                f"(avg score {mean_score:+.2f}, -1 = very negative, +1 = very positive)")
    sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    for a in articles[:5]:
        emoji = sentiment_emoji.get(a["sentiment_label"], "⚪")
        if a.get("link"):
            st.markdown(f"{emoji} [{a['title']}]({a['link']}) — *{a['publisher']}*  "
                        f"(sentiment: {a['sentiment_score']:+.2f})")
        else:
            st.markdown(f"{emoji} {a['title']} — *{a['publisher']}*  (sentiment: {a['sentiment_score']:+.2f})")

st.markdown("#### Insider activity")
st.caption("Recent SEC Form 4 filings — informational only, not seen by the model. Filing "
           "timing is irregular and there's no clean historical archive, so unlike news this "
           "can't even be backtested as a report-only signal; read it as context, not a rule.")
insiders = report.get("insider_transactions", [])
if not insiders:
    st.markdown("_No recent insider filings found._")
else:
    sales = [t for t in insiders if t["kind"] == "Sale"]
    buys = [t for t in insiders if t["kind"] == "Purchase"]
    other = len(insiders) - len(sales) - len(buys)
    st.markdown(f"Recent filings: **{len(buys)} purchase(s)**, **{len(sales)} sale(s)**, "
                f"{other} other (awards/exercises/etc.)")
    disp = pd.DataFrame(insiders[:8])[["Start Date", "Insider", "Position", "kind", "Shares", "Value"]]
    disp.columns = pd.Index(["Date", "Insider", "Position", "Type", "Shares", "Value ($)"])
    disp["Value ($)"] = disp["Value ($)"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a")
    disp["Shares"] = disp["Shares"].map(lambda s: f"{s:,}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Not financial advice. Models see technical/earnings-timing/macro data only — no news or "
    "fundamentals-depth is part of the prediction, only this page's Fundamentals and Recent "
    "headlines sections. Stock prices are close to a random walk over short horizons; treat "
    "this as one input among several, not a verdict."
)
