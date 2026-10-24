"""
Shared single-stock info panel: fundamentals, technical read, earnings
timing, 50/200-day moving averages, all-time high, this stock's own
model-predicted forecast per horizon, and an on-demand price chart.

Pulled out of My Watchlist into its own module (rather than copied, the
way each page's CSS is) because the Daily Briefing page needs the exact
same panel for its "click a ticker to see its info" feature -- this is
substantial logic, not a cheap-to-duplicate style block.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import HORIZONS, MODELS_DIR
from data_fetcher import fetch_fundamentals, fetch_price_stats, fetch_earnings
from predict import latest_features_for_ticker, load_models
from features import FEATURE_COLUMNS
from insights import generate_why_bullets
from analyze import fmt_money, fmt, build_price_history

HORIZON_LABELS = {
    5: "1 week", 10: "2 weeks", 15: "3 weeks", 30: "1 month",
    50: "~2.5 months", 200: "~9.5 months",
}


@st.cache_resource(show_spinner=False)
def get_models():
    missing = [h for h in HORIZONS if not (MODELS_DIR / f"horizon_{h}.joblib").exists()]
    if missing:
        return None, missing
    return load_models(), []


def build_price_chart(price_df: pd.DataFrame, earnings_df: pd.DataFrame | None) -> go.Figure:
    """Price + SMA20/50 + Bollinger Bands on top, volume below, earnings dates marked."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03,
    )

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
        margin=dict(l=0, r=0, t=70, b=0),
        height=380,
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                y=1.3,
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


def build_stock_info(ticker: str, models: dict | None) -> dict | None:
    """Fundamentals + technical read + long-horizon price stats, plus this
    stock's own per-horizon forecast if trained models are available. No
    cross-stock comparison happens anywhere in this panel."""
    feat = latest_features_for_ticker(ticker)
    if feat is None:
        return None
    fund = fetch_fundamentals(ticker)
    price_stats = fetch_price_stats(ticker)

    predictions = {}
    if models is not None:
        X = feat[FEATURE_COLUMNS].astype(float).to_frame().T
        for h in HORIZONS:
            reg = models[h]["regressor"]
            clf = models[h]["classifier"]
            pred_return = float(reg.predict(X)[0])
            predictions[h] = {
                "pred_return_pct": pred_return * 100,
                "prob_up": float(clf.predict_proba(X)[0, 1]),
            }

    return {"ticker": ticker, "feat": feat, "fund": fund, "price_stats": price_stats,
            "predictions": predictions}


def render_stock_panel(info: dict, sections: list[str] | None = None, key_prefix: str = "",
                        expanded: bool = False) -> None:
    """Collapsed-by-default expander (ticker — name — price (day change))
    containing the full info panel. `key_prefix` keeps Streamlit widget
    keys unique when this is called for the same ticker from more than one
    place on a page (e.g. a ticker appearing in both the "biggest moves"
    table and a category table on the Daily Briefing page)."""
    fund = info["fund"]
    feat = info["feat"]
    name = fund.get("long_name") or info["ticker"]
    price = feat["_last_close"]
    day_chg = feat.get("ret_1d")
    chg_str = f"{day_chg * 100:+.2f}%" if pd.notna(day_chg) else "n/a"
    label = f"{info['ticker']} — {name} — ${price:.2f} ({chg_str})"
    with st.expander(label, expanded=expanded):
        _render_stock_panel_content(info, sections, key_prefix)


def _render_stock_panel_content(info: dict, sections: list[str] | None, key_prefix: str) -> None:
    ticker = info["ticker"]
    feat = info["feat"]
    fund = info["fund"]
    stats = info.get("price_stats") or {}
    predictions = info.get("predictions") or {}
    price = feat["_last_close"]

    if sections:
        st.caption("In sections: " + ", ".join(sections))

    if st.checkbox("📈 Show price chart", key=f"{key_prefix}show_chart_{ticker}"):
        cache_key = f"chart_data_{ticker}"
        if cache_key not in st.session_state:
            with st.spinner(f"Loading chart for {ticker}…"):
                st.session_state[cache_key] = (build_price_history(ticker), fetch_earnings(ticker))
        price_df, earnings_df = st.session_state[cache_key]
        if price_df is not None and not price_df.empty:
            st.plotly_chart(build_price_chart(price_df, earnings_df), use_container_width=True,
                             theme=None, key=f"{key_prefix}chart_{ticker}")
            st.caption("Dashed vertical lines mark earnings reports — green if that report beat "
                       "estimates, red if it missed.")
        else:
            st.caption("No price history available for a chart.")
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fundamentals**")
        if not fund:
            st.markdown("_Unavailable_")
        else:
            st.markdown(f"Sector / industry: {fund.get('sector', 'n/a')} / {fund.get('industry', 'n/a')}")
            st.markdown(f"Market cap: {fmt_money(fund.get('market_cap'))}")
            st.markdown(f"P/E: trailing {fmt(fund.get('trailing_pe'))}, forward {fmt(fund.get('forward_pe'))}")
            st.markdown(f"P/B: {fmt(fund.get('price_to_book'))} · Beta: {fmt(fund.get('beta'))}")
            div = fund.get("dividend_yield")
            st.markdown(f"Dividend yield: {fmt(div, 2, '%') if div else 'none'}")
            lo, hi = fund.get("fifty_two_week_low"), fund.get("fifty_two_week_high")
            if lo and hi:
                pos = (price - lo) / (hi - lo) * 100
                st.markdown(f"52-week range: \\${lo:.2f} – \\${hi:.2f} ({pos:.0f}% up the range)")
            target = fund.get("target_mean_price")
            if target:
                upside = (target / price - 1) * 100
                n = fund.get("num_analyst_opinions")
                st.markdown(f"Analyst mean target: \\${target:.2f} ({upside:+.1f}% vs. current), "
                            f"consensus **{fund.get('recommendation_key', 'n/a')}**"
                            + (f" ({n} analysts)" if n else ""))

        st.markdown("**Moving averages & all-time high**")
        if not stats:
            st.markdown("_Unavailable_")
        else:
            sma_50, sma_200 = stats.get("sma_50"), stats.get("sma_200")
            if sma_50 is not None:
                rel = "above" if price > sma_50 else "below"
                st.markdown(f"50-day average: **${sma_50:.2f}** (price is {abs(price/sma_50-1)*100:.1f}% {rel} it)")
            if sma_200 is not None:
                rel = "above" if price > sma_200 else "below"
                st.markdown(f"200-day average: **${sma_200:.2f}** (price is {abs(price/sma_200-1)*100:.1f}% {rel} it)")
            if sma_50 is not None and sma_200 is not None:
                cross = "bullish — golden cross (50d above 200d)" if sma_50 > sma_200 else "bearish — death cross (50d below 200d)"
                st.markdown(f"Trend signal: **{cross}**")
            ath = stats.get("all_time_high")
            if ath:
                below_ath = (1 - price / ath) * 100
                st.markdown(f"All-time high: **${ath:.2f}** (set {stats.get('all_time_high_date', 'n/a')}, "
                            f"currently {below_ath:.1f}% below it)" if below_ath > 0.05
                            else f"All-time high: **${ath:.2f}** — at or near an all-time high right now 🚀")

    with col2:
        st.markdown("**Technical read**")
        for bullet in generate_why_bullets(feat, 0.0, 0.5):
            st.markdown(f"- {bullet}")
        st.markdown(f"- Volatility: **{feat['vol_20d'] * (252 ** 0.5) * 100:.1f}%** annualized (20d)")

        st.markdown("**Earnings**")
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
            note = " ⚠️ within a week" if days_until <= 7 else ""
            st.markdown(f"Next report in **{days_until:.0f} days**{note}")

    st.markdown("**Forecast**")
    st.caption("This stock's own model prediction per horizon — not compared against other stocks.")
    if not predictions:
        st.markdown("_Models aren't trained yet — run `python train.py` to enable forecasts._")
    else:
        rows = [{
            "Horizon": HORIZON_LABELS[h],
            "Predicted return": f"{predictions[h]['pred_return_pct']:+.2f}%",
            "Chance of a gain": f"{predictions[h]['prob_up'] * 100:.0f}%",
        } for h in HORIZONS]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
