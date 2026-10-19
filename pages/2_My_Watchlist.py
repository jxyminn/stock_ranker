"""
My Watchlist: an information browser for the tickers pulled from the user's
own stock screenshots/infographics (config.WATCHLIST_SECTIONS).

No cross-stock ranking or comparison -- each stock gets its own self-
contained panel: fundamentals, technical read, earnings timing, 50/200-day
moving averages, all-time high, and this stock's own model-predicted
forecast per horizon (5d through 200d/~9.5mo). Nothing here sorts or picks
winners across stocks -- for that, see the Ranker page.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HORIZONS, MODELS_DIR, WATCHLIST_SECTIONS
from data_fetcher import fetch_fundamentals, fetch_price_stats
from predict import latest_features_for_ticker, load_models
from features import FEATURE_COLUMNS
from insights import generate_why_bullets
from analyze import fmt_money, fmt

st.set_page_config(
    page_title="My Watchlist · Stock Research",
    page_icon="📋",
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
    padding: 28px 36px; margin-bottom: 24px;
}
.main-header h1 { font-size: 2.1rem; font-weight: 700; color: #58a6ff; margin: 0; }
.main-header p  { color: #8b949e; margin: 6px 0 0; font-size: 0.95rem; }

.stButton > button {
    background: #238636; color: #fff; border: none;
    border-radius: 8px; font-weight: 600; padding: 10px 28px;
    font-size: 0.95rem; transition: background .2s;
}
.stButton > button:hover { background: #2ea043; }

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
.section-chip {
    display: inline-block; background: #21262d; color: #8b949e;
    border-radius: 999px; padding: 3px 11px; font-size: .76rem;
    margin: 2px 4px 2px 0; border: 1px solid #30363d;
}
.section-chip b { color: #58a6ff; }
.section-heading {
    font-size: 1.3rem; font-weight: 700; color: #e6edf3;
    margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


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


def build_stock_info(ticker: str, models: dict | None) -> dict | None:
    """Fundamentals + technical read + long-horizon price stats, plus this
    stock's own per-horizon forecast if trained models are available. No
    cross-stock comparison happens anywhere in this page."""
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


def render_stock_panel(info: dict, sections: list[str]) -> None:
    ticker = info["ticker"]
    feat = info["feat"]
    fund = info["fund"]
    stats = info.get("price_stats") or {}
    predictions = info.get("predictions") or {}
    name = fund.get("long_name") or ticker
    price = feat["_last_close"]
    day_chg = feat.get("ret_1d")
    day_chg_pct = day_chg * 100 if pd.notna(day_chg) else None
    chg_str = f"{day_chg_pct:+.2f}%" if day_chg_pct is not None else "n/a"

    label = f"{ticker} — {name} — ${price:.2f} ({chg_str})"
    with st.expander(label):
        if len(sections) > 1:
            st.caption("In sections: " + ", ".join(sections))

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Your sections")
    st.caption("From your saved stock images. Uncheck a section to leave it out.")
    st.markdown("---")

    chosen: list[str] = []
    ticker_sections: dict[str, list[str]] = {}
    for section_name, tickers in WATCHLIST_SECTIONS.items():
        include = st.checkbox(f"{section_name} ({len(tickers)})", value=True, key=f"mw_{section_name}")
        if include:
            chosen.extend(tickers)
            for t in tickers:
                ticker_sections.setdefault(t, []).append(section_name)
    chosen = list(dict.fromkeys(chosen))

    st.markdown("---")
    filter_text = st.text_input("Filter by ticker", placeholder="e.g. NVDA")

    st.markdown("---")
    st.caption(f"{len(chosen)} unique stock{'s' if len(chosen) != 1 else ''} selected.")
    load_btn = st.button("Load Stock Info", use_container_width=True, disabled=not chosen)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
    <h1>📋 My Watchlist</h1>
    <p>The stocks from your saved screenshots and infographics, grouped into themed
    sections — fundamentals, technical read, moving averages, all-time high, earnings
    timing, and a forecast per horizon for each one. No stock is compared or ranked
    against another; everything here is about that one stock on its own.</p>
</div>""", unsafe_allow_html=True)

st.markdown("".join(
    f'<span class="section-chip"><b>{len(t)}</b> {name}</span>'
    for name, t in WATCHLIST_SECTIONS.items()
), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""<div class="warn-box">
    ⚠️ <b>About the Forecast numbers:</b> the model is only trained on ~50 large, liquid
    US stocks (Apple, Microsoft, etc.) — not on the smaller, more volatile names common in
    sections like Quantum Computing or Space/Robotics. For those, and especially at the
    longer 50-day and 200-day horizons, predicted returns can come out extreme (e.g. triple
    digits) and should be read as a rough directional signal at best, not a number to bank on.
</div>""", unsafe_allow_html=True)

if not load_btn and "watchlist_info" not in st.session_state:
    st.markdown("""<div class="info-box">
        👈 &nbsp;Pick which sections to include in the sidebar (all are selected by default),
        then click <strong>Load Stock Info</strong>. The first load pulls live data for every
        stock, so it can take a little while for a large selection — results are cached for
        an hour after that.
    </div>""", unsafe_allow_html=True)
    st.stop()

if load_btn:
    models, missing_horizons = get_models()
    if missing_horizons:
        st.caption(f"⚠️ No trained model for horizon(s) {missing_horizons} — forecasts will be "
                   f"skipped until you run `python train.py`. Everything else still loads.")

    progress_bar = st.progress(0, text="Starting up…")
    info_map: dict[str, dict] = {}
    skipped: list[str] = []
    for i, t in enumerate(chosen):
        pct = int(100 * (i + 1) / len(chosen))
        progress_bar.progress(pct, text=f"Pulling data for {t}… ({i + 1} of {len(chosen)})")
        info = build_stock_info(t, models)
        if info is not None:
            info_map[t] = info
        else:
            skipped.append(t)
    progress_bar.empty()

    st.session_state["watchlist_info"] = info_map
    st.session_state["watchlist_sections_map"] = ticker_sections
    st.session_state["watchlist_skipped"] = skipped

info_map = st.session_state.get("watchlist_info", {})
sec_map = st.session_state.get("watchlist_sections_map", ticker_sections)
skipped = st.session_state.get("watchlist_skipped", [])

if not info_map:
    st.warning("No data could be retrieved for your selection. Try again or check your symbols.")
    st.stop()

if skipped:
    st.caption(f"⚠️ Couldn't pull data for {len(skipped)} symbol(s): {', '.join(skipped)}")

rcol, _ = st.columns([1, 4])
with rcol:
    if st.button("Refresh data"):
        for k in ("watchlist_info", "watchlist_skipped"):
            st.session_state.pop(k, None)
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

shown_any = False
for section_name, tickers in WATCHLIST_SECTIONS.items():
    if not st.session_state.get(f"mw_{section_name}", True):
        continue
    section_tickers = [t for t in tickers if t in info_map]
    if filter_text:
        section_tickers = [t for t in section_tickers if filter_text.strip().upper() in t]
    if not section_tickers:
        continue
    shown_any = True
    st.markdown(f'<div class="section-heading">{section_name} ({len(section_tickers)})</div>',
                unsafe_allow_html=True)
    for t in section_tickers:
        render_stock_panel(info_map[t], sec_map.get(t, [section_name]))

if not shown_any:
    st.info("No stocks match your current filter.")

st.markdown("---")
st.caption(
    "Fundamentals and technical reads are pulled fresh from live data — not model predictions. "
    "Not financial advice; always do your own research."
)
