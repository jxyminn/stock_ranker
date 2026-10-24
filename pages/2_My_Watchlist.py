"""
My Watchlist: an information browser for the tickers pulled from the user's
own stock screenshots/infographics (config.WATCHLIST_SECTIONS).

No cross-stock ranking or comparison -- each stock gets its own self-
contained panel: price chart, fundamentals, technical read, earnings timing,
50/200-day moving averages, all-time high, and this stock's own model-
predicted forecast per horizon (5d through 200d/~9.5mo). Nothing here sorts
or picks winners across stocks -- for that, see the Ranker page.

Charts are loaded on demand (a per-stock toggle) rather than all at once --
with 200+ tickers, building and rendering every chart up front would make
the page very slow to load.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_info import get_models, build_stock_info, render_stock_panel
from watchlist_store import add_ticker, get_effective_sections

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Your sections")

    with st.expander("➕ Add a stock", expanded=False):
        new_query = st.text_input("Ticker or company name", key="add_ticker_input",
                                   placeholder="e.g. AMAT or 'Applied Materials'")
        if st.button("Add to watchlist", key="add_ticker_btn", use_container_width=True):
            if not new_query.strip():
                st.warning("Type a ticker or company name first.")
            else:
                with st.spinner(f"Looking up '{new_query}'…"):
                    result = add_ticker(new_query)
                if result["ok"]:
                    st.success(result["message"])
                    st.session_state.pop("watchlist_info", None)
                    if result["persisted"]:
                        st.caption("Refresh the page in a minute or two to see it everywhere "
                                   "once Streamlit Cloud redeploys.")
                else:
                    st.error(result["message"])

    st.caption("From your saved stock images. Uncheck a section to leave it out.")
    st.markdown("---")

    chosen: list[str] = []
    ticker_sections: dict[str, list[str]] = {}
    watchlist_sections = get_effective_sections()
    for section_name, tickers in watchlist_sections.items():
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
    for name, t in watchlist_sections.items()
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
for section_name, tickers in watchlist_sections.items():
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
