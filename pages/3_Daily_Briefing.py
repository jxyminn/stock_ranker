"""
Daily Briefing: shows last night's automated run (nightly_briefing.py),
which runs on a schedule via .github/workflows/nightly.yml and commits its
output to briefing/latest.json. This page only reads that file -- it never
calls yfinance or runs models itself, so it loads instantly.

This is a summary digest of what changed overnight, not buy/sell/hold
instructions -- see the disclaimer at the bottom.

Rendered as sortable tables rather than one card of prose per ticker --
with 280+ tickers, a stock that triggered six separate forecast-move notes
(one per horizon) used to mean eleven lines of text to read just for that
one ticker. A table lets you sort by magnitude and scan the whole night at
once instead.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIEFING_DIR
from reliability import load_backtest_results

st.set_page_config(
    page_title="Daily Briefing · Stock Research",
    page_icon="🌙",
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
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.metric-card .label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; margin-top: 4px; }
.metric-card .sub   { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

CATEGORIES = [
    ("📅 Earnings this week", lambda n: "earnings in" in n),
    ("📈 Big forecast moves", lambda n: "forecast moved" in n or "chance-of-gain moved" in n),
    ("🔀 Trend crossovers", lambda n: "cross" in n),
    ("🚀 Near an all-time high", lambda n: "all-time high" in n),
    ("📰 Notably one-sided news", lambda n: "headlines" in n),
]

# Pulls the sortable number back out of nightly_briefing.py's note strings
# (ticker/horizon/magnitude), so the page can sort by "biggest" instead of
# just listing notes in whatever order they happened to be generated.
_NOTE_PATTERNS = [
    (re.compile(r"^(\d+)d forecast moved ([+-]?[\d.]+)pp overnight \(now ([+-]?[\d.]+)%\)$"),
     lambda m: {"Horizon": int(m.group(1)), "Signal": "Forecast return",
                "Change": f"{float(m.group(2)):+.1f}pp", "New value": f"{float(m.group(3)):+.1f}%",
                "_magnitude": abs(float(m.group(2)))}),
    (re.compile(r"^(\d+)d chance-of-gain moved ([+-]?[\d.]+)pp overnight \(now (\d+)%\)$"),
     lambda m: {"Horizon": int(m.group(1)), "Signal": "Chance of gain",
                "Change": f"{float(m.group(2)):+.0f}pp", "New value": f"{m.group(3)}%",
                "_magnitude": abs(float(m.group(2)))}),
    (re.compile(r"^earnings in (\d+) days?$"),
     lambda m: {"Days until": int(m.group(1)), "_magnitude": -int(m.group(1))}),
    (re.compile(r"^within ([\d.]+)% of its all-time high$"),
     lambda m: {"% from ATH": f"{float(m.group(1)):.1f}%", "_magnitude": -float(m.group(1))}),
    (re.compile(r"^notably (positive|negative) headlines \(score ([+-][\d.]+)\)$"),
     lambda m: {"Tone": m.group(1).capitalize(), "Score": m.group(2),
                "_magnitude": abs(float(m.group(2)))}),
    (re.compile(r"^(golden|death) cross \((bullish|bearish)\)$"),
     lambda m: {"Signal": f"{m.group(1).capitalize()} cross", "Read": m.group(2).capitalize(),
                "_magnitude": 0}),
]


def parse_note(note: str) -> dict:
    """Structured fields for a note string, or just {"Detail": note} if it
    doesn't match a known pattern (keeps the table useful even for a note
    format this page doesn't specifically know how to break apart)."""
    for pattern, builder in _NOTE_PATTERNS:
        m = pattern.match(note)
        if m:
            return builder(m)
    return {"Detail": note, "_magnitude": 0}


def build_table(items: list[dict], matcher) -> pd.DataFrame:
    """One row per matching note (so a ticker with 6 flagged horizons gets
    6 rows, not 1 row crammed with 6 lines of text), sorted by magnitude
    descending so the biggest moves are at the top by default -- click any
    column header to re-sort."""
    rows = []
    for item in items:
        for note in item["notes"]:
            if not matcher(note):
                continue
            parsed = parse_note(note)
            chg = item.get("day_chg_pct")
            rows.append({
                "Ticker": item["ticker"],
                "Company": item["name"],
                "Price": item["price"],
                "Today": chg,
                **{k: v for k, v in parsed.items() if k != "_magnitude"},
                "_magnitude": parsed["_magnitude"],
            })
    df = pd.DataFrame(rows).sort_values("_magnitude", ascending=False).drop(columns="_magnitude")
    df["Price"] = df["Price"].map(lambda v: f"${v:.2f}")
    df["Today"] = df["Today"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    return df.reset_index(drop=True)


def load_briefing() -> dict | None:
    path = BRIEFING_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


st.markdown("""<div class="main-header">
    <h1>🌙 Daily Briefing</h1>
    <p>What changed overnight across your watchlist, generated automatically once a day.
    A summary of notable movement — not a list of things to buy or sell.</p>
</div>""", unsafe_allow_html=True)

briefing = load_briefing()

if briefing is None:
    st.markdown("""<div class="info-box">
        No briefing has been generated yet. It runs automatically every night via GitHub
        Actions once this is deployed — or run <code>python nightly_briefing.py</code>
        locally to generate the first one now.
    </div>""", unsafe_allow_html=True)
    st.stop()

generated_at = datetime.fromisoformat(briefing["generated_at"])
age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600

c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="metric-card">
    <div class="label">Generated</div>
    <div class="value" style="font-size:1.1rem">{generated_at.strftime('%b %d, %Y %H:%M UTC')}</div>
    <div class="sub">{age_hours:.0f} hours ago</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card">
    <div class="label">Stocks Covered</div>
    <div class="value">{briefing['tickers_covered']}</div>
    <div class="sub">{len(briefing.get('tickers_skipped', []))} skipped</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card">
    <div class="label">Notable Tonight</div>
    <div class="value">{len(briefing['notable'])}</div>
    <div class="sub">out of {briefing['tickers_covered']} covered</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if age_hours > 30:
    st.markdown(f"""<div class="warn-box">
        ⚠️ This briefing is {age_hours:.0f} hours old — the nightly job may not be running.
        Check the Actions tab on GitHub, or run <code>python nightly_briefing.py</code> manually.
    </div>""", unsafe_allow_html=True)

if not briefing.get("models_available", True):
    st.markdown("""<div class="warn-box">
        ⚠️ Trained models weren't found when this briefing ran, so forecast-based notes
        (forecast moves, chance-of-gain moves) are missing tonight. Run <code>python train.py</code>.
    </div>""", unsafe_allow_html=True)

if briefing.get("is_first_run"):
    st.markdown("""<div class="info-box">
        This is the first briefing on record, so there's nothing yet to compare against —
        tomorrow's run will start flagging overnight changes.
    </div>""", unsafe_allow_html=True)

HORIZON_LABELS = {5: "1 week", 10: "2 weeks", 15: "3 weeks", 30: "1 month",
                   50: "~2.5 months", 200: "~9.5 months"}


tab_notable, tab_backtest, tab_track = st.tabs(
    [f"🔔 Notable Tonight ({len(briefing.get('notable', []))})",
     "📚 Historical Backtest", "📊 Live Track Record"]
)

with tab_notable:
    notable = briefing.get("notable", [])
    if not notable:
        st.info("Nothing notable overnight across your watchlist.")
    else:
        forecast_matcher = CATEGORIES[1][1]  # "Big forecast moves"
        top_moves = build_table(notable, forecast_matcher).head(15)
        if not top_moves.empty:
            st.markdown("##### 🔥 Biggest moves tonight")
            st.caption("The 15 largest forecast/chance-of-gain swings across the whole watchlist, "
                       "regardless of category — start here.")
            st.dataframe(top_moves, use_container_width=True, hide_index=True)
            st.markdown("---")

        used = set()
        for heading, matcher in CATEGORIES:
            items = [n for n in notable if any(matcher(note) for note in n["notes"])]
            if not items:
                continue
            used.update(item["ticker"] for item in items)
            with st.expander(f"{heading} ({len(items)})"):
                st.dataframe(build_table(items, matcher), use_container_width=True, hide_index=True)

        leftover = [n for n in notable if n["ticker"] not in used]
        if leftover:
            with st.expander(f"📌 Other notable changes ({len(leftover)})"):
                st.dataframe(build_table(leftover, lambda n: True), use_container_width=True, hide_index=True)

with tab_backtest:
    st.caption(
        "From walk-forward testing on years of historical data across the full training "
        "universe — a much bigger sample than the live tracker can build up on its own, "
        "but it's backward-looking: how the model performed on the past, not a guarantee "
        "about what's ahead."
    )
    backtest_results = load_backtest_results()
    if not backtest_results:
        st.info("No backtest on record yet — run `python backtest.py`.")
    else:
        run_at = backtest_results["run_at"][:10]
        rows = []
        for h_str, stats in backtest_results["horizons"].items():
            h = int(h_str)
            rows.append({
                "Horizon": HORIZON_LABELS.get(h, f"{h}d"),
                "Mean IC": f"{stats['mean_ic']:+.3f}",
                "AUC": f"{stats['mean_auc']:.3f}",
                "% days IC > 0": f"{stats['pct_days_positive_ic'] * 100:.0f}%",
                "Folds": stats["n_folds"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Run {run_at} on {backtest_results['universe_size']} tickers, "
                   f"{backtest_results['folds']} walk-forward folds. AUC 0.5 = coin flip; "
                   f"IC near 0 means the ranking has little real signal that horizon.")

with tab_track:
    st.caption(
        "Every prediction made for every stock gets logged and checked again once that "
        "horizon's time window has passed — not just the ones flagged as notable, since "
        "only tracking flagged ones would bias this toward stocks already in motion. This "
        "is how the model has actually done in live use, which can drift from the backtest "
        "as market conditions change."
    )
    track_record = briefing.get("track_record", {})
    n_pending = briefing.get("track_record_pending", 0)

    if not track_record:
        st.info(f"No predictions have had enough time to resolve yet "
                 f"({n_pending} pending). Check back as the briefing accumulates history.")
    else:
        rows = []
        for h_str, stats in track_record.items():
            h = int(h_str)
            rows.append({
                "Horizon": HORIZON_LABELS.get(h, f"{h}d"),
                "Resolved": stats["n_resolved"],
                "Direction correct": f"{stats['direction_accuracy'] * 100:.0f}%",
                "Chance-of-gain correct": f"{stats['classifier_accuracy'] * 100:.0f}%",
                "Avg error (pp)": stats["mean_abs_error_pp"],
                "Avg predicted": f"{stats['mean_predicted_pct']:+.1f}%",
                "Avg actual": f"{stats['mean_actual_pct']:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"**Direction correct**: did the stock actually move the same way (up/down) the "
            f"forecast said? **Chance-of-gain correct**: did the \"chance of a gain\" call end "
            f"up on the right side of 50%? **{n_pending}** predictions are still waiting for "
            f"their horizon to pass. Small sample sizes early on (especially for 50-/200-day "
            f"horizons, which take months to resolve) will be noisy — a handful of resolved "
            f"predictions doesn't prove much either way yet."
        )

st.markdown("---")
st.caption(
    "This is an automated summary of what changed overnight — not a recommendation to buy, "
    "sell, or hold anything. Forecasts come from models trained on historical patterns and "
    "are noisy, especially for smaller/more volatile names. For the full picture on any "
    "stock here, use Stock Research or My Watchlist. Not financial advice."
)
