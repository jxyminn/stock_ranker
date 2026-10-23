"""
Portfolio: positions you actually own, entered by hand (ticker, shares,
cost basis). Shows real unrealized P&L next to each position and in total --
the one thing none of the other pages do, since everything else here is
watchlist/research, not "what do I actually hold."

No brokerage API integration -- see portfolio_store.py's docstring for why.
Data is stored in a separate PRIVATE repo (not the public stock_ranker repo)
since share counts and cost basis reveal real dollar amounts.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetcher import fetch_fundamentals, fetch_history, search_ticker
from portfolio_store import add_position, load_portfolio, remove_position

st.set_page_config(
    page_title="Portfolio · Stock Research",
    page_icon="💼",
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

.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.metric-card .label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; margin-top: 4px; }
.metric-card .sub   { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }

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
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="main-header">
    <h1>💼 Portfolio</h1>
    <p>Positions you actually own, entered by hand — real unrealized P&L next to each one.
    No other page here knows what you hold; this is the one that does.</p>
</div>""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _current_price(ticker: str) -> float | None:
    df = fetch_history(ticker, lookback_days=10, use_cache=True)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


# ── Sidebar: add a position ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Add a position")
    query = st.text_input("Ticker or company name", placeholder="e.g. AAPL or 'Apple'")
    resolved_ticker = None
    if query:
        candidate = query.strip().upper()
        if _current_price(candidate) is not None:
            resolved_ticker = candidate
        else:
            hits = search_ticker(query, max_results=5)
            equity_hits = [h for h in hits if h.get("symbol")]
            if equity_hits:
                resolved_ticker = st.selectbox(
                    "Did you mean", [h["symbol"] for h in equity_hits],
                    format_func=lambda s: f"{s} — {next(h['name'] for h in equity_hits if h['symbol'] == s)}",
                )
            else:
                st.caption("No matching stock found.")

    shares = st.number_input("Shares", min_value=0.0, step=1.0, format="%.4f")
    cost_basis = st.number_input("Cost basis per share ($)", min_value=0.0, step=0.01, format="%.2f")
    purchase_date = st.date_input("Purchase date (optional)", value=None, max_value=date.today())
    notes = st.text_input("Notes (optional)", placeholder="e.g. Roth IRA")

    if st.button("Add Position", use_container_width=True, disabled=not resolved_ticker):
        with st.spinner("Saving…"):
            result = add_position(
                resolved_ticker, shares, cost_basis,
                purchase_date.isoformat() if purchase_date else None, notes,
            )
        if result["ok"]:
            st.success(result["message"])
            if not result["persisted"]:
                st.caption("⚠️ Not saved to the private repo — see portfolio_store.py's docstring.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(result["message"])

    st.markdown("---")
    st.caption("Stored in a separate **private** GitHub repo — never the public one, since this "
               "reveals real dollar amounts.")


# ── Main area ────────────────────────────────────────────────────────────
portfolio = load_portfolio()
positions = portfolio.get("positions", [])

if not positions:
    st.markdown("""<div class="info-box">
        👈 Add your first position in the sidebar — ticker, how many shares, and what you paid
        per share.
    </div>""", unsafe_allow_html=True)
    st.stop()

rows = []
missing_price = []
for p in positions:
    price = _current_price(p["ticker"])
    if price is None:
        missing_price.append(p["ticker"])
        continue
    cost_total = p["shares"] * p["cost_basis_per_share"]
    value = p["shares"] * price
    pnl = value - cost_total
    pnl_pct = (pnl / cost_total * 100) if cost_total > 0 else 0.0
    fund = fetch_fundamentals(p["ticker"])
    rows.append({
        "id": p["id"],
        "Ticker": p["ticker"],
        "Company": fund.get("long_name") or p["ticker"],
        "Shares": p["shares"],
        "Cost/Share": p["cost_basis_per_share"],
        "Current Price": price,
        "Cost Basis": cost_total,
        "Current Value": value,
        "P&L ($)": pnl,
        "P&L (%)": pnl_pct,
        "Purchased": p.get("purchase_date") or "—",
        "Notes": p.get("notes") or "",
    })

if missing_price:
    st.warning(f"Couldn't get a current price for: {', '.join(missing_price)} (shown positions exclude these).")

if not rows:
    st.stop()

df = pd.DataFrame(rows)
total_cost = df["Cost Basis"].sum()
total_value = df["Current Value"].sum()
total_pnl = total_value - total_cost
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="metric-card">
    <div class="label">Total Invested</div>
    <div class="value">${total_cost:,.2f}</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card">
    <div class="label">Current Value</div>
    <div class="value">${total_value:,.2f}</div>
</div>""", unsafe_allow_html=True)
pnl_color = "#3fb950" if total_pnl >= 0 else "#f85149"
c3.markdown(f"""<div class="metric-card">
    <div class="label">Unrealized P&L</div>
    <div class="value" style="color:{pnl_color}">{total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

disp = df.drop(columns=["id"]).copy()
disp["Cost/Share"] = disp["Cost/Share"].map(lambda v: f"${v:,.2f}")
disp["Current Price"] = disp["Current Price"].map(lambda v: f"${v:,.2f}")
disp["Cost Basis"] = disp["Cost Basis"].map(lambda v: f"${v:,.2f}")
disp["Current Value"] = disp["Current Value"].map(lambda v: f"${v:,.2f}")
disp["P&L ($)"] = disp["P&L ($)"].map(lambda v: f"{v:+,.2f}")
disp["P&L (%)"] = disp["P&L (%)"].map(lambda v: f"{v:+.1f}%")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.markdown("#### Remove a position")
remove_id = st.selectbox(
    "Position", df["id"],
    format_func=lambda i: f"{df.loc[df['id'] == i, 'Ticker'].iloc[0]} — "
                           f"{df.loc[df['id'] == i, 'Shares'].iloc[0]:g} shares "
                           f"@ ${df.loc[df['id'] == i, 'Cost/Share'].iloc[0]:.2f}",
)
if st.button("Remove", type="secondary"):
    result = remove_position(remove_id)
    if result["ok"]:
        st.success(result["message"])
        st.cache_data.clear()
        st.rerun()
    else:
        st.error(result["message"])

st.markdown("---")
st.caption(
    "Prices are live quotes, refreshed every 5 minutes. Not financial advice — this page just "
    "tracks what you've told it you own; it doesn't validate against a real brokerage account."
)
