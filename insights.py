"""
Turn raw technical features into plain-English read-outs. Shared by app.py
(Streamlit "why" bullets on the ranking table) and analyze.py (single-stock
deep-dive report) so both describe the same underlying features the same way.
"""
import pandas as pd


def _fv(feat: pd.Series, key: str, default: float = 0.0) -> float:
    """Safe float accessor for a feature Series -- avoids Pyright None complaints."""
    v = feat.get(key)
    return float(v) if v is not None else default


def generate_why_bullets(feat: pd.Series, pred_return: float, prob_up: float) -> list[str]:
    """Turn raw technical features into plain-English bullet points."""
    bullets: list[str] = []

    # Momentum — RSI
    rsi = _fv(feat, "rsi_14", 50)
    if rsi >= 70:
        bullets.append(f"RSI {rsi:.0f} — strong momentum but getting stretched (may be overbought)")
    elif rsi >= 55:
        bullets.append(f"RSI {rsi:.0f} — healthy upward momentum")
    elif rsi <= 30:
        bullets.append(f"RSI {rsi:.0f} — potentially oversold; price may be due for a bounce")
    elif rsi <= 45:
        bullets.append(f"RSI {rsi:.0f} — weak momentum; price has been under pressure")

    # Trend vs 50-day average
    sma50_pct = _fv(feat, "sma_50_ratio") * 100
    if sma50_pct > 8:
        bullets.append(f"Price is {sma50_pct:.1f}% above its 50-day average — strong uptrend")
    elif sma50_pct > 2:
        bullets.append(f"Price is {sma50_pct:.1f}% above its 50-day average — trending higher")
    elif sma50_pct >= 0:
        bullets.append("Price is just above its 50-day average — mild upward lean")
    elif sma50_pct > -5:
        bullets.append(f"Price has slipped {abs(sma50_pct):.1f}% below its 50-day average")
    else:
        bullets.append(f"Price is {abs(sma50_pct):.1f}% below its 50-day average — in a downtrend")

    # MACD histogram (short-term vs long-term momentum)
    macd_hist = _fv(feat, "macd_hist")
    if macd_hist > 0:
        bullets.append("MACD is positive — short-term momentum is outpacing the longer-term trend")
    else:
        bullets.append("MACD is negative — short-term momentum is lagging the longer-term trend")

    # Volume vs normal
    vol_ratio = _fv(feat, "volume_ratio_20d", 1.0)
    if vol_ratio >= 1.5:
        bullets.append(f"Trading at {vol_ratio:.1f}× its normal volume — elevated investor interest")
    elif vol_ratio >= 1.2:
        bullets.append(f"Volume is {vol_ratio:.1f}× above average — more activity than usual")
    elif vol_ratio < 0.7:
        bullets.append("Volume is below normal — limited trading interest right now")

    # Recent 20-day return
    ret_20d = _fv(feat, "ret_20d") * 100
    if ret_20d > 5:
        bullets.append(f"Already up {ret_20d:.1f}% this past month — strong recent run")
    elif ret_20d > 0:
        bullets.append(f"Up {ret_20d:.1f}% over the past month — positive recent trend")
    elif ret_20d < -5:
        bullets.append(f"Down {abs(ret_20d):.1f}% this past month — recent weakness")

    # 52-week position
    dist_high = _fv(feat, "dist_from_52w_high", -0.5) * 100
    dist_low = _fv(feat, "dist_from_52w_low", 0.5) * 100
    if dist_high >= -4:
        bullets.append(f"Near its 52-week high (only {abs(dist_high):.1f}% away) — price strength")
    elif dist_low <= 15:
        bullets.append("Near its 52-week low — could be a value entry or a continued slide")

    # Bollinger Band position
    bb = _fv(feat, "bb_pct_b", 0.5)
    if bb > 0.9:
        bullets.append("Near the top of its Bollinger Band — price may be extended short-term")
    elif bb < 0.1:
        bullets.append("Near the bottom of its Bollinger Band — price may be oversold short-term")

    return bullets[:4]
