"""
Label construction for training. Labels look FORWARD in time (future return),
so this module is only ever used on historical data during training -- never
at live prediction time (there is no future yet).
"""
import pandas as pd


def build_labels(close: pd.Series, horizon: int) -> pd.DataFrame:
    """
    For each row t, compute the forward return from t to t+horizon.
    Returns a dataframe with:
      fwd_return_{h}  -- continuous forward % return (regression target)
      up_{h}          -- 1 if forward return > 0 else 0 (classification target)
    The last `horizon` rows will be NaN (no future data available) -- drop them
    before training.
    """
    fwd_price = close.shift(-horizon)
    fwd_return = fwd_price / close - 1
    # `fwd_return > 0` evaluates NaN > 0 as False, so a plain .astype("float")
    # would silently label the trailing no-future-data rows as "down" (0.0)
    # instead of NaN -- explicitly re-mask them to honor the no-lookahead
    # guarantee this function documents.
    up = fwd_return.gt(0).astype("float")
    up[fwd_return.isna()] = float("nan")
    return pd.DataFrame({
        f"fwd_return_{horizon}": fwd_return,
        f"up_{horizon}": up,
    })
