import sys
from pathlib import Path

# Every module in this project uses bare `from config import ...`-style
# imports that assume the repo root is on sys.path (true when scripts are run
# directly from the repo root). Make that true for pytest too, regardless of
# where it's invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 400, seed: int = 42, start: str = "2024-01-02",
                drift: float = 0.0003, vol: float = 0.015) -> pd.DataFrame:
    """
    Deterministic synthetic daily OHLCV, tz-aware (America/New_York) like
    yfinance's real output, with enough rows to fill every rolling window
    features.py uses (252-day 52w high/low included).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n, tz="America/New_York")
    daily_ret = rng.normal(drift, vol, size=n)
    close = 100 * np.cumprod(1 + daily_ret)

    intraday = rng.uniform(0.001, 0.01, size=n)
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=n)
    volume = rng.lognormal(mean=15, sigma=0.3, size=n).astype(int)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def ohlcv():
    return make_ohlcv()
