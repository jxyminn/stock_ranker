import numpy as np
import pandas as pd
import pytest

from labeling import build_labels


def test_forward_return_matches_manual_calc():
    close = pd.Series([100.0, 110.0, 121.0, 108.9, 130.0], index=pd.RangeIndex(5))
    labels = build_labels(close, horizon=2)

    # row 0: close[2]/close[0] - 1 = 121/100 - 1 = 0.21
    assert labels["fwd_return_2"].iloc[0] == pytest.approx(0.21)
    # row 1: close[3]/close[1] - 1 = 108.9/110 - 1 = -0.01
    assert labels["fwd_return_2"].iloc[1] == pytest.approx(-0.01)


def test_last_horizon_rows_are_nan_no_lookahead():
    close = pd.Series(np.linspace(100, 200, 50))
    horizon = 7
    labels = build_labels(close, horizon)

    # the last `horizon` rows have no future price available, so must be NaN --
    # this is the no-lookahead guarantee train.py's dropna relies on.
    assert labels["fwd_return_7"].iloc[-horizon:].isna().all()
    assert labels["up_7"].iloc[-horizon:].isna().all()
    # every row before that must be populated
    assert labels["fwd_return_7"].iloc[:-horizon].notna().all()


def test_up_flag_matches_sign_of_forward_return():
    close = pd.Series([100.0, 90.0, 130.0, 95.0, 150.0, 149.0, 200.0])
    labels = build_labels(close, horizon=1)
    valid = labels.dropna()

    assert (valid["up_1"] == (valid["fwd_return_1"] > 0).astype(float)).all()
