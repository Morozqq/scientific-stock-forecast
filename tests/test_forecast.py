import numpy as np
import pandas as pd

from src.forecast import build_features


def test_build_features_creates_expected_columns():
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = pd.DataFrame(
        {"Close": np.linspace(100, 120, len(index)),
         "Volume": np.linspace(1_000_000, 1_200_000, len(index))},
        index=index,
    )
    result = build_features(prices)
    expected = {
        "return_1d", "return_5d", "return_10d", "volatility_5d",
        "volatility_10d", "volume_change_1d", "close_lag_1",
        "close_lag_2", "close_lag_3", "close_lag_5", "target_next_close",
    }
    assert expected.issubset(result.columns)


def test_target_is_next_session_close():
    index = pd.date_range("2024-01-01", periods=15, freq="B")
    prices = pd.DataFrame(
        {"Close": np.arange(100.0, 115.0), "Volume": np.full(15, 1000)},
        index=index,
    )
    result = build_features(prices)
    assert result.loc[index[5], "target_next_close"] == prices.loc[index[6], "Close"]


def test_features_do_not_use_future_close_as_input():
    index = pd.date_range("2024-01-01", periods=20, freq="B")
    prices = pd.DataFrame(
        {"Close": np.arange(100.0, 120.0), "Volume": np.full(20, 1000)},
        index=index,
    )
    result = build_features(prices)
    assert "target_next_close" not in [
        "return_1d", "return_5d", "return_10d", "close_lag_1",
        "close_lag_2", "close_lag_3", "close_lag_5"
    ]
