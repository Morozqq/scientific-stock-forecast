"""A small, reproducible stock-price forecasting example.

The program downloads daily market data, builds lagged technical features,
trains a Random Forest regressor using a chronological split, and reports
out-of-sample metrics. It is an educational research tool, not investment advice.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d",
    "volatility_5d", "volatility_10d",
    "volume_change_1d", "close_lag_1", "close_lag_2",
    "close_lag_3", "close_lag_5",
]


@dataclass
class ForecastResult:
    ticker: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    next_day_forecast: float
    output_dir: Path


def download_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV data and return a validated Close/Volume frame."""
    import yfinance as yf

    data = yf.download(
        ticker, period=period, interval="1d", auto_adjust=True,
        progress=False, threads=False
    )
    if data.empty:
        raise ValueError(f"No market data returned for ticker {ticker!r}.")
    # yfinance can return either flat or MultiIndex columns depending on version.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Close", "Volume"}
    if not required.issubset(data.columns):
        raise ValueError(f"Downloaded data is missing columns: {required - set(data.columns)}")
    frame = data[["Close", "Volume"]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["Close"] > 0]
    if len(frame) < 100:
        raise ValueError("At least 100 valid daily observations are required.")
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build lagged predictors and next-session target without future leakage."""
    frame = prices.copy()
    close = frame["Close"]
    frame["return_1d"] = close.pct_change(1)
    frame["return_5d"] = close.pct_change(5)
    frame["return_10d"] = close.pct_change(10)
    daily_returns = close.pct_change()
    frame["volatility_5d"] = daily_returns.rolling(5).std()
    frame["volatility_10d"] = daily_returns.rolling(10).std()
    frame["volume_change_1d"] = frame["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    frame["close_lag_1"] = close.shift(1)
    frame["close_lag_2"] = close.shift(2)
    frame["close_lag_3"] = close.shift(3)
    frame["close_lag_5"] = close.shift(5)
    frame["target_next_close"] = close.shift(-1)
    return frame.replace([np.inf, -np.inf], np.nan)


def run_forecast(
    ticker: str = "AAPL",
    period: str = "5y",
    output_dir: str | Path = "outputs",
    n_estimators: int = 300,
) -> ForecastResult:
    """Train/evaluate chronologically and generate a one-session-ahead estimate."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices = download_prices(ticker, period)
    featured = build_features(prices)
    usable = featured.dropna(subset=FEATURE_COLUMNS + ["target_next_close"])
    if len(usable) < 100:
        raise ValueError("Not enough rows remain after feature engineering.")
    X = usable[FEATURE_COLUMNS]
    y = usable["target_next_close"]
    split = int(len(usable) * 0.8)
    if split <= 0 or split >= len(usable):
        raise ValueError("Unable to create chronological train/test split.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    mae = float(mean_absolute_error(y_test, test_pred))
    # Directional accuracy: whether the predicted next close is above the
    # same day's close, compared with the actual next close.
    same_day_close = usable.loc[X_test.index, "Close"].to_numpy()
    actual_direction = y_test.to_numpy() > same_day_close
    predicted_direction = test_pred > same_day_close
    directional_accuracy = float(np.mean(actual_direction == predicted_direction))

    predictions = pd.DataFrame(
        {"actual_next_close": y_test, "predicted_next_close": test_pred},
        index=X_test.index,
    )
    predictions.to_csv(output / f"{ticker.upper()}_test_predictions.csv", index_label="date")

    latest_X = featured.dropna(subset=FEATURE_COLUMNS)[FEATURE_COLUMNS].iloc[[-1]]
    next_day_forecast = float(model.predict(latest_X)[0])
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "directional_accuracy": directional_accuracy,
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }
    pd.DataFrame([{"ticker": ticker.upper(), **metrics, "next_day_forecast": next_day_forecast}]).to_csv(
        output / f"{ticker.upper()}_metrics.csv", index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(predictions.index, predictions["actual_next_close"], label="Actual")
    plt.plot(predictions.index, predictions["predicted_next_close"], label="Predicted")
    plt.title(f"{ticker.upper()} next-session close: test period")
    plt.xlabel("Date")
    plt.ylabel("Adjusted close (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"{ticker.upper()}_forecast.png", dpi=160)
    plt.close()

    return ForecastResult(ticker.upper(), metrics, predictions, next_day_forecast, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker, e.g. AAPL or MSFT")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 1y, 2y, 5y")
    parser.add_argument("--output", default="outputs", help="Directory for CSV and plot outputs")
    parser.add_argument("--trees", type=int, default=300, help="Number of Random Forest trees")
    args = parser.parse_args()
    result = run_forecast(args.ticker, args.period, args.output, args.trees)
    print(f"Ticker: {result.ticker}")
    print(f"MAE: {result.metrics['MAE']:.4f}")
    print(f"RMSE: {result.metrics['RMSE']:.4f}")
    print(f"Directional accuracy: {result.metrics['directional_accuracy']:.3f}")
    print(f"Next-session estimated close: {result.next_day_forecast:.2f}")
    print(f"Outputs saved to: {result.output_dir.resolve()}")


if __name__ == "__main__":
    main()
