# Scientific Stock Forecast

A small scientific-software example for Assignment 3: Software Development and Integration. It downloads historical daily prices, engineers lagged features, trains a Random Forest regressor, evaluates on a chronological hold-out set, and exports a CSV and a chart.

> **Important:** This is an educational demonstration, not financial advice. Stock prices are noisy, results depend on the period and data source, and historical test performance does not guarantee future results.

## Objectives

- Demonstrate Git-based version control and a public GitHub repository workflow.
- Provide a reproducible Python data-processing and modeling pipeline.
- Run automated unit tests in GitHub Actions on pushes and pull requests.
- Keep evaluation chronological to reduce look-ahead leakage.

## Technology choices

| Tool | Purpose | Reason |
|---|---|---|
| Python 3.10+ | Main language | Widely used in scientific computing and has a mature data ecosystem |
| pandas / NumPy | Data cleaning and features | Tabular time-series operations and numerical calculations |
| yfinance | Historical market data | Convenient educational interface to Yahoo Finance data |
| scikit-learn | Random Forest and metrics | Reproducible model API and established evaluation tools |
| Matplotlib | Plot output | Static chart generation in local and CI environments |
| pytest | Unit tests | Simple, widely adopted test framework |
| Git / GitHub | Version control and collaboration | Change history, pull requests, issues, and public sharing |
| GitHub Actions | CI | Automatically installs dependencies and runs tests |

## Requirements

- Python 3.10, 3.11, or 3.12
- Internet access to download data from Yahoo Finance
- pip

## Run locally

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -v
python -m src.forecast --ticker AAPL --period 5y --output outputs
```

Example for Microsoft:

```bash
python -m src.forecast --ticker MSFT --period 2y --output outputs
```

## Outputs

The `outputs/` directory is generated at runtime and intentionally excluded from Git:
- `<TICKER>_test_predictions.csv` — actual and predicted next-session closing prices on the test set.
- `<TICKER>_metrics.csv` — MAE, RMSE, directional accuracy, and sample counts.
- `<TICKER>_forecast.png` — chart comparing actual and predicted values.

The exact numeric results vary with the data provider, the retrieval date, and the selected ticker/period. Do not cite invented or fixed performance numbers.

## Method

1. Retrieve daily adjusted OHLCV data.
2. Create returns over 1, 5, and 10 sessions; rolling return volatility; volume change; and lagged closing prices.
3. Define the target as the next session's closing price.
4. Sort by date and split the usable observations chronologically: first 80% for training, final 20% for testing.
5. Train a Random Forest regressor with a fixed random seed.
6. Report mean absolute error (MAE), root mean squared error (RMSE), and directional accuracy.
7. Refit is not performed on the test set; a next-session estimate is produced using the latest available features after evaluation.

## CI/CD

The workflow in `.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch. It tests the project on Python 3.10, 3.11, and 3.12. This assignment implements **continuous integration (CI)** and automated verification. It does not automatically deploy a production service; therefore it should not be described as full continuous deployment.

## Suggested Git workflow

```bash
git init
git add .
git commit -m "Initial scientific forecasting project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/scientific-stock-forecast.git
git push -u origin main
```

Then, on GitHub:
1. Create a **public** repository named `scientific-stock-forecast` (do not initialize it with a second README if pushing this folder).
2. Push the files using the commands above.
3. Open the **Actions** tab and confirm the Python CI workflow completes successfully.
4. Create at least one issue, such as “Add a naive baseline forecast”, to demonstrate repository issue tracking.
5. Replace `YOUR-USERNAME` with the actual account name and add the real repository URL to the report.

## Limitations and future work

- The model uses only a small set of price/volume-derived features.
- A single hold-out period can be sensitive to market regimes.
- No transaction costs, slippage, risk constraints, or benchmark strategy are included.
- Yahoo Finance availability and terms can change.
- Potential extensions include walk-forward validation, a naive baseline, confidence intervals, news sentiment, and comparison with linear regression.

## License

MIT. See `LICENSE`.
