# Pair Trading Research Toolkit

Research code for studying mean-reverting equity pairs with econometric diagnostics, rolling hedge ratios, Z-score signals, backtesting, and alternative portfolio-weighting rules.

The project combines a notebook-based workflow with reusable Python modules. It is intended as an analytical portfolio project—not as a live trading system or investment recommendation.

## What it implements

- Data ingestion from local sector datasets and optional Yahoo Finance downloads.
- Pair diagnostics using ADF and Johansen cointegration tests, Hurst exponent, and estimated half-life.
- Rolling linear-regression hedge ratios, spread construction, and Z-score entry/exit signals.
- Long/short backtesting and capital-path calculations.
- Allocation alternatives including equal weight, volatility-aware weight, Z-score rules, and dynamic Kelly-style sizing.
- An end-to-end exploratory workflow in `pair_trading.ipynb`.

## Research flow

```text
price series
   ↓
pair diagnostics (ADF, Johansen, Hurst, half-life)
   ↓
rolling hedge ratio and spread
   ↓
Z-score signal and position state
   ↓
weighting rule and backtest
   ↓
capital path and risk/return diagnostics
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab pair_trading.ipynb
```

The notebook can use local files under `dat/`. The expected raw inputs are `stock_metadata.csv` and `historical_prices.csv`; generated `.npz` files cache cleaned time series. `read_data.yahoo_download(...)` provides an optional market-data route for explicit ticker lists.

## Repository map

- `pair_trading.ipynb`: exploratory research and backtest workflow.
- `read_data.py`: ingestion, cleaning, caching, and Yahoo Finance helper.
- `statistics.py`: cointegration and mean-reversion diagnostics.
- `cointegracion.py`: rolling spread/Z-score signals and pair-level capital simulation.
- `weights.py`: portfolio-allocation rules and holding-period logic.
- `utils.py`: rolling statistics, transformations, and shared helpers.

## Methodology notes

For a price pair \(x_t, y_t\), the toolkit estimates a hedge relationship and studies the residual spread:

```text
spread_t = y_t − (alpha_t + beta_t × x_t)
z_t      = (spread_t − rolling_mean) / rolling_std
```

The diagnostics are filters and research signals, not guarantees of future mean reversion. Before interpreting a backtest, validate data provenance, point-in-time availability, train/test separation, transaction costs, slippage, liquidity, and survivorship bias.

## Status

Active research prototype. The next useful improvements are automated tests, a reproducible sample dataset, transaction-cost modeling, and a documented out-of-sample evaluation protocol.
