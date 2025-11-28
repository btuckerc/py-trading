# Point-in-Time ML Trading Stack

A research-grade, point-in-time-correct ML trading stack that:

* Ingests market + fundamentals + news/sentiment + (optionally) microstructure data
* Builds multi-horizon forecasts (1d / 1w / 1m / 6m) with uncertainty
* Converts those into portfolio decisions and backtests them properly (no lookahead, no survivorship bias)
* Is wired so the same code can run in "backtest mode" and "live / paper-trading mode"

## Design Principles

### Point-in-Time Correctness
Every observation `x(t, asset)` is built only from data that would have been known at or before time `t`. This means:
- Fundamentals use `report_release_date` to ensure only released data is used
- News/sentiment features only aggregate events with timestamps ≤ t
- Technical indicators use only backward-looking windows (no centered windows)
- Growth rate estimates use only historical data available at that point in time

### No Survivorship Bias
- Uses historical S&P 500 constituents datasets to track index membership over time
- Universe selection logic filters by what was actually tradable at each date
- Backtests only trade assets that existed and were in the index at that time

### No Lookahead Bias
- Explicit "simulation clock" abstraction ensures all queries are "as-of" a specific date
- Feature builders query data through the clock API, not directly from raw tables
- Validation tests intentionally break performance when future data is leaked

### Unified Backtest/Live Code Path
- Same data ingestion, feature engineering, and model inference code used in both modes
- Broker abstraction allows switching between paper and live trading
- Live engine reuses exact same model artifacts and feature pipelines

## Architecture

```
data/          - Raw vendor data, normalized schemas, DuckDB/Parquet storage
features/      - Feature engineering pipelines (technical, fundamentals, sentiment)
labels/        - Multi-horizon return labels, regime labels, event-based labels
models/        - Baseline models, PyTorch sequence models, uncertainty estimation
portfolio/     - Signal-to-position conversion, risk constraints, transaction costs
backtest/      - Vectorized and event-driven backtesting engines
live/          - Broker abstraction, paper trading, live execution loop
configs/       - Centralized configuration files
scripts/       - Data ingestion and processing scripts
notebooks/     - Research notebooks and visualizations
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Ingest initial data:
```bash
python scripts/ingest_sp500_constituents.py
python scripts/ingest_bars.py --start-date 2000-01-01 --end-date 2024-12-31
```

4. Build features and labels:
```bash
python scripts/build_features.py --start-date 2000-01-01 --end-date 2024-12-31
python scripts/build_labels.py --start-date 2000-01-01 --end-date 2024-12-31
```

5. Train a baseline model:
```bash
python scripts/train_baseline.py --model xgboost --train-start 2000-01-01 --train-end 2014-12-31
```

6. Run backtest:
```bash
python scripts/run_backtest.py --strategy long_top_k --start-date 2015-01-01 --end-date 2024-12-31
```

## Data Sources

- **Historical EOD**: Tiingo, Yahoo Finance (via yfinance)
- **Fundamentals**: Tiingo, Financial Modeling Prep
- **News/Sentiment**: Finnhub, Alpha Vantage
- **Universe**: Historical S&P 500 constituents (fja05680/sp500)

## License

MIT

