# py-finance

A point-in-time ML trading stack with regime-aware risk management.

## What It Does

Builds multi-horizon return forecasts, converts them to portfolio weights, and executes via paper or live brokers—all with strict point-in-time correctness (no lookahead, no survivorship bias).

**Key Results (2020–2024 backtest):**
| Metric | Baseline | Regime-Aware | SPY B&H |
|--------|----------|--------------|---------|
| CAGR | 29.1% | 58.9% | 12.4% |
| Sharpe | 0.54 | 1.52 | 0.65 |
| Max DD | -64.6% | -30.2% | -33.7% |

## Quick Start

```bash
# Setup
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
brew install libomp  # Required for XGBoost/LightGBM on macOS

# Ingest data
python scripts/ingest_bars.py --start-date 2020-01-01 --end-date 2024-12-31 \
  --symbols SPY AAPL MSFT NVDA GOOGL META AMZN

# Run backtest
python scripts/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31 \
  --model xgboost --horizon 20 --top-k 10 --regime-aware

# Paper trade (daily after market close)
python scripts/run_live_loop.py --regime-aware --sector-tilts --drawdown-throttle --dry-run
```

## How It Works

```
Data → Features → Model → Scores → Risk Manager → Broker
         ↑                            ↑
    Regime Detection ────────────────┘
```

1. **Data Layer** — Yahoo/Tiingo bars, fundamentals, news. Stored in DuckDB + Parquet with as-of query API.
2. **Feature Pipeline** — Technical (momentum, volatility), cross-sectional (z-scores), calendar effects, regime indicators.
3. **Models** — XGBoost/LightGBM for tabular, Conv1D+LSTM/TCN for sequences. Ensemble uncertainty via MC dropout.
4. **Risk Management** — Position caps (20%), sector caps (40%), drawdown throttling, regime-based exposure scaling.
5. **Execution** — Paper broker for testing, Alpaca for live (commission-free, ideal for small accounts).

## Regime-Aware Trading

The system detects four market regimes via K-means clustering on returns + volatility:

| Regime | Exposure | Sector Tilt |
|--------|----------|-------------|
| `bull_low_vol` | 100% | Growth/Tech |
| `bull_high_vol` | 40% | Balanced |
| `bear_low_vol` | 70% | Defensive |
| `bear_high_vol` | 25% | Max Defensive |

Exposure multipliers are data-driven from historical regime performance.

## Project Structure

```
data/       Storage, normalization, as-of query API
features/   Technical, cross-sectional, fundamental, regime features  
labels/     Return labels, regime detection
models/     XGBoost, LightGBM, PyTorch sequence models
portfolio/  Strategies, risk management, sector tilts
backtest/   Vectorized backtester
live/       Paper broker, Alpaca broker, alerting
configs/    YAML configuration
scripts/    CLI tools for ingestion, backtest, live trading
```

## Configuration

Copy `.env.example` to `.env` and add API keys:

```bash
# Data vendors (Yahoo is free, no key needed)
TIINGO_API_KEY=...

# Broker (Alpaca recommended for small accounts)
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...

# Alerts (optional)
ALERT_SLACK_WEBHOOK_URL=...
```

Risk parameters in `configs/base.yaml`:
- `portfolio.risk.max_position_pct`: 0.20
- `portfolio.risk.max_sector_pct`: 0.40
- `portfolio.drawdown.throttle_threshold_pct`: 0.15
- `portfolio.regime_policy.exposure_multipliers`: per-regime scaling

## Going Live

1. **Paper trade for 30+ days** — Run `scripts/run_live_loop.py` daily
2. **Check readiness** — `python scripts/check_live_readiness.py`
3. **Start small** — 5% allocation, ramp up per `live_gates.ramp_up_schedule`

Quantitative gates before live: min 30 days paper, max 20% drawdown, max 40% volatility.

## Operations

### Required Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Data Vendors
TIINGO_API_KEY=...           # Get at https://www.tiingo.com
FINNHUB_API_KEY=...          # Get at https://finnhub.io

# Alpaca Broker (commission-free, recommended for small accounts)
ALPACA_API_KEY=...           # Get at https://alpaca.markets
ALPACA_SECRET_KEY=...

# Gmail SMTP Alerts (use App Password, not regular password)
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=your-email@gmail.com
ALERT_EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"  # Gmail App Password
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=recipient@example.com        # Comma-separated for multiple

# Slack Alerts (optional, leave empty to disable)
ALERT_SLACK_WEBHOOK_URL=
```

**Gmail App Password Setup:**
1. Enable 2FA on your Google account
2. Go to Google Account → Security → App passwords
3. Generate a new app password for "Mail"
4. Use the 16-character password (with spaces) in `ALERT_EMAIL_PASSWORD`

### Daily Paper Trading Job

Run after US market close (e.g., 16:30-17:00 ET):

```bash
cd /path/to/py-finance
python scripts/run_live_loop.py \
  --skip-if-logged \
  --regime-aware \
  --vol-scaling \
  --drawdown-throttle \
  --sector-tilts \
  --enable-alerts
```

**Cron example (Mac mini, 4:30 PM ET = 21:30 UTC in winter):**
```cron
30 21 * * 1-5 cd /path/to/py-finance && source .venv/bin/activate && python scripts/run_live_loop.py --skip-if-logged --regime-aware --vol-scaling --drawdown-throttle --sector-tilts --enable-alerts >> logs/cron.log 2>&1
```

### Checking Readiness Gates

After 30+ trading days of paper logs:

```bash
python scripts/check_live_readiness.py --verbose
```

Gates checked:
- `min_paper_trading_days`: 30
- `min_paper_trades`: 50
- `max_paper_drawdown_pct`: 0.20
- `max_paper_volatility`: 0.40
- `min_paper_sharpe`: 0.0
- `max_consecutive_errors`: 3

### Switching to Live Trading

Once gates pass:

```bash
# Run with Alpaca paper first for final validation
python scripts/run_live_loop.py --broker alpaca_paper --enable-alerts

# Then switch to live (5% allocation recommended initially)
python scripts/run_live_loop.py --broker alpaca_live --enable-alerts
```

### Monitoring

- **Daily logs:** `logs/live_trading/daily_log_YYYY-MM-DD.json`
- **Email alerts:** Sent on errors, regime changes, and daily summaries
- **Readiness check:** `python scripts/check_live_readiness.py --json`

### Weekly Research Job

Run weekly to refresh backtest metrics:

```bash
python scripts/run_multi_horizon_backtest.py \
  --start-date 2020-01-01 \
  --end-date $(date +%Y-%m-%d) \
  --model-type xgboost \
  --horizons 1 5 20 \
  --run-benchmarks
```

**Cron example (Sundays at 2 AM):**
```cron
0 2 * * 0 cd /path/to/py-finance && source .venv/bin/activate && python scripts/run_multi_horizon_backtest.py --start-date 2020-01-01 --end-date $(date +%Y-%m-%d) --model-type xgboost --horizons 1 5 20 --run-benchmarks >> logs/weekly_backtest.log 2>&1
```

## Future Direction

- **Universe expansion** — Full S&P 500 with survivorship-bias-free membership
- **Alternative data** — Options flow, short interest, insider transactions
- **Sequence models** — Transformer architectures for longer-horizon forecasts
- **Multi-asset** — Futures, crypto, international equities
- **Execution optimization** — VWAP/TWAP, optimal trade scheduling

## License

MIT
