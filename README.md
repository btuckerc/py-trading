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

# Run backtest (data is auto-fetched if missing)
python scripts/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31 \
  --model xgboost --horizon 20 --top-k 10 --regime-aware --run-benchmarks

# Run backtest with specific benchmarks (default: all 3 - SPY, DIA, QQQ)
python scripts/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31 \
  --model xgboost --horizon 20 --top-k 10 --run-benchmarks --benchmark sp500
python scripts/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31 \
  --model xgboost --horizon 20 --top-k 10 --run-benchmarks --benchmarks sp500,dow,nasdaq

# Paper trade (daily after market close)
python scripts/run_live_loop.py --regime-aware --sector-tilts --drawdown-throttle --dry-run
```

## How It Works

```
Data → Features → Model → Scores → Risk Manager → Broker
         ↑                            ↑
    Regime Detection ────────────────┘
```

1. **Data Layer** — Yahoo/Tiingo bars stored in DuckDB + Parquet. As-of query API ensures point-in-time correctness. Auto-fetches missing data on backtest/live runs.
2. **Feature Pipeline** — Technical (momentum, volatility), cross-sectional (z-scores), calendar effects, regime indicators.
3. **Models** — XGBoost/LightGBM for tabular, Conv1D+LSTM/TCN for sequences.
4. **Risk Management** — Position caps (20%), sector caps (40%), drawdown throttling, regime-based exposure scaling.
5. **Execution** — Paper broker for testing, Alpaca for live.

## Universe

Default universe: 126 curated large-cap S&P 500 stocks (~70-80% of index market cap). Stored in `data/sp500_constituents.csv` and loaded into `universe_membership` table for survivorship-bias-free backtesting.

```bash
# Refresh universe membership (auto-fetches missing price data)
python scripts/refresh_universe_membership.py --fetch-data
```

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
data/           Storage, normalization, as-of query API, maintenance
features/       Technical, cross-sectional, fundamental, regime features  
labels/         Return labels, regime detection
models/         XGBoost, LightGBM, PyTorch sequence models
portfolio/      Strategies, risk management, sector tilts
backtest/       Vectorized backtester
live/           Paper broker, Alpaca broker, alerting
configs/        YAML configuration (base.yaml)
scripts/        CLI tools (see below)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `run_backtest.py` | Run backtests with ML models |
| `run_live_loop.py` | Daily paper/live trading loop |
| `simulate_daily_trading.py` | Simulate daily decisions for a date range |
| `generate_performance_report.py` | Generate charts and performance reports |
| `ensure_data_coverage.py` | Check/fetch missing price data |
| `refresh_universe_membership.py` | Rebuild universe from constituents CSV |
| `check_live_readiness.py` | Verify readiness gates before going live |

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

## Performance Reports

Generate professional performance reports with charts:

```bash
# Simulate daily trading and generate report
python scripts/simulate_daily_trading.py --start-date 2025-09-01 --end-date 2025-11-26

# Generate charts and HTML report (shows all 3 benchmarks by default: SPY, DIA, QQQ)
python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26 --html

# Generate report with specific benchmark(s)
python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26 --benchmark sp500
python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26 --benchmarks sp500,dow,nasdaq
```

**Benchmarks:**
- By default, reports compare against all three major indices: S&P 500 (SPY), Dow Jones (DIA), and Nasdaq Composite (QQQ)
- Use `--benchmark` to select a single benchmark or `--benchmarks` for a comma-separated list
- Benchmark tickers are configurable in `configs/base.yaml` under the `benchmarks` section

**Output files:**
- `report_*.txt` — Text summary with tables
- `report_*.json` — Full data for programmatic use
- `report_*_dashboard.png` — Single-page visual summary
- `equity_curve.png`, `drawdown.png`, `monthly_returns.png` — Individual charts
- `report_*.html` — Interactive HTML report

**Sample metrics:**
```
Period: 2025-09-02 to 2025-11-26 (62 trading days)

ML Strategy:     +15.10%
S&P 500 (SPY):   +6.45%
Dow Jones (DIA): +5.20%
Nasdaq (QQQ):    +8.30%
Alpha (vs SPY):  +8.65%

Sharpe Ratio:    2.84
Max Drawdown:    -5.88%
Win Rate:        62.3%
```

## Docker Deployment

The system is designed for zero-touch deployment. Just start the container and it handles everything:

```bash
# Start the scheduler (runs daily at 4:30 PM ET)
docker-compose up -d trading-scheduler

# Or run manually
docker-compose run --rm trading
```

### Auto-Bootstrap

**No manual data setup required.** When the container starts with an empty or sparse database:
1. System detects missing data (< 100 bars per asset)
2. Automatically fetches full historical data (2020-present, ~6 years)
3. Builds universe membership from `data/sp500_constituents.csv`
4. Proceeds with normal trading loop

This takes 2-5 minutes on first run, then data is persisted in the `./data` volume.

### Force Mode (Testing/Insights)

Run anytime, including weekends/holidays, to see current recommendations:

```bash
# Get insights using most recent available data (no fetch attempts)
docker-compose run --rm trading python scripts/run_live_loop.py --force --top-k 10

# With all features enabled
docker-compose run --rm trading python scripts/run_live_loop.py --force --top-k 10 \
  --regime-aware --sector-tilts --vol-scaling
```

`--force` mode:
- Uses whatever data is available (no fetch attempts on weekends)
- Automatically runs in dry-run mode (no orders submitted)
- Bypasses trading-day and idempotency checks
- Perfect for checking "what would the model recommend today?"

### Safety Features

The system prevents duplicate orders automatically:
- **Trading day check**: Skips execution on weekends/holidays
- **Idempotency**: Won't run twice on the same trading day
- **Force mode**: Always dry-run unless explicitly overridden

## Data Management

Data coverage is managed automatically:

- **Auto-bootstrap**: Empty databases are automatically populated with full history
- **Daily top-up**: Missing recent data is fetched before each run
- **Manual check**: `python scripts/ensure_data_coverage.py --report`

Configuration in `configs/base.yaml`:
```yaml
data:
  min_history_start_date: "2020-01-01"
  auto_fetch_on_backtest: true
  auto_fetch_on_live: true
```

## Future Direction

- **Universe expansion** — Full S&P 500 with historical constituent changes
- **Alternative data** — Options flow, short interest, insider transactions
- **Sequence models** — Transformer architectures for longer-horizon forecasts
- **Multi-asset** — Futures, crypto, international equities
- **Execution optimization** — VWAP/TWAP, optimal trade scheduling

## License

MIT
