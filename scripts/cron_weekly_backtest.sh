#!/bin/bash
# Weekly backtest refresh script
# Schedule: Run on Sundays at 2 AM to refresh backtest metrics
#
# Crontab entry:
#   0 2 * * 0 /path/to/py-finance/scripts/cron_weekly_backtest.sh >> /path/to/py-finance/logs/weekly_backtest.log 2>&1
#
# This script:
# 1. Fetches any missing recent data
# 2. Runs a multi-horizon backtest over a trailing window
# 3. Saves results to artifacts/backtest_results/
# 4. Optionally sends an email summary

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/.venv"
LOG_DIR="${PROJECT_ROOT}/logs"

# Backtest configuration
TRAIN_START="2020-01-01"
BACKTEST_YEARS=5  # How many years of backtest to run
MODEL_TYPE="xgboost"
HORIZONS="1 5 20"
TOP_K=10

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Load environment variables from .env
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Change to project root
cd "$PROJECT_ROOT"

# Log start
echo "=========================================="
echo "Weekly Backtest Refresh - $(date)"
echo "=========================================="

# Calculate dates
# End date: last Friday (or today if it's a weekend)
END_DATE=$(python -c "
from datetime import date, timedelta
today = date.today()
# Go back to last Friday
days_since_friday = (today.weekday() - 4) % 7
if days_since_friday == 0 and today.weekday() != 4:
    days_since_friday = 7
last_friday = today - timedelta(days=days_since_friday)
print(last_friday.isoformat())
")

# Start date: BACKTEST_YEARS ago
START_DATE=$(python -c "
from datetime import date
import sys
end_date = date.fromisoformat('$END_DATE')
start_date = date(end_date.year - $BACKTEST_YEARS, end_date.month, end_date.day)
print(start_date.isoformat())
")

# Training end: 80% of the way through the backtest period
TRAIN_END=$(python -c "
from datetime import date, timedelta
start = date.fromisoformat('$START_DATE')
end = date.fromisoformat('$END_DATE')
days = (end - start).days
train_end = start + timedelta(days=int(days * 0.8))
print(train_end.isoformat())
")

echo "Backtest period: $START_DATE to $END_DATE"
echo "Training period: $TRAIN_START to $TRAIN_END"
echo "Model: $MODEL_TYPE"
echo "Horizons: $HORIZONS"
echo "Top K: $TOP_K"
echo ""

# Run multi-horizon backtest with auto-fetch
echo "Running multi-horizon backtest..."
python scripts/run_multi_horizon_backtest.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --train-start "$TRAIN_START" \
    --train-end "$TRAIN_END" \
    --model-type "$MODEL_TYPE" \
    --horizons $HORIZONS \
    --top-k "$TOP_K" \
    --run-benchmarks \
    --use-uncertainty

# Also run single-horizon backtest for comparison
echo ""
echo "Running single-horizon (20d) backtest with benchmarks..."
python scripts/run_backtest.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --train-start "$TRAIN_START" \
    --train-end "$TRAIN_END" \
    --model "$MODEL_TYPE" \
    --horizon 20 \
    --top-k "$TOP_K" \
    --run-benchmarks \
    --auto-fetch \
    --vendor yahoo

# Log completion
echo ""
echo "=========================================="
echo "Weekly backtest completed at $(date)"
echo "Results saved to artifacts/backtest_results/"
echo "=========================================="
