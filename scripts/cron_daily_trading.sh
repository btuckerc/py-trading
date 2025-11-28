#!/bin/bash
# Daily trading loop cron script
# Schedule: Run at 4:30 PM ET (after market close) on weekdays
#
# Crontab entry (adjust timezone as needed):
#   30 16 * * 1-5 /path/to/py-finance/scripts/cron_daily_trading.sh >> /path/to/py-finance/logs/cron.log 2>&1
#
# For Mac mini deployment:
#   1. Copy project to Mac mini
#   2. Set up .venv with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   3. Copy .env with API keys
#   4. Add crontab entry above (use crontab -e)
#   5. For launchd alternative, see scripts/com.pyfinance.daily-trading.plist

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/.venv"
LOG_DIR="${PROJECT_ROOT}/logs"
HEARTBEAT_PATH="${LOG_DIR}/heartbeat.json"

# Broker mode: "paper" for PaperBroker, "alpaca_paper" for Alpaca paper, "alpaca_live" for live
# Default to alpaca_paper so trades show up in Alpaca dashboard during soak period
BROKER_MODE="${BROKER_MODE:-alpaca_paper}"

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
echo "Daily Trading Loop - $(date)"
echo "Broker Mode: ${BROKER_MODE}"
echo "=========================================="

# Run the live trading loop with all regime-aware features enabled
# --skip-if-logged ensures idempotency (won't re-run if already ran today)
# --enable-alerts sends email notifications on errors and daily summaries
python scripts/run_live_loop.py \
    --broker "$BROKER_MODE" \
    --regime-aware \
    --sector-tilts \
    --vol-scaling \
    --drawdown-throttle \
    --top-k 10 \
    --initial-capital 100000 \
    --heartbeat-path "$HEARTBEAT_PATH" \
    --skip-if-logged \
    --enable-alerts

# Log completion
echo "Daily trading loop completed at $(date)"
echo ""

