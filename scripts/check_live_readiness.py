"""Check if the system is ready for live trading.

This script verifies all quantitative gates defined in configs/base.yaml
are met before transitioning from paper to live trading.

Usage:
    python scripts/check_live_readiness.py
    python scripts/check_live_readiness.py --verbose
    python scripts/check_live_readiness.py --json  # Output as JSON for automation
    python scripts/check_live_readiness.py --notify  # Send email notification with results
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import argparse
import json
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from configs.loader import get_config
from loguru import logger


def load_paper_trading_logs(log_dir: Path) -> List[Dict]:
    """Load all paper trading logs."""
    logs = []
    if not log_dir.exists():
        return logs
    
    for log_file in sorted(log_dir.glob("daily_log_*.json")):
        try:
            with open(log_file) as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            logger.warning(f"Could not load {log_file}: {e}")
    
    return logs


def compute_paper_metrics(logs: List[Dict]) -> Dict[str, Any]:
    """Compute metrics from paper trading logs."""
    if len(logs) == 0:
        return {
            'trading_days': 0,
            'total_trades': 0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0,
            'total_return': 0.0,
            'consecutive_errors': 0,
        }
    
    # Extract equity curve
    account_values = []
    dates = []
    total_trades = 0
    consecutive_errors = 0
    max_consecutive_errors = 0
    
    for log in logs:
        account_value = log.get('account_value', 0)
        orders = log.get('orders', [])
        trading_date = log.get('trading_date')
        
        if account_value > 0:
            account_values.append(account_value)
            dates.append(trading_date)
            consecutive_errors = 0
        else:
            consecutive_errors += 1
            max_consecutive_errors = max(max_consecutive_errors, consecutive_errors)
        
        # Count trades (non-dry-run orders)
        for order in orders:
            if not order.get('dry_run', False) and order.get('status') != 'failed':
                total_trades += 1
    
    if len(account_values) < 2:
        return {
            'trading_days': len(logs),
            'total_trades': total_trades,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0,
            'total_return': 0.0,
            'consecutive_errors': max_consecutive_errors,
        }
    
    # Compute returns
    import numpy as np
    values = np.array(account_values)
    returns = np.diff(values) / values[:-1]
    
    # Compute metrics
    total_return = (values[-1] / values[0]) - 1
    
    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)
    
    # Volatility (annualized)
    daily_vol = np.std(returns)
    annualized_vol = daily_vol * np.sqrt(252)
    
    # Sharpe (annualized, assuming 0 risk-free rate)
    daily_return = np.mean(returns)
    sharpe = (daily_return / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
    
    return {
        'trading_days': len(logs),
        'total_trades': total_trades,
        'max_drawdown': float(max_drawdown),
        'volatility': float(annualized_vol),
        'sharpe': float(sharpe),
        'total_return': float(total_return),
        'consecutive_errors': max_consecutive_errors,
        'first_date': dates[0] if dates else None,
        'last_date': dates[-1] if dates else None,
    }


def check_gate(
    name: str,
    value: Any,
    threshold: Any,
    comparison: str = "<=",
    description: str = ""
) -> Tuple[bool, str]:
    """
    Check if a gate is met.
    
    Args:
        name: Gate name
        value: Current value
        threshold: Threshold value
        comparison: Comparison operator ("<=", ">=", "==", "<", ">")
        description: Human-readable description
    
    Returns:
        Tuple of (passed: bool, message: str)
    """
    comparisons = {
        "<=": lambda v, t: v <= t,
        ">=": lambda v, t: v >= t,
        "==": lambda v, t: v == t,
        "<": lambda v, t: v < t,
        ">": lambda v, t: v > t,
    }
    
    passed = comparisons[comparison](value, threshold)
    
    if passed:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    # Format values for display
    if isinstance(value, float):
        value_str = f"{value:.2%}" if abs(value) < 10 else f"{value:.2f}"
    else:
        value_str = str(value)
    
    if isinstance(threshold, float):
        threshold_str = f"{threshold:.2%}" if abs(threshold) < 10 else f"{threshold:.2f}"
    else:
        threshold_str = str(threshold)
    
    message = f"{status} {name}: {value_str} {comparison} {threshold_str}"
    if description:
        message += f" ({description})"
    
    return passed, message


def send_readiness_notification(results: Dict, verbose_output: str = None) -> bool:
    """Send email notification with readiness check results."""
    try:
        from live.alerting import AlertManager, AlertLevel
        
        alerter = AlertManager(email_enabled=True, slack_enabled=True, dry_run=False)
        
        if results['all_passed']:
            subject = "🎉 LIVE TRADING GATES PASSED"
            level = AlertLevel.INFO
            message = (
                "All quantitative gates have passed!\n\n"
                "The system is ready for live trading.\n\n"
                f"Paper Trading Days: {results['metrics']['trading_days']}\n"
                f"Total Trades: {results['metrics']['total_trades']}\n"
                f"Max Drawdown: {results['metrics']['max_drawdown']:.1%}\n"
                f"Sharpe Ratio: {results['metrics']['sharpe']:.2f}\n\n"
                "Next steps:\n"
                "1. Review the detailed metrics\n"
                "2. Set LIVE_TRADING_ENABLED=1 in your environment\n"
                "3. Switch to --broker alpaca_live with 5% allocation\n"
            )
        else:
            subject = "📊 Live Trading Readiness Check - Gates Not Yet Met"
            level = AlertLevel.WARNING
            failed_gates = [g for g in results['gates'] if not g['passed']]
            message = (
                "Some gates have not yet been met.\n\n"
                f"Paper Trading Days: {results['metrics']['trading_days']} / 30 required\n"
                f"Total Trades: {results['metrics']['total_trades']} / 50 required\n\n"
                "Failed Gates:\n"
            )
            for gate in failed_gates:
                message += f"  - {gate['message']}\n"
            message += "\nContinue paper trading to accumulate more data."
        
        fields = {
            "Trading Days": str(results['metrics']['trading_days']),
            "Total Trades": str(results['metrics']['total_trades']),
            "Max Drawdown": f"{results['metrics']['max_drawdown']:.1%}",
            "Sharpe": f"{results['metrics']['sharpe']:.2f}",
            "All Passed": "Yes" if results['all_passed'] else "No",
        }
        
        return alerter.send_alert(subject, message, level, fields)
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check live trading readiness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--notify", action="store_true", help="Send email notification with results")
    args = parser.parse_args()
    
    config = get_config()
    gates_config = config.live_gates if hasattr(config, 'live_gates') else {}
    
    # Load paper trading logs
    log_dir = Path("logs/live_trading")
    logs = load_paper_trading_logs(log_dir)
    
    # Compute metrics
    metrics = compute_paper_metrics(logs)
    
    # Check all gates
    results = {
        'timestamp': datetime.now().isoformat(),
        'gates': [],
        'metrics': metrics,
        'all_passed': True,
    }
    
    # Gate 1: Minimum paper trading days
    passed, msg = check_gate(
        "Paper Trading Days",
        metrics['trading_days'],
        gates_config.get('min_paper_trading_days', 30),
        ">=",
        "Minimum soak period"
    )
    results['gates'].append({'name': 'min_paper_trading_days', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Gate 2: Minimum trades
    passed, msg = check_gate(
        "Total Trades",
        metrics['total_trades'],
        gates_config.get('min_paper_trades', 50),
        ">=",
        "Minimum trade count"
    )
    results['gates'].append({'name': 'min_paper_trades', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Gate 3: Max drawdown
    passed, msg = check_gate(
        "Max Drawdown",
        metrics['max_drawdown'],
        gates_config.get('max_paper_drawdown_pct', 0.20),
        "<=",
        "Maximum acceptable drawdown"
    )
    results['gates'].append({'name': 'max_paper_drawdown', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Gate 4: Max volatility
    passed, msg = check_gate(
        "Annualized Volatility",
        metrics['volatility'],
        gates_config.get('max_paper_volatility', 0.40),
        "<=",
        "Maximum acceptable volatility"
    )
    results['gates'].append({'name': 'max_paper_volatility', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Gate 5: Min Sharpe
    passed, msg = check_gate(
        "Sharpe Ratio",
        metrics['sharpe'],
        gates_config.get('min_paper_sharpe', 0.0),
        ">=",
        "Minimum Sharpe ratio"
    )
    results['gates'].append({'name': 'min_paper_sharpe', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Gate 6: Max consecutive errors
    passed, msg = check_gate(
        "Consecutive Errors",
        metrics['consecutive_errors'],
        gates_config.get('max_consecutive_errors', 3),
        "<=",
        "System stability"
    )
    results['gates'].append({'name': 'max_consecutive_errors', 'passed': passed, 'message': msg})
    results['all_passed'] &= passed
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("LIVE TRADING READINESS CHECK")
        print("=" * 60)
        print(f"Check Time: {results['timestamp']}")
        print(f"Paper Trading Logs: {len(logs)}")
        print()
        
        print("GATE RESULTS:")
        print("-" * 60)
        for gate in results['gates']:
            print(gate['message'])
        
        print()
        print("-" * 60)
        
        if results['all_passed']:
            print("✅ ALL GATES PASSED - System is ready for live trading")
            print()
            initial_alloc = gates_config.get('initial_live_allocation_pct', 0.05)
            print(f"Recommended initial allocation: {initial_alloc:.0%} of capital")
            print()
            print("Next steps:")
            print("  1. Set up Alpaca API keys in .env")
            print("  2. Run with --broker alpaca_paper for final validation")
            print("  3. After validation, run with --broker alpaca_live")
        else:
            print("❌ SOME GATES FAILED - Continue paper trading")
            print()
            print("Actions needed:")
            for gate in results['gates']:
                if not gate['passed']:
                    print(f"  - {gate['name']}: {gate['message']}")
        
        print("=" * 60)
        
        if args.verbose:
            print("\nDETAILED METRICS:")
            print("-" * 60)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # Send notification if requested
    if args.notify:
        print("\nSending notification...")
        if send_readiness_notification(results):
            print("Notification sent successfully!")
        else:
            print("Failed to send notification (check email configuration)")
    
    # Exit code based on gate results
    sys.exit(0 if results['all_passed'] else 1)


if __name__ == "__main__":
    main()

