#!/usr/bin/env python3
"""
Run performance reports for recent periods: past week, month-to-date, and year-to-date.

This script generates leakage-safe performance reports in order from shortest to longest period.
It uses Python's datetime for cross-platform date calculations.

Usage:
    # Run all three reports (week, MTD, YTD) with HTML output
    python scripts/run_recent_reports.py --format html

    # Run with PDF output
    python scripts/run_recent_reports.py --format pdf

    # Run with both HTML and PDF
    python scripts/run_recent_reports.py --format all

    # Run without charts (faster, text/JSON only)
    python scripts/run_recent_reports.py --no-charts

    # Run only specific periods
    python scripts/run_recent_reports.py --periods week mtd

    # Custom initial capital (default: 100000)
    python scripts/run_recent_reports.py --initial-capital 10000
"""

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.universe import TradingCalendar


def get_date_ranges():
    """Calculate start dates for week, MTD, and YTD periods."""
    today = date.today()
    calendar = TradingCalendar()
    
    # Past week: 14 calendar days ago (will filter to trading days)
    start_week = today - timedelta(days=14)
    
    # Month-to-date: first day of current month
    start_mtd = today.replace(day=1)
    
    # Year-to-date: January 1 of current year
    start_ytd = today.replace(month=1, day=1)
    
    return {
        'week': (start_week, today),
        'mtd': (start_mtd, today),
        'ytd': (start_ytd, today),
    }


def run_report(period_name: str, start_date: date, end_date: date, args):
    """Run generate_performance_report.py for a specific period."""
    print(f"\n{'='*80}")
    print(f"Generating {period_name.upper()} report: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    # Determine effective format (handle backward compatibility with --html)
    effective_format = args.format
    if args.html and args.format == "text":
        effective_format = "html"
        print("Warning: --html is deprecated. Use --format html instead.", file=sys.stderr)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/generate_performance_report.py",
        "--start-date", str(start_date),
        "--end-date", str(end_date),
    ]
    
    if effective_format != "text":
        cmd.extend(["--format", effective_format])
    
    if args.no_charts:
        cmd.append("--no-charts")
    
    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"\n✅ {period_name.upper()} report completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {period_name.upper()} report failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {period_name.upper()} report interrupted by user\n")
        return False


def run_simulation_with_capital(period_name: str, start_date: date, end_date: date, initial_capital: float, args):
    """Run simulate_daily_trading.py with specific capital, then generate report from JSON."""
    print(f"\n{'='*80}")
    print(f"Generating {period_name.upper()} report with ${initial_capital:,.0f} initial capital")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate JSON filename
    json_filename = f"sim_{period_name}_{int(initial_capital)}.json"
    json_path = output_dir / json_filename
    
    # Step 1: Run simulation
    print(f"Step 1: Running simulation...")
    sim_cmd = [
        sys.executable,
        "scripts/simulate_daily_trading.py",
        "--start-date", str(start_date),
        "--end-date", str(end_date),
        "--initial-capital", str(initial_capital),
        "--output-json",
    ]
    
    if args.top_k:
        sim_cmd.extend(["--top-k", str(args.top_k)])
    
    try:
        with open(json_path, 'w') as f:
            result = subprocess.run(sim_cmd, check=True, stdout=f, cwd=Path(__file__).parent.parent)
        print(f"✅ Simulation completed, saved to {json_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"⚠️  Simulation interrupted by user\n")
        return False
    
    # Step 2: Generate report from JSON
    print(f"Step 2: Generating report from JSON...")
    
    # Determine effective format (handle backward compatibility with --html)
    effective_format = args.format
    if args.html and args.format == "text":
        effective_format = "html"
        print("Warning: --html is deprecated. Use --format html instead.", file=sys.stderr)
    
    report_cmd = [
        sys.executable,
        "scripts/generate_performance_report.py",
        "--from-json", str(json_path),
        "--report-name", f"{period_name}_{int(initial_capital)}",
    ]
    
    if effective_format != "text":
        report_cmd.extend(["--format", effective_format])
    
    if args.no_charts:
        report_cmd.append("--no-charts")
    
    try:
        result = subprocess.run(report_cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ Report generated successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Report generation failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"⚠️  Report generation interrupted by user\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run performance reports for recent periods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--periods", nargs="+", choices=["week", "mtd", "ytd"], default=["week", "mtd", "ytd"],
        help="Which periods to run (default: all three)"
    )
    parser.add_argument(
        "--format", type=str, choices=["text", "html", "pdf", "all"], default="text",
        help="Output format: 'text' (default), 'html', 'pdf', or 'all' (html+pdf)"
    )
    parser.add_argument(
        "--html", action="store_true",
        help="[DEPRECATED] Generate HTML reports. Use --format html instead."
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation (text/JSON only, faster)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top stocks to hold (default: 5)"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=None,
        help="Use specific initial capital and generate separate reports (default: use built-in 100000)"
    )
    parser.add_argument(
        "--capital-variants", action="store_true",
        help="Run reports for both $10k and $100k initial capital"
    )
    
    args = parser.parse_args()
    
    # Get date ranges
    date_ranges = get_date_ranges()
    
    # Determine which periods to run (in order: week, mtd, ytd)
    periods_to_run = []
    if "week" in args.periods:
        periods_to_run.append(("week", date_ranges["week"]))
    if "mtd" in args.periods:
        periods_to_run.append(("mtd", date_ranges["mtd"]))
    if "ytd" in args.periods:
        periods_to_run.append(("ytd", date_ranges["ytd"]))
    
    if not periods_to_run:
        print("Error: No valid periods specified")
        sys.exit(1)
    
    # Run reports
    success_count = 0
    total_count = 0
    
    if args.capital_variants:
        # Run for both $10k and $100k
        capitals = [10000, 100000]
        for capital in capitals:
            print(f"\n{'#'*80}")
            print(f"Running reports with ${capital:,.0f} initial capital")
            print(f"{'#'*80}")
            for period_name, (start_date, end_date) in periods_to_run:
                total_count += 1
                if run_simulation_with_capital(period_name, start_date, end_date, capital, args):
                    success_count += 1
    elif args.initial_capital:
        # Run with specific capital
        for period_name, (start_date, end_date) in periods_to_run:
            total_count += 1
            if run_simulation_with_capital(period_name, start_date, end_date, args.initial_capital, args):
                success_count += 1
    else:
        # Use default (built-in 100000 in generate_performance_report.py)
        for period_name, (start_date, end_date) in periods_to_run:
            total_count += 1
            if run_report(period_name, start_date, end_date, args):
                success_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {success_count}/{total_count} reports completed successfully")
    print(f"{'='*80}\n")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()

