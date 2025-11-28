#!/usr/bin/env python
"""Ensure data coverage for bars_daily.

This script checks data coverage against configured requirements and
automatically backfills missing data via the configured vendor.

Usage:
    # Ensure full history coverage (from config min_history_start_date to today)
    python scripts/ensure_data_coverage.py --mode full-history

    # Daily top-up (just recent days)
    python scripts/ensure_data_coverage.py --mode daily-top-up

    # Custom date range
    python scripts/ensure_data_coverage.py --mode custom --start-date 2023-01-01 --end-date 2024-12-31

    # Check coverage without fetching
    python scripts/ensure_data_coverage.py --mode full-history --check-only

    # Specify symbols explicitly
    python scripts/ensure_data_coverage.py --mode full-history --symbols AAPL MSFT GOOGL

    # Generate coverage report
    python scripts/ensure_data_coverage.py --report
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.maintenance import DataMaintenanceManager, ensure_data_coverage
from configs.loader import get_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Ensure data coverage for bars_daily",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full-history", "daily-top-up", "custom"],
        default="full-history",
        help="Coverage mode: full-history (from config start to today), daily-top-up (recent days only), or custom"
    )
    
    # Custom date range
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for custom mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for custom mode (YYYY-MM-DD, default: today)"
    )
    
    # Symbol selection
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to check/fetch (default: from config or all in database)"
    )
    
    # Vendor selection
    parser.add_argument(
        "--vendor",
        type=str,
        choices=["yahoo", "tiingo"],
        help="Data vendor to use (default: from config)"
    )
    
    # Behavior flags
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check coverage, don't fetch missing data"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and print a coverage report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    # Load config
    config = get_config()
    
    # Initialize storage
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    try:
        # Handle report mode
        if args.report:
            manager = DataMaintenanceManager(storage, config.__dict__)
            report = manager.get_coverage_report()
            
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print_coverage_report(report, storage=storage)
            
            storage.close()
            return
        
        # Parse dates for custom mode
        target_start = None
        target_end = None
        
        if args.mode == "custom":
            if not args.start_date:
                parser.error("--start-date is required for --mode custom")
            target_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        
        if args.end_date:
            target_end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        
        # Run coverage check/fetch
        result = ensure_data_coverage(
            storage=storage,
            config=config.__dict__,
            mode=args.mode,
            target_start=target_start,
            target_end=target_end,
            symbols=args.symbols,
            vendor=args.vendor,
            auto_fetch=not args.check_only
        )
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_result(result, args.check_only)
        
        # Exit code based on status
        if result['status'] == 'ok':
            sys.exit(0)
        elif result['status'] == 'partial':
            sys.exit(0)  # Partial success is still success
        elif result['status'] == 'gaps_found' and args.check_only:
            sys.exit(1)  # Gaps found in check-only mode
        else:
            sys.exit(1)
    
    finally:
        storage.close()


def print_coverage_report(report: dict, storage=None):
    """Print a human-readable coverage report."""
    print("\n" + "=" * 60)
    print("DATA COVERAGE REPORT")
    print("=" * 60)
    
    summary = report.get('summary', {})
    if summary.get('has_data'):
        print(f"\nDate Range: {summary['min_date']} to {summary['max_date']}")
        print(f"Total Bars: {summary['total_bars']:,}")
        print(f"Assets: {summary['num_assets']}")
    else:
        print("\nNo data in database.")
    
    config = report.get('config', {})
    print(f"\nConfig:")
    print(f"  Min History Start: {config.get('min_history_start_date')}")
    print(f"  Max History Lag: {config.get('max_history_lag_days')} days")
    print(f"  Default Vendor: {config.get('default_vendor')}")
    
    # Show universe coverage if storage is provided
    if storage:
        try:
            universe_stats = storage.query("""
                SELECT 
                    um.index_name,
                    COUNT(DISTINCT um.asset_id) as universe_members,
                    COUNT(DISTINCT CASE WHEN bd.asset_id IS NOT NULL THEN um.asset_id END) as members_with_data
                FROM (
                    SELECT DISTINCT asset_id, index_name 
                    FROM universe_membership
                ) um
                LEFT JOIN (
                    SELECT DISTINCT asset_id FROM bars_daily
                ) bd ON um.asset_id = bd.asset_id
                GROUP BY um.index_name
            """)
            
            if len(universe_stats) > 0:
                print(f"\nUniverse Coverage:")
                for _, row in universe_stats.iterrows():
                    total = row['universe_members']
                    with_data = row['members_with_data']
                    missing = total - with_data
                    pct = (with_data / total * 100) if total > 0 else 0
                    status = "✓" if missing == 0 else "!"
                    print(f"  {status} {row['index_name']}: {with_data}/{total} ({pct:.1f}%) have data")
                    if missing > 0:
                        print(f"    ⚠ {missing} symbols missing price data!")
        except Exception:
            pass
    
    metrics = report.get('metrics')
    if metrics:
        print(f"\nPer-Asset Metrics:")
        print(f"  Assets with data: {metrics['assets_with_data']}")
        print(f"  Avg bars/asset: {metrics['avg_bars_per_asset']:.0f}")
        print(f"  Min bars/asset: {metrics['min_bars_per_asset']}")
        print(f"  Max bars/asset: {metrics['max_bars_per_asset']}")
    
    per_asset = report.get('per_asset', [])
    if per_asset and len(per_asset) <= 30:
        print(f"\nPer-Asset Coverage ({len(per_asset)} assets):")
        print(f"  {'Symbol':<10} {'Min Date':<12} {'Max Date':<12} {'Bars':>8}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        for asset in per_asset:
            print(f"  {asset['symbol']:<10} {str(asset['min_date']):<12} {str(asset['max_date']):<12} {asset['bar_count']:>8}")
    elif per_asset:
        print(f"\nPer-Asset Coverage: {len(per_asset)} assets (too many to display)")
    
    print("=" * 60 + "\n")


def print_result(result: dict, check_only: bool):
    """Print a human-readable result summary."""
    print("\n" + "-" * 60)
    print(f"Mode: {result['mode']}")
    print(f"Target Range: {result.get('target_start')} to {result.get('target_end')}")
    print(f"Symbols: {result.get('symbols_count', 'N/A')}")
    print("-" * 60)
    
    status = result['status']
    status_emoji = {
        'ok': '✓',
        'partial': '~',
        'warning': '!',
        'gaps_found': '!',
        'error': '✗'
    }.get(status, '?')
    
    print(f"\nStatus: {status_emoji} {status.upper()}")
    print(f"Message: {result.get('message', 'N/A')}")
    
    if result.get('gaps_identified'):
        gaps = result.get('gaps', {})
        print(f"\nGaps Identified:")
        for gap_range in gaps.get('backfill_ranges', []):
            print(f"  - {gap_range[0]} to {gap_range[1]}")
        if gaps.get('missing_symbols'):
            print(f"  Missing symbols: {', '.join(gaps['missing_symbols'][:10])}")
            if len(gaps['missing_symbols']) > 10:
                print(f"    ... and {len(gaps['missing_symbols']) - 10} more")
    
    if result.get('fetch_attempted'):
        print(f"\nFetch Results:")
        for fetch in result.get('fetch_results', []):
            r = fetch['result']
            status_str = "OK" if r['success'] else "FAILED"
            print(f"  {fetch['range'][0]} to {fetch['range'][1]}: {status_str}")
            if r['success']:
                print(f"    Bars: {r['bars_fetched']}, Symbols: {r['symbols_fetched']}")
            if r.get('errors'):
                for err in r['errors']:
                    print(f"    Error: {err}")
    elif check_only and result.get('gaps_identified'):
        print(f"\nNote: Use without --check-only to automatically fetch missing data")
    
    coverage_after = result.get('coverage_after')
    if coverage_after and coverage_after.get('has_data'):
        print(f"\nCurrent Coverage:")
        print(f"  Date Range: {coverage_after['min_date']} to {coverage_after['max_date']}")
        print(f"  Total Bars: {coverage_after['total_bars']:,}")
        print(f"  Assets: {coverage_after['num_assets']}")
    
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()

