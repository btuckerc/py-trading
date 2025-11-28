#!/usr/bin/env python
"""Refresh universe membership table from historical constituents CSV.

This script builds or rebuilds the universe_membership table in DuckDB
from a historical constituents CSV file.

Usage:
    # Rebuild from scratch (default)
    python scripts/refresh_universe_membership.py --mode rebuild

    # Incremental update (append new data)
    python scripts/refresh_universe_membership.py --mode incremental

    # Use custom CSV and index name
    python scripts/refresh_universe_membership.py --csv-path data/my_universe.csv --index-name MY_INDEX

    # Show current universe status
    python scripts/refresh_universe_membership.py --status
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.universe import refresh_universe_membership, UniverseManager
from configs.loader import get_config
from loguru import logger


def print_universe_status(storage: StorageBackend, index_name: str = "SP500"):
    """Print current universe membership status."""
    print("\n" + "=" * 60)
    print("UNIVERSE MEMBERSHIP STATUS")
    print("=" * 60)
    
    try:
        # Get overall stats
        result = storage.query(f"""
            SELECT 
                index_name,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(*) as total_records,
                COUNT(DISTINCT asset_id) as unique_assets,
                COUNT(DISTINCT date) as trading_days
            FROM universe_membership
            WHERE index_name = '{index_name}'
            GROUP BY index_name
        """)
        
        if len(result) == 0:
            print(f"\nNo membership data found for index: {index_name}")
            print("Run with --mode rebuild to populate the table.")
            print("=" * 60 + "\n")
            return
        
        row = result.iloc[0]
        print(f"\nIndex: {row['index_name']}")
        print(f"Date Range: {row['min_date']} to {row['max_date']}")
        print(f"Trading Days: {row['trading_days']}")
        print(f"Unique Assets: {row['unique_assets']}")
        print(f"Total Records: {row['total_records']:,}")
        
        # Get sample membership counts
        print("\nSample Membership Counts:")
        sample = storage.query(f"""
            SELECT date, COUNT(DISTINCT asset_id) as members
            FROM universe_membership
            WHERE index_name = '{index_name}' AND in_index = TRUE
            GROUP BY date
            ORDER BY date
            LIMIT 5
        """)
        
        for _, r in sample.iterrows():
            print(f"  {r['date']}: {r['members']} members")
        
        # Get latest membership count
        latest = storage.query(f"""
            SELECT date, COUNT(DISTINCT asset_id) as members
            FROM universe_membership
            WHERE index_name = '{index_name}' AND in_index = TRUE
            GROUP BY date
            ORDER BY date DESC
            LIMIT 1
        """)
        
        if len(latest) > 0:
            print(f"\nLatest: {latest.iloc[0]['date']}: {latest.iloc[0]['members']} members")
        
    except Exception as e:
        print(f"\nError getting status: {e}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Refresh universe membership table from historical constituents CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rebuild", "incremental"],
        default="rebuild",
        help="Mode: rebuild (truncate + full rebuild) or incremental (append new)"
    )
    
    # Custom paths
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to constituents CSV (default: from config)"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help="Index name (default: from config, typically SP500)"
    )
    
    # Date range
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD, default: earliest in CSV)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD, default: today)"
    )
    
    # Data fetching
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        default=True,
        help="Fetch price data for symbols without any (default: True)"
    )
    parser.add_argument(
        "--no-fetch-data",
        action="store_true",
        help="Skip fetching price data for new symbols"
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default="yahoo",
        choices=["yahoo", "tiingo"],
        help="Data vendor to use for fetching (default: yahoo)"
    )
    
    # Status and output
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current universe membership status and exit"
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
        # Determine index name (for status display)
        index_name = args.index_name or config.universe.index_name
        
        # Handle status mode
        if args.status:
            print_universe_status(storage, index_name)
            storage.close()
            return
        
        # Determine if we should fetch data
        fetch_data = args.fetch_data and not args.no_fetch_data
        
        # Run refresh
        logger.info(f"Refreshing universe membership (mode: {args.mode}, fetch_data: {fetch_data})")
        
        result = refresh_universe_membership(
            storage=storage,
            config=config,
            mode=args.mode,
            index_name=args.index_name,
            csv_path=args.csv_path,
            fetch_missing_data=fetch_data,
            vendor=args.vendor
        )
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n" + "-" * 60)
            print("REFRESH RESULT")
            print("-" * 60)
            
            status_emoji = "✓" if result['success'] else "✗"
            print(f"\nStatus: {status_emoji} {'SUCCESS' if result['success'] else 'FAILED'}")
            
            if result['date_range']:
                print(f"Date Range: {result['date_range'][0]} to {result['date_range'][1]}")
            
            print(f"Records Created: {result['records_created']:,}")
            print(f"Unique Assets: {result['unique_assets']}")
            
            # Show data fetch results if available
            if 'data_fetch' in result:
                df = result['data_fetch']
                print(f"\nData Fetch:")
                print(f"  Symbols Checked: {df.get('symbols_checked', 0)}")
                print(f"  Symbols Missing Data: {df.get('symbols_missing_data', 0)}")
                print(f"  Symbols Fetched: {df.get('symbols_fetched', 0)}")
                if df.get('errors'):
                    for err in df['errors']:
                        print(f"  ⚠ {err}")
            
            if result['errors']:
                print(f"\nErrors/Warnings:")
                for err in result['errors']:
                    print(f"  - {err}")
            
            print("-" * 60 + "\n")
        
        # Exit code
        sys.exit(0 if result['success'] else 1)
    
    finally:
        storage.close()


if __name__ == "__main__":
    main()

