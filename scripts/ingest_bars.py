"""Script to ingest daily bars from vendors."""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.vendors.yahoo import YahooClient
from data.vendors.tiingo import TiingoClient
from data.normalize import DataNormalizer
from data.quality import DataQualityChecker
from configs.loader import get_config


def main():
    parser = argparse.ArgumentParser(description="Ingest daily bars from vendors")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols to ingest")
    parser.add_argument("--vendor", type=str, default="yahoo", choices=["yahoo", "tiingo"], help="Vendor to use")
    parser.add_argument("--check-quality", action="store_true", help="Run quality checks after ingestion")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    # Initialize vendor client
    if args.vendor == "yahoo":
        vendor_client = YahooClient()
    elif args.vendor == "tiingo":
        vendor_client = TiingoClient()
    else:
        raise ValueError(f"Unknown vendor: {args.vendor}")
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        # Default to SPY for testing
        symbols = ["SPY"]
    
    print(f"Ingesting bars for {len(symbols)} symbols from {start_date} to {end_date} using {args.vendor}")
    
    # Fetch bars
    bars_df = vendor_client.fetch_daily_bars(symbols, start_date, end_date)
    
    if len(bars_df) == 0:
        print("No bars fetched")
        return
    
    print(f"Fetched {len(bars_df)} bars")
    
    # Normalize
    normalizer = DataNormalizer(storage)
    normalized_bars = normalizer.normalize_bars(bars_df, vendor=args.vendor)
    
    print(f"Normalized to {len(normalized_bars)} records")
    
    # Save to Parquet
    storage.save_parquet(normalized_bars, "bars_daily")
    
    # Insert into DuckDB
    storage.insert_dataframe("bars_daily", normalized_bars, if_exists="append")
    
    print("Saved to Parquet and DuckDB")
    
    # Quality checks
    if args.check_quality:
        checker = DataQualityChecker(storage)
        report = checker.generate_quality_report(normalized_bars, start_date=start_date, end_date=end_date)
        print("\nQuality Report:")
        print(f"  Total assets: {report['total_assets']}")
        print(f"  Total bars: {report['total_bars']}")
        print(f"  Issues: {report['issues']}")
    
    storage.close()
    print("Done")


if __name__ == "__main__":
    main()

