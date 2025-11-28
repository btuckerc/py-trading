"""Build universe membership table from historical S&P 500 constituents.

This script reads a CSV file with historical S&P 500 membership changes and
populates the universe_membership table in DuckDB, ensuring point-in-time
correctness for backtests.
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.normalize import DataNormalizer
from data.universe import UniverseManager, TradingCalendar
from configs.loader import get_config
from loguru import logger


def get_symbol_to_asset_id(storage: StorageBackend) -> dict:
    """Get mapping from symbol to asset_id."""
    assets_df = storage.query("SELECT asset_id, symbol FROM assets")
    if len(assets_df) == 0:
        return {}
    return dict(zip(assets_df['symbol'], assets_df['asset_id']))


def main():
    parser = argparse.ArgumentParser(description="Build universe membership table from historical constituents")
    parser.add_argument("--csv-path", type=str, help="Path to S&P 500 constituents CSV (default: data/sp500_constituents.csv)")
    parser.add_argument("--start-date", type=str, help="Start date for universe (YYYY-MM-DD, default: earliest in CSV)")
    parser.add_argument("--end-date", type=str, help="End date for universe (YYYY-MM-DD, default: latest in CSV or today)")
    parser.add_argument("--index-name", type=str, default="SP500", help="Index name (default: SP500)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing universe_membership data")
    
    args = parser.parse_args()
    
    # Initialize storage
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    normalizer = DataNormalizer(storage)
    universe_manager = UniverseManager(db_path=config.database.duckdb_path)
    
    # Load constituents CSV
    csv_path = args.csv_path or "data/sp500_constituents.csv"
    logger.info(f"Loading constituents from {csv_path}")
    
    constituents_df = universe_manager.load_sp500_constituents(csv_path)
    
    if len(constituents_df) == 0:
        logger.error(f"No constituents data found in {csv_path}")
        logger.info("Expected CSV format:")
        logger.info("  date,symbol,action")
        logger.info("  2020-01-01,AAPL,added")
        logger.info("  2020-06-01,XYZ,removed")
        return
    
    logger.info(f"Loaded {len(constituents_df)} constituent changes")
    
    # Determine date range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = constituents_df['date'].min()
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = max(constituents_df['date'].max(), date.today())
    
    logger.info(f"Building universe membership from {start_date} to {end_date}")
    
    # Build membership DataFrame (with symbols)
    membership_df = universe_manager.build_universe_membership(
        start_date=start_date,
        end_date=end_date,
        index_name=args.index_name
    )
    
    if len(membership_df) == 0:
        logger.warning("No membership records generated. Check CSV format and date range.")
        return
    
    logger.info(f"Generated {len(membership_df)} membership records")
    
    # Check if we have symbol or asset_id column
    if 'symbol' not in membership_df.columns and 'asset_id' not in membership_df.columns:
        logger.error("Membership DataFrame missing both 'symbol' and 'asset_id' columns")
        return
    
    # If we have symbol column, map to asset_ids
    if 'symbol' in membership_df.columns:
        logger.info("Mapping symbols to asset_ids...")
        symbol_to_asset_id = get_symbol_to_asset_id(storage)
        
        # Get unique symbols from membership
        unique_symbols = set(membership_df['symbol'].unique())
        
        # Create asset_ids for symbols that don't exist yet
        missing_symbols = unique_symbols - set(symbol_to_asset_id.keys())
        if len(missing_symbols) > 0:
            logger.info(f"Creating asset_ids for {len(missing_symbols)} new symbols...")
            for symbol in missing_symbols:
                # This will create the asset_id via DataNormalizer
                normalizer._get_or_create_asset_id(symbol)
            
            # Reload mapping
            symbol_to_asset_id = get_symbol_to_asset_id(storage)
        
        # Replace symbol with asset_id
        membership_df['asset_id'] = membership_df['symbol'].map(symbol_to_asset_id)
        membership_df = membership_df.drop(columns=['symbol'])
    # else: asset_id already exists (from previous run or different format)
    
    # Remove rows with missing asset_ids
    before_count = len(membership_df)
    membership_df = membership_df.dropna(subset=['asset_id'])
    if len(membership_df) < before_count:
        logger.warning(f"Removed {before_count - len(membership_df)} rows with missing asset_ids")
    
    # Ensure asset_id is integer
    membership_df['asset_id'] = membership_df['asset_id'].astype(int)
    
    # Select final columns (ensure we have the right ones)
    required_columns = ['date', 'asset_id', 'index_name', 'in_index']
    for col in required_columns:
        if col not in membership_df.columns:
            logger.error(f"Missing required column: {col}")
            return
    
    membership_df = membership_df[required_columns]
    
    logger.info(f"Final membership table: {len(membership_df)} records")
    logger.info(f"  Date range: {membership_df['date'].min()} to {membership_df['date'].max()}")
    logger.info(f"  Unique assets: {membership_df['asset_id'].nunique()}")
    
    # Save to database
    if args.overwrite:
        logger.info("Overwriting existing universe_membership table...")
        storage.conn.execute("DELETE FROM universe_membership WHERE index_name = ?", [args.index_name])
    
    logger.info("Inserting into universe_membership table...")
    storage.insert_dataframe('universe_membership', membership_df, if_exists='append')
    
    # Also save to Parquet
    logger.info("Saving to Parquet...")
    storage.save_parquet(membership_df, 'universe_membership')
    
    logger.info("Universe membership table built successfully!")
    
    # Print summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Total records: {len(membership_df)}")
    logger.info(f"  Unique assets: {membership_df['asset_id'].nunique()}")
    logger.info(f"  Date range: {membership_df['date'].min()} to {membership_df['date'].max()}")
    
    # Count members per date (sample)
    sample_dates = sorted(membership_df['date'].unique())[::max(1, len(membership_df['date'].unique()) // 10)]
    logger.info("\nSample membership counts:")
    for sample_date in sample_dates[:5]:
        count = len(membership_df[(membership_df['date'] == sample_date) & (membership_df['in_index'])])
        logger.info(f"  {sample_date}: {count} assets")
    
    storage.close()


if __name__ == "__main__":
    main()

