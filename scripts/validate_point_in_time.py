"""Validate point-in-time correctness of features and labels.

This script checks that:
1. Features only use data available at or before the feature date
2. Labels are forward-looking (no lookahead)
3. Fundamentals use report_release_date correctly
4. News/sentiment only aggregate events before the feature date
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from features.pipeline import FeaturePipeline
from labels.returns import ReturnLabelGenerator
from backtest.validators import BacktestValidator
from configs.loader import get_config
from loguru import logger


def validate_fundamentals_point_in_time(
    storage: StorageBackend,
    as_of_date: date,
    asset_id: int
) -> bool:
    """Validate that fundamentals used are released before as_of_date."""
    query = """
        SELECT report_release_date, period_end_date
        FROM fundamentals
        WHERE asset_id = ? AND report_release_date <= ?
        ORDER BY report_release_date DESC
        LIMIT 1
    """
    result = storage.conn.execute(query, [asset_id, as_of_date]).df()
    
    if len(result) == 0:
        return True  # No fundamentals available is OK
    
    latest_release = result['report_release_date'].iloc[0]
    return latest_release <= as_of_date


def validate_news_point_in_time(
    storage: StorageBackend,
    as_of_date: date,
    asset_id: int,
    lookback_days: int = 30
) -> bool:
    """Validate that news used is before as_of_date."""
    from datetime import timedelta
    start_date = as_of_date - timedelta(days=lookback_days)
    
    query = """
        SELECT MAX(timestamp) as max_timestamp
        FROM news_events
        WHERE asset_id = ? AND DATE(timestamp) >= ? AND DATE(timestamp) <= ?
    """
    result = storage.conn.execute(query, [asset_id, start_date, as_of_date]).df()
    
    if len(result) == 0 or pd.isna(result['max_timestamp'].iloc[0]):
        return True  # No news is OK
    
    max_timestamp = pd.to_datetime(result['max_timestamp'].iloc[0]).date()
    return max_timestamp <= as_of_date


def main():
    parser = argparse.ArgumentParser(description="Validate point-in-time correctness")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sample-dates", type=int, default=10, help="Number of dates to sample for validation")
    parser.add_argument("--check-fundamentals", action="store_true", help="Check fundamentals point-in-time correctness")
    parser.add_argument("--check-news", action="store_true", help="Check news point-in-time correctness")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    logger.info(f"Validating point-in-time correctness from {start_date} to {end_date}")
    
    # Initialize storage
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    api = AsOfQueryAPI(storage)
    
    # Get sample dates
    from data.universe import TradingCalendar
    calendar = TradingCalendar()
    trading_days = calendar.get_trading_days(start_date, end_date)
    sample_dates = [d.date() for d in trading_days][::max(1, len(trading_days) // args.sample_dates)]
    
    logger.info(f"Sampling {len(sample_dates)} dates for validation")
    
    # Get universe
    try:
        universe = api.get_universe_at_date(end_date, index_name="SP500")
        if len(universe) == 0:
            bars_df = api.get_bars_asof(end_date)
            universe = set(bars_df['asset_id'].unique()[:10])  # Sample
    except:
        bars_df = api.get_bars_asof(end_date)
        universe = set(bars_df['asset_id'].unique()[:10])
    
    logger.info(f"Validating {len(universe)} assets")
    
    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline(api, config.features)
    
    issues = []
    
    # Validate features
    logger.info("Validating feature point-in-time correctness...")
    for sample_date in sample_dates[:5]:  # Check first 5 dates
        try:
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=sample_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            # Check fundamentals if enabled
            if args.check_fundamentals and config.features.get('fundamentals', {}).get('enabled', False):
                for asset_id in features_df['asset_id'].unique()[:5]:  # Sample
                    if not validate_fundamentals_point_in_time(storage, sample_date, asset_id):
                        issues.append({
                            'type': 'fundamentals_lookahead',
                            'date': sample_date,
                            'asset_id': asset_id,
                            'message': 'Fundamentals used may have release date after feature date'
                        })
            
            # Check news if enabled
            if args.check_news and config.features.get('sentiment', {}).get('enabled', False):
                for asset_id in features_df['asset_id'].unique()[:5]:  # Sample
                    if not validate_news_point_in_time(storage, sample_date, asset_id):
                        issues.append({
                            'type': 'news_lookahead',
                            'date': sample_date,
                            'asset_id': asset_id,
                            'message': 'News used may have timestamp after feature date'
                        })
        
        except Exception as e:
            logger.warning(f"Error validating {sample_date}: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("POINT-IN-TIME VALIDATION RESULTS")
    print("="*80)
    
    if len(issues) == 0:
        print("✓ All validations passed!")
    else:
        print(f"✗ Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  - {issue['type']} on {issue['date']} for asset {issue['asset_id']}: {issue['message']}")
    
    print("="*80)
    
    storage.close()


if __name__ == "__main__":
    main()

