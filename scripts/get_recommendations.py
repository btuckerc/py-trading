"""Generate current-day portfolio recommendations.

This script builds features as-of today (or the last available trading day),
trains a model on historical data, and outputs ranked recommendations with
suggested portfolio weights.
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from data.universe import TradingCalendar
from labels.returns import ReturnLabelGenerator
from features.pipeline import FeaturePipeline
from models.tabular import XGBoostModel, LightGBMModel
from portfolio.strategies import LongTopKStrategy
from configs.loader import get_config
from loguru import logger


def get_universe_asset_ids(storage: StorageBackend, symbols: list) -> set:
    """Get asset_ids for given symbols."""
    symbol_list = "', '".join(symbols)
    query = f"SELECT asset_id, symbol FROM assets WHERE symbol IN ('{symbol_list}')"
    df = storage.query(query)
    return set(df['asset_id'].values) if len(df) > 0 else set(), dict(zip(df['asset_id'], df['symbol']))


def main():
    parser = argparse.ArgumentParser(description="Generate current-day portfolio recommendations")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to analyze (default: all in database)")
    parser.add_argument("--as-of-date", type=str, help="Date to generate recommendations for (YYYY-MM-DD, default: today)")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm"], help="Model type")
    parser.add_argument("--horizon", type=int, default=20, help="Prediction horizon (days)")
    parser.add_argument("--train-days", type=int, default=1000, help="Number of days of training data to use")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top assets to recommend")
    
    args = parser.parse_args()
    
    # Determine as-of date
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    else:
        # Use last trading day
        calendar = TradingCalendar()
        today = date.today()
        try:
            as_of_date = calendar.previous_trading_day(today)
        except:
            as_of_date = today
    
    logger.info(f"Generating recommendations as-of {as_of_date}")
    
    # Initialize storage and API
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    api = AsOfQueryAPI(storage)
    
    # Get universe
    if args.symbols:
        universe, asset_id_to_symbol = get_universe_asset_ids(storage, args.symbols)
        logger.info(f"Using specified symbols: {args.symbols}")
    else:
        # Try to use universe_membership table (survivorship-bias-free)
        try:
            universe = api.get_universe_at_date(as_of_date, index_name="SP500")
            if len(universe) > 0:
                logger.info(f"Using S&P 500 universe from universe_membership table: {len(universe)} assets")
            else:
                # Fallback: get all asset_ids from bars_daily
                bars_df = api.get_bars_asof(as_of_date)
                universe = set(bars_df['asset_id'].unique())
                logger.info(f"Universe_membership table empty, using all assets in database: {len(universe)} assets")
        except Exception as e:
            # Fallback: get all asset_ids from bars_daily
            logger.warning(f"Could not load universe from universe_membership table: {e}")
            bars_df = api.get_bars_asof(as_of_date)
            universe = set(bars_df['asset_id'].unique())
            logger.info(f"Using all assets in database: {len(universe)} assets")
        
        # Get symbol mapping
        assets_df = storage.query("SELECT asset_id, symbol FROM assets")
        asset_id_to_symbol = dict(zip(assets_df['asset_id'], assets_df['symbol']))
    
    if len(universe) == 0:
        logger.error("No assets found in universe")
        return
    
    logger.info(f"Universe size: {len(universe)} assets")
    
    # Get training period
    train_start = pd.Timestamp(as_of_date) - pd.Timedelta(days=args.train_days)
    train_start = train_start.date()
    
    logger.info(f"Training period: {train_start} to {as_of_date}")
    
    # Generate labels for training
    logger.info("Generating return labels for training...")
    label_generator = ReturnLabelGenerator(storage)
    labels_df = label_generator.generate_labels(
        start_date=train_start,
        end_date=as_of_date,
        horizons=[args.horizon],
        benchmark_symbol="SPY",
        universe=list(universe)
    )
    
    if len(labels_df) == 0:
        logger.error("No labels generated")
        return
    
    logger.info(f"Generated {len(labels_df)} labels")
    
    # Initialize feature pipeline
    logger.info("Initializing feature pipeline...")
    feature_pipeline = FeaturePipeline(api, config.features)
    
    # Get trading calendar
    calendar = TradingCalendar()
    trading_days = calendar.get_trading_days(train_start, as_of_date)
    
    # Build features and train model
    logger.info("Building features for training...")
    train_features_list = []
    train_labels_list = []
    
    # Sample training dates (every 5 days)
    train_dates = [d.date() for d in trading_days if train_start <= d.date() < as_of_date][::5]
    logger.info(f"Sampling {len(train_dates)} training dates")
    
    successful_dates = 0
    for train_date in train_dates:
        try:
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=train_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            # Ensure date and asset_id columns exist
            if 'date' not in features_df.columns:
                features_df['date'] = train_date
            if 'asset_id' not in features_df.columns:
                continue
            
            # Get labels for this date
            date_labels = labels_df[
                (labels_df['date'] == train_date) & 
                (labels_df['horizon'] == args.horizon)
            ]
            
            if len(date_labels) == 0:
                continue
            
            # Merge features with labels
            merged = features_df.merge(
                date_labels[['asset_id', 'target_excess_log_return']],
                on='asset_id',
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Extract feature columns
            feature_cols = [c for c in merged.columns if c not in ['asset_id', 'date', 'target_excess_log_return']]
            X = merged[feature_cols].copy()
            # Convert to numeric and fill NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
            y = merged['target_excess_log_return'].values
            
            train_features_list.append(X)
            train_labels_list.append(y)
            successful_dates += 1
            
        except Exception as e:
            if successful_dates == 0:
                logger.warning(f"Error building features for {train_date}: {e}")
            continue
    
    logger.info(f"Successfully built features for {successful_dates} training dates")
    
    if len(train_features_list) == 0:
        logger.error("No training features generated")
        return
    
    # Combine training data
    X_train = pd.concat(train_features_list, ignore_index=True)
    y_train = np.concatenate(train_labels_list)
    
    logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
    
    # Train model
    logger.info(f"Training {args.model} model...")
    if args.model == "xgboost":
        model = XGBoostModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
    else:
        model = LightGBMModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
    
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    # Build features for as-of date
    logger.info(f"Building features for {as_of_date}...")
    features_df = feature_pipeline.build_features_cross_sectional(
        as_of_date=as_of_date,
        universe=universe,
        lookback_days=252
    )
    
    if len(features_df) == 0:
        logger.error("No features generated for as-of date")
        return
    
    # Ensure date column exists
    if 'date' not in features_df.columns:
        features_df['date'] = as_of_date
    
    # Extract feature columns
    feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
    X = features_df[feature_cols].copy()
    # Convert to numeric and fill NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Predict
    logger.info("Generating predictions...")
    predictions = model.predict(X)
    
    # Create scores DataFrame
    scores_df = pd.DataFrame({
        'asset_id': features_df['asset_id'].values,
        'score': predictions,
        'confidence': np.ones(len(predictions))  # Placeholder
    })
    
    # Compute weights using strategy
    logger.info("Computing portfolio weights...")
    strategy = LongTopKStrategy(k=args.top_k, min_score_threshold=-np.inf)
    weights_df = strategy.compute_weights(scores_df, as_of_date=as_of_date)
    
    # Map asset_ids to symbols
    weights_df['symbol'] = weights_df['asset_id'].map(asset_id_to_symbol)
    
    # Sort by weight (descending)
    weights_df = weights_df.sort_values('weight', ascending=False)
    
    # Print recommendations
    print("\n" + "="*70)
    print("PORTFOLIO RECOMMENDATIONS")
    print("="*70)
    print(f"As-of Date: {as_of_date}")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon} days")
    print(f"Top K: {args.top_k}")
    print(f"\nRecommended Positions:")
    print("-"*70)
    print(f"{'Symbol':<10} {'Asset ID':<10} {'Weight':<12} {'Score':<12}")
    print("-"*70)
    
    total_weight = 0.0
    for _, row in weights_df.iterrows():
        symbol = row.get('symbol', f"ID_{row['asset_id']}")
        asset_id = row['asset_id']
        weight = row['weight']
        score = scores_df[scores_df['asset_id'] == asset_id]['score'].iloc[0]
        total_weight += weight
        print(f"{symbol:<10} {asset_id:<10} {weight:>10.2%} {score:>12.6f}")
    
    print("-"*70)
    print(f"{'Total Weight':<22} {total_weight:>10.2%}")
    print("="*70)
    
    # Save to CSV
    output_path = Path("artifacts") / "recommendations"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(output_path / f"recommendations_{as_of_date}_{args.model}.csv", index=False)
    logger.info(f"Recommendations saved to {output_path / f'recommendations_{as_of_date}_{args.model}.csv'}")
    
    storage.close()


if __name__ == "__main__":
    main()

