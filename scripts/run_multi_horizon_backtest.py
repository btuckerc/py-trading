"""Multi-horizon, uncertainty-aware backtest script.

This script runs a complete ML trading pipeline with:
- Multi-horizon return predictions (1d, 5d, 20d, 120d)
- Uncertainty estimation (MC dropout for sequence models, ensemble/residual for tabular)
- Risk-adjusted score combination via ScoreConverter
- Portfolio construction with confidence thresholds
- Benchmark comparisons
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from data.universe import TradingCalendar
from labels.returns import ReturnLabelGenerator
from features.pipeline import FeaturePipeline
from models.tabular import XGBoostModel, LightGBMModel
from models.torch.conv_lstm import ConvLSTMModel
from models.torch.tcn import TCNModel
from models.torch.dataset import SequenceDataset
from models.torch.losses import MultiHorizonLoss
from models.training import SequenceTrainer
from portfolio.strategies import LongTopKStrategy
from portfolio.scores import ScoreConverter
from backtest.vectorized import VectorizedBacktester
from backtest.metrics import PerformanceMetrics
from backtest.benchmarks import BenchmarkStrategies
from configs.loader import get_config
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_universe_asset_ids(storage: StorageBackend, symbols: list) -> set:
    """Get asset_ids for given symbols."""
    symbol_list = "', '".join(symbols)
    query = f"SELECT asset_id FROM assets WHERE symbol IN ('{symbol_list}')"
    df = storage.query(query)
    return set(df['asset_id'].values) if len(df) > 0 else set()


def train_tabular_multi_horizon(
    X_train_dict: dict,
    y_train_dict: dict,
    model_type: str = "xgboost",
    **model_kwargs
) -> dict:
    """
    Train one model per horizon for tabular models.
    
    Args:
        X_train_dict: Dict mapping horizon -> DataFrame of features
        y_train_dict: Dict mapping horizon -> array of labels
    
    Returns:
        Dict mapping horizon -> trained model
    """
    models = {}
    
    for horizon, y_train in y_train_dict.items():
        if len(y_train) == 0:
            logger.warning(f"Skipping horizon {horizon}d - no training data")
            continue
            
        X_train = X_train_dict.get(horizon)
        if X_train is None or len(X_train) == 0:
            logger.warning(f"Skipping horizon {horizon}d - no features")
            continue
            
        logger.info(f"Training {model_type} model for horizon {horizon}d ({len(X_train)} samples)...")
        
        if model_type == "xgboost":
            model = XGBoostModel(task_type="regression", **model_kwargs)
        elif model_type == "lightgbm":
            model = LightGBMModel(task_type="regression", **model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.fit(X_train, y_train)
        models[horizon] = model
    
    return models


def predict_tabular_with_uncertainty(
    models: dict,
    X: pd.DataFrame,
    n_ensemble: int = 5
) -> dict:
    """
    Make predictions with uncertainty estimation.
    
    Uses explicit uncertainty if model supports it (e.g., EnsembleXGBoostModel),
    otherwise falls back to heuristic estimates.
    
    Returns:
        Dict mapping horizon -> (mean, std) arrays
    """
    predictions = {}
    
    for horizon, model in models.items():
        # Check if model supports explicit uncertainty
        if hasattr(model, 'predict_with_uncertainty'):
            # Use explicit uncertainty (e.g., EnsembleXGBoostModel)
            pred_mean, pred_std = model.predict_with_uncertainty(X)
            predictions[horizon] = (pred_mean, pred_std)
        else:
            # Fallback: use heuristic uncertainty
            pred_mean = model.predict(X)
            # Heuristic: scale uncertainty by prediction magnitude
            pred_std = np.abs(pred_mean) * 0.1 + 0.05  # 10% of magnitude + baseline
            predictions[horizon] = (pred_mean, pred_std)
    
    return predictions


def fetch_missing_data(
    storage,
    symbols: list,
    start_date,
    end_date,
    vendor: str = "yahoo"
) -> bool:
    """Fetch missing data for the specified date range."""
    from data.vendors.yahoo import YahooClient
    from data.vendors.tiingo import TiingoClient
    from data.normalize import DataNormalizer
    
    logger.info(f"Fetching missing data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    if vendor == "yahoo":
        vendor_client = YahooClient()
    elif vendor == "tiingo":
        vendor_client = TiingoClient()
    else:
        logger.error(f"Unknown vendor: {vendor}")
        return False
    
    try:
        bars_df = vendor_client.fetch_daily_bars(symbols, start_date, end_date)
        if len(bars_df) == 0:
            logger.warning("No bars fetched from vendor")
            return False
        
        logger.info(f"Fetched {len(bars_df)} bars from {vendor}")
        normalizer = DataNormalizer(storage, vendor_client=vendor_client)
        normalized_bars = normalizer.normalize_bars(bars_df, vendor=vendor)
        
        storage.save_parquet(normalized_bars, "bars_daily")
        storage.insert_dataframe("bars_daily", normalized_bars, if_exists="append")
        logger.info("Saved fetched data to storage")
        return True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return False


def check_and_fetch_missing_data(storage, start_date, end_date, symbols=None, vendor="yahoo", auto_fetch=True) -> bool:
    """Check if data exists for the date range and optionally fetch missing data."""
    result = storage.query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM bars_daily")
    
    if len(result) == 0 or result['max_date'].iloc[0] is None:
        if not auto_fetch or symbols is None:
            return False
        return fetch_missing_data(storage, symbols, start_date, end_date, vendor)
    
    db_max_date = result['max_date'].iloc[0]
    if hasattr(db_max_date, 'date'):
        db_max_date = db_max_date.date()
    
    if end_date <= db_max_date:
        logger.info("All required data is available")
        return True
    
    if not auto_fetch:
        logger.error(f"Missing data up to {end_date}. Use --auto-fetch to download.")
        return False
    
    if symbols is None:
        symbols_df = storage.query("SELECT DISTINCT symbol FROM assets ORDER BY symbol")
        symbols = symbols_df['symbol'].tolist()
    
    fetch_start = db_max_date + pd.Timedelta(days=1)
    if isinstance(fetch_start, pd.Timestamp):
        fetch_start = fetch_start.date()
    
    return fetch_missing_data(storage, symbols, fetch_start, end_date, vendor)


def main():
    parser = argparse.ArgumentParser(description="Run multi-horizon, uncertainty-aware backtest")
    parser.add_argument("--start-date", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to backtest (default: use universe_membership)")
    parser.add_argument("--train-start", type=str, help="Training start date (default: same as start-date)")
    parser.add_argument("--train-end", type=str, help="Training end date (default: 80% of backtest period)")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "conv_lstm", "tcn"], help="Model type")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 120], help="Prediction horizons (days)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top assets to hold")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length for sequence models")
    parser.add_argument("--use-uncertainty", action="store_true", help="Use uncertainty estimation (MC dropout for sequence models)")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run benchmark strategies")
    parser.add_argument("--random-portfolio-runs", type=int, default=0, help="Number of random portfolio runs")
    parser.add_argument("--auto-fetch", action="store_true", help="Automatically fetch missing data from vendor")
    parser.add_argument("--vendor", type=str, default="yahoo", choices=["yahoo", "tiingo"], help="Data vendor for auto-fetch")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    # Training period
    if args.train_start:
        train_start = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    else:
        train_start = start_date
    
    if args.train_end:
        train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date()
    else:
        total_days = (end_date - start_date).days
        train_end = start_date + pd.Timedelta(days=int(total_days * 0.8))
    
    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Training period: {train_start} to {train_end}")
    logger.info(f"Horizons: {args.horizons}")
    logger.info(f"Model type: {args.model_type}")
    
    # Initialize storage and API
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    api = AsOfQueryAPI(storage)
    
    # Check for missing data and optionally fetch it
    if args.auto_fetch:
        logger.info("Checking for missing data...")
        data_available = check_and_fetch_missing_data(
            storage=storage,
            start_date=start_date,
            end_date=end_date,
            symbols=args.symbols,
            vendor=args.vendor,
            auto_fetch=True
        )
        if not data_available:
            logger.error("Failed to fetch missing data")
            storage.close()
            return
    
    # Get universe
    if args.symbols:
        universe = get_universe_asset_ids(storage, args.symbols)
        logger.info(f"Using specified symbols: {args.symbols}")
    else:
        try:
            universe = api.get_universe_at_date(end_date, index_name="SP500")
            if len(universe) > 0:
                logger.info(f"Using S&P 500 universe: {len(universe)} assets")
            else:
                bars_df = api.get_bars_asof(end_date)
                universe = set(bars_df['asset_id'].unique())
                logger.info(f"Using all assets in database: {len(universe)} assets")
        except Exception as e:
            logger.warning(f"Could not load universe: {e}")
            bars_df = api.get_bars_asof(end_date)
            universe = set(bars_df['asset_id'].unique())
            logger.info(f"Using all assets in database: {len(universe)} assets")
    
    if len(universe) == 0:
        logger.error("No assets found in universe")
        return
    
    # Get all bars
    max_horizon = max(args.horizons)
    extended_end = pd.Timestamp(end_date) + pd.Timedelta(days=max_horizon * 2)
    all_bars = api.get_bars_asof(extended_end.date(), universe=universe)
    
    if len(all_bars) == 0:
        logger.error("No bars data found")
        return
    
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    bars_df = all_bars[
        (all_bars['date'] >= start_date) & 
        (all_bars['date'] <= end_date)
    ].copy()
    
    # Generate labels for all horizons
    logger.info("Generating multi-horizon return labels...")
    label_generator = ReturnLabelGenerator(storage)
    labels_df = label_generator.generate_labels(
        start_date=train_start,
        end_date=end_date,
        horizons=args.horizons,
        benchmark_symbol="SPY",
        universe=list(universe)
    )
    
    if len(labels_df) == 0:
        logger.error("No labels generated")
        return
    
    logger.info(f"Generated {len(labels_df)} labels across {len(args.horizons)} horizons")
    
    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline(api, config.features)
    calendar = TradingCalendar()
    
    # Get trading days for TRAINING period (not backtest period)
    training_trading_days = calendar.get_trading_days(train_start, train_end)
    
    # Build training features
    logger.info("Building training features...")
    # Store features per horizon separately
    train_features_dict = {h: [] for h in args.horizons}
    train_labels_dict = {h: [] for h in args.horizons}
    
    # Sample every 10th trading day from the training period
    train_dates = [d.date() for d in training_trading_days][::10]
    logger.info(f"Sampling {len(train_dates)} training dates")
    
    successful_dates = 0
    feature_cols = None
    
    for train_date in train_dates:
        try:
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=train_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            if 'asset_id' not in features_df.columns:
                continue
            
            # Get labels for all horizons
            date_labels = labels_df[labels_df['date'] == train_date]
            
            if len(date_labels) == 0:
                continue
            
            # Determine feature columns (once)
            if feature_cols is None:
                feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
            
            # Process each horizon separately
            date_has_data = False
            for horizon in args.horizons:
                horizon_labels = date_labels[date_labels['horizon'] == horizon]
                if len(horizon_labels) == 0:
                    continue
                
                # Merge features with labels for this horizon
                horizon_merged = features_df.merge(
                    horizon_labels[['asset_id', 'target_excess_log_return']],
                    on='asset_id',
                    how='inner'
                )
                
                if len(horizon_merged) == 0:
                    continue
                
                # Extract features
                horizon_X = horizon_merged[feature_cols].copy()
                for col in horizon_X.columns:
                    horizon_X[col] = pd.to_numeric(horizon_X[col], errors='coerce')
                horizon_X = horizon_X.fillna(0)
                horizon_y = horizon_merged['target_excess_log_return'].values
                
                train_features_dict[horizon].append(horizon_X)
                train_labels_dict[horizon].append(horizon_y)
                date_has_data = True
            
            if date_has_data:
                successful_dates += 1
            
        except Exception as e:
            if successful_dates == 0:
                logger.warning(f"Error building features for {train_date}: {e}")
            continue
    
    logger.info(f"Successfully built features for {successful_dates} training dates")
    
    # Combine training data per horizon
    X_train_dict = {}
    for horizon in args.horizons:
        if len(train_features_dict[horizon]) > 0:
            X_train_dict[horizon] = pd.concat(train_features_dict[horizon], ignore_index=True)
            train_labels_dict[horizon] = np.concatenate(train_labels_dict[horizon])
            logger.info(f"Horizon {horizon}d: {len(X_train_dict[horizon])} samples")
        else:
            logger.warning(f"No training data for horizon {horizon}d")
            X_train_dict[horizon] = pd.DataFrame()
            train_labels_dict[horizon] = np.array([])
    
    if all(len(X_train_dict[h]) == 0 for h in args.horizons):
        logger.error("No training features generated for any horizon")
        return
    
    # Use first non-empty horizon for reference
    reference_horizon = next(h for h in args.horizons if len(X_train_dict[h]) > 0)
    logger.info(f"Training data: {len(X_train_dict[reference_horizon])} samples per horizon, {len(feature_cols)} features")
    
    # Train models
    if args.model_type in ["xgboost", "lightgbm"]:
        logger.info(f"Training {args.model_type} models for each horizon...")
        models = train_tabular_multi_horizon(
            X_train_dict,
            train_labels_dict,
            model_type=args.model_type,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        sequence_model = None
    elif args.model_type in ["conv_lstm", "tcn"]:
        logger.info(f"Training {args.model_type} sequence model...")
        
        # Build sequences for training
        logger.info(f"Building sequences of length {args.sequence_length}...")
        sequences_list = []
        labels_dict = {h: [] for h in args.horizons}
        asset_ids_list = []
        dates_list = []
        
        # Sample training dates (every 10th day)
        for train_date in train_dates:
            try:
                # Build sequence ending at train_date
                sequence_array = feature_pipeline.build_features_sequence(
                    as_of_date=train_date,
                    sequence_length=args.sequence_length,
                    universe=universe
                )
                
                if sequence_array.size == 0:
                    continue
                
                # Get labels for this date (all horizons)
                date_labels = labels_df[labels_df['date'] == train_date]
                if len(date_labels) == 0:
                    continue
                
                # Align sequences with labels
                # sequence_array shape: (num_assets, sequence_length, num_features)
                # We need to match asset_ids
                features_df = feature_pipeline.build_features_cross_sectional(
                    as_of_date=train_date,
                    universe=universe,
                    lookback_days=252
                )
                
                if len(features_df) == 0 or 'asset_id' not in features_df.columns:
                    continue
                
                # Get asset_ids in same order as sequence_array
                asset_ids = features_df['asset_id'].values
                
                # Extract labels for each horizon
                horizon_labels_dict = {}
                for horizon in args.horizons:
                    horizon_labels = date_labels[date_labels['horizon'] == horizon]
                    if len(horizon_labels) == 0:
                        continue
                    
                    # Create label array aligned with asset_ids
                    label_map = dict(zip(horizon_labels['asset_id'], horizon_labels['target_excess_log_return']))
                    label_array = np.array([label_map.get(aid, np.nan) for aid in asset_ids])
                    
                    # Only keep samples with valid labels
                    valid_mask = ~np.isnan(label_array)
                    if valid_mask.sum() == 0:
                        continue
                    
                    horizon_labels_dict[horizon] = (label_array, valid_mask)
                
                if len(horizon_labels_dict) == 0:
                    continue
                
                # Use intersection of valid masks across horizons
                valid_mask = np.ones(len(asset_ids), dtype=bool)
                for horizon, (_, mask) in horizon_labels_dict.items():
                    valid_mask = valid_mask & mask
                
                if valid_mask.sum() == 0:
                    continue
                
                # Filter to valid samples
                valid_sequences = sequence_array[valid_mask]
                valid_asset_ids = asset_ids[valid_mask]
                
                # Store sequences and labels
                sequences_list.append(valid_sequences)
                asset_ids_list.append(valid_asset_ids)
                dates_list.append([train_date] * len(valid_asset_ids))
                
                for horizon in args.horizons:
                    if horizon in horizon_labels_dict:
                        label_array, _ = horizon_labels_dict[horizon]
                        labels_dict[horizon].append(label_array[valid_mask])
                    else:
                        # Fill with NaN for missing horizons
                        labels_dict[horizon].append(np.full(valid_mask.sum(), np.nan))
                
            except Exception as e:
                if len(sequences_list) == 0:
                    logger.warning(f"Error building sequence for {train_date}: {e}")
                continue
        
        if len(sequences_list) == 0:
            logger.error("No sequences built for training")
            models = {}
            sequence_model = None
        else:
            # Combine all sequences
            all_sequences = np.concatenate(sequences_list, axis=0)
            all_asset_ids = np.concatenate(asset_ids_list)
            all_dates = np.concatenate(dates_list)
            
            # Combine labels - use intersection of valid samples across all horizons
            # Find samples that have valid labels for ALL horizons
            valid_mask = np.ones(len(all_sequences), dtype=bool)
            for horizon in args.horizons:
                horizon_labels = np.concatenate(labels_dict[horizon])
                horizon_valid = ~np.isnan(horizon_labels)
                valid_mask = valid_mask & horizon_valid
            
            if valid_mask.sum() == 0:
                logger.error("No samples with valid labels for all horizons")
                models = {}
                sequence_model = None
            else:
                # Filter sequences and labels to valid samples
                all_sequences = all_sequences[valid_mask]
                all_asset_ids = all_asset_ids[valid_mask]
                all_dates = all_dates[valid_mask]
                
                combined_labels = {}
                for horizon in args.horizons:
                    horizon_labels = np.concatenate(labels_dict[horizon])
                    combined_labels[horizon] = horizon_labels[valid_mask]
                
                logger.info(f"Valid samples after filtering: {len(all_sequences)}")
                
                # Determine input dimension from sequence shape
                # Determine input dimension from sequence shape
                _, seq_len, input_dim = all_sequences.shape
                logger.info(f"Training sequences: {len(all_sequences)} samples, shape {all_sequences.shape}")
                
                # Create dataset
                dataset = SequenceDataset(
                    sequences=all_sequences,
                    labels=combined_labels,
                    asset_ids=all_asset_ids,
                    dates=all_dates
                )
                
                # Create train/val split (80/20)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Create model
                if args.model_type == "conv_lstm":
                    model = ConvLSTMModel(
                        input_dim=input_dim,
                        sequence_length=seq_len,
                        horizons=list(combined_labels.keys()),
                        use_uncertainty=args.use_uncertainty
                    )
                else:  # tcn
                    model = TCNModel(
                        input_dim=input_dim,
                        sequence_length=seq_len,
                        horizons=list(combined_labels.keys()),
                        use_uncertainty=args.use_uncertainty
                    )
                
                # Create loss function
                loss_fn = MultiHorizonLoss(
                    horizons=list(combined_labels.keys()),
                    loss_type="gaussian_nll" if args.use_uncertainty else "mse"
                )
                
                # Create optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                # Create trainer
                trainer = SequenceTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                # Train
                logger.info("Training sequence model...")
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=50,
                    early_stopping_patience=10,
                    save_best=True
                )
                
                # Load best model
                trainer.load_checkpoint("best_model.pt")
                model.eval()
                
                logger.info("Sequence model training complete")
                models = {}  # Not used for sequence models
                sequence_model = model
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    # Run backtest
    logger.info("Running backtest...")
    strategy = LongTopKStrategy(k=args.top_k, min_score_threshold=-np.inf, min_confidence=0.5 if args.use_uncertainty else 0.0)
    
    all_weights = []
    # Get trading days for the backtest period
    backtest_trading_days = calendar.get_trading_days(start_date, end_date)
    backtest_dates = [d.date() for d in backtest_trading_days]
    
    for backtest_date in backtest_dates:
        try:
            # Build features
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=backtest_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            if 'asset_id' not in features_df.columns:
                continue
            
            # Extract features - use the same feature_cols from training to ensure consistency
            # feature_cols was set during training and should be reused here
            available_cols = [c for c in feature_cols if c in features_df.columns]
            if len(available_cols) != len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                logger.debug(f"Missing {len(missing)} features for {backtest_date}: {list(missing)[:5]}...")
            
            # Only use features that were present during training
            X = features_df[available_cols].copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
            
            # Add missing columns as zeros to match training shape
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            
            # Reorder to match training column order
            X = X[feature_cols]
            
            # Predict for all horizons
            if args.model_type in ["xgboost", "lightgbm"]:
                predictions = predict_tabular_with_uncertainty(models, X)
            elif args.model_type in ["conv_lstm", "tcn"] and sequence_model is not None:
                # Sequence model predictions
                # Build sequence ending at backtest_date
                sequence_array = feature_pipeline.build_features_sequence(
                    as_of_date=backtest_date,
                    sequence_length=args.sequence_length,
                    universe=universe
                )
                
                if sequence_array.size == 0:
                    predictions = {}
                else:
                    # Convert to tensor
                    sequence_tensor = torch.FloatTensor(sequence_array)
                    device = next(sequence_model.parameters()).device
                    sequence_tensor = sequence_tensor.to(device)
                    
                    # Predict with uncertainty if enabled
                    if args.use_uncertainty:
                        predictions = sequence_model.predict_mc(sequence_tensor, n_samples=50)
                    else:
                        # Single forward pass
                        sequence_model.eval()
                        with torch.no_grad():
                            outputs = sequence_model(sequence_tensor)
                            predictions = {}
                            for horizon, pred in outputs.items():
                                if isinstance(pred, tuple):
                                    # Mean-variance output
                                    mean = pred[0].cpu().numpy()
                                    logvar = pred[1].cpu().numpy()
                                    std = np.sqrt(np.exp(logvar))
                                    predictions[horizon] = (mean, std)
                                else:
                                    # Single output - use heuristic uncertainty
                                    mean = pred.cpu().numpy()
                                    std = np.abs(mean) * 0.1 + 0.05
                                    predictions[horizon] = (mean, std)
                    
                    # Map predictions back to asset_ids
                    # sequence_array shape: (num_assets, seq_len, features)
                    # features_df has asset_ids in same order
                    if len(features_df) > 0 and 'asset_id' in features_df.columns:
                        asset_ids = features_df['asset_id'].values
                        # Align predictions with asset_ids
                        aligned_predictions = {}
                        for horizon in predictions:
                            mu, sigma = predictions[horizon]
                            if len(mu) == len(asset_ids):
                                aligned_predictions[horizon] = (mu, sigma)
                            else:
                                # Mismatch - use first N or pad
                                n = min(len(mu), len(asset_ids))
                                aligned_predictions[horizon] = (mu[:n], sigma[:n])
                        predictions = aligned_predictions
            else:
                # Sequence model not trained
                predictions = {}
            
            # Convert to scores using ScoreConverter
            scores = []
            for idx, row in features_df.iterrows():
                asset_id = row['asset_id']
                
                # Build prediction dict for this asset
                pred_dict = {}
                for horizon in args.horizons:
                    if horizon in predictions:
                        mu, sigma = predictions[horizon]
                        pred_dict[horizon] = {'mu': mu[idx], 'sigma': sigma[idx]}
                
                if len(pred_dict) > 0:
                    # Combine multi-horizon scores
                    score = ScoreConverter.combine_multi_horizon_scores(
                        pred_dict,
                        method="weighted_average"
                    )
                    
                    # Compute confidence from uncertainty
                    avg_uncertainty = np.mean([p['sigma'] for p in pred_dict.values()])
                    confidence = 1.0 / (1.0 + avg_uncertainty) if avg_uncertainty > 0 else 1.0
                    
                    scores.append({
                        'asset_id': asset_id,
                        'score': score,
                        'confidence': confidence
                    })
            
            if len(scores) == 0:
                continue
            
            scores_df = pd.DataFrame(scores)
            
            # Compute weights
            weights_df = strategy.compute_weights(scores_df, as_of_date=backtest_date)
            
            if len(weights_df) > 0:
                weights_df['date'] = backtest_date
                all_weights.append(weights_df)
                
        except Exception as e:
            logger.warning(f"Error processing {backtest_date}: {e}")
            continue
    
    if len(all_weights) == 0:
        logger.error("No weights generated")
        return
    
    target_weights_df = pd.concat(all_weights, ignore_index=True)
    logger.info(f"Generated weights for {len(target_weights_df)} date-asset pairs")
    
    # Prepare prices DataFrame
    prices_df = bars_df[['date', 'asset_id', 'adj_close']].copy()
    
    # Get symbol mapping for benchmarks
    try:
        assets_df = storage.query("SELECT asset_id, symbol FROM assets")
        symbol_map = dict(zip(assets_df['asset_id'], assets_df['symbol']))
        prices_df['symbol'] = prices_df['asset_id'].map(symbol_map)
    except Exception as e:
        logger.warning(f"Could not load symbol mapping: {e}")
    
    # Run vectorized backtest
    logger.info("Running vectorized backtest...")
    backtester = VectorizedBacktester(initial_capital=args.initial_capital)
    equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
    
    logger.info(f"Backtest complete: {len(equity_curve)} trading days")
    
    # Compute metrics
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    # Store results
    all_results = {
        'ml_strategy': {
            'name': f"{args.model_type}_multi_horizon_top{args.top_k}",
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    }
    
    # Run benchmarks (reuse code from run_backtest.py)
    if args.run_benchmarks:
        logger.info("Running benchmark strategies...")
        benchmarks = BenchmarkStrategies(backtester)
        
        # SPY buy-and-hold
        try:
            spy_found = False
            if 'symbol' in prices_df.columns:
                spy_found = len(prices_df[prices_df['symbol'] == 'SPY']) > 0
            
            if not spy_found:
                spy_df = storage.query("SELECT asset_id FROM assets WHERE symbol = 'SPY'")
                if len(spy_df) > 0:
                    spy_asset_id = spy_df['asset_id'].iloc[0]
                    spy_bars = api.get_bars_asof(end_date, universe={spy_asset_id})
                    if len(spy_bars) > 0:
                        spy_bars['date'] = pd.to_datetime(spy_bars['date']).dt.date
                        spy_bars = spy_bars[(spy_bars['date'] >= start_date) & (spy_bars['date'] <= end_date)]
                        if len(spy_bars) > 0:
                            spy_bars['symbol'] = 'SPY'
                            prices_df = pd.concat([prices_df, spy_bars[['date', 'asset_id', 'adj_close', 'symbol']]], ignore_index=True)
                            spy_found = True
            
            if spy_found:
                spy_equity = benchmarks.buy_and_hold(prices_df, symbol="SPY")
                spy_metrics = PerformanceMetrics.compute_metrics(spy_equity)
                all_results['spy_buy_and_hold'] = {
                    'name': 'SPY Buy and Hold',
                    'equity_curve': spy_equity,
                    'metrics': spy_metrics
                }
        except Exception as e:
            logger.warning(f"Failed to run SPY buy-and-hold: {e}")
        
        # Equal-weight universe
        try:
            ew_equity = benchmarks.equal_weight_universe(prices_df, rebalance_frequency="monthly")
            ew_metrics = PerformanceMetrics.compute_metrics(ew_equity)
            all_results['equal_weight_universe'] = {
                'name': 'Equal Weight Universe (Monthly)',
                'equity_curve': ew_equity,
                'metrics': ew_metrics
            }
        except Exception as e:
            logger.warning(f"Failed to run equal-weight universe: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("MULTI-HORIZON BACKTEST RESULTS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Horizons: {args.horizons}")
    print(f"Model: {args.model_type}")
    print(f"Top K: {args.top_k}")
    print(f"Uncertainty: {'Enabled' if args.use_uncertainty else 'Disabled'}")
    print("\n" + "-"*80)
    print("ML STRATEGY PERFORMANCE:")
    print("-"*80)
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    if args.run_benchmarks:
        print("\n" + "-"*80)
        print("BENCHMARK COMPARISONS:")
        print("-"*80)
        comparison_data = []
        comparison_data.append({
            'Strategy': all_results['ml_strategy']['name'],
            'CAGR': metrics['cagr'],
            'Sharpe': metrics['sharpe_ratio'],
            'Max DD': metrics['max_drawdown']
        })
        
        for key in ['spy_buy_and_hold', 'equal_weight_universe']:
            if key in all_results:
                bm = all_results[key]
                comparison_data.append({
                    'Strategy': bm['name'],
                    'CAGR': bm['metrics']['cagr'],
                    'Sharpe': bm['metrics']['sharpe_ratio'],
                    'Max DD': bm['metrics']['max_drawdown']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    print("="*80)
    
    # Save results
    output_path = Path("artifacts") / "backtest_results"
    output_path.mkdir(parents=True, exist_ok=True)
    
    equity_curve.to_csv(output_path / f"equity_curve_{args.model_type}_multi_horizon.csv", index=False)
    
    comparison_summary = {
        'backtest_config': {
            'start_date': str(start_date),
            'end_date': str(end_date),
            'horizons': args.horizons,
            'model_type': args.model_type,
            'top_k': args.top_k,
            'use_uncertainty': args.use_uncertainty
        },
        'strategies': {k: {'name': v['name'], 'metrics': v['metrics']} for k, v in all_results.items()}
    }
    
    summary_path = output_path / f"comparison_summary_{args.model_type}_multi_horizon.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    storage.close()


if __name__ == "__main__":
    main()

