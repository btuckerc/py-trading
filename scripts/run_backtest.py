"""Full pipeline backtest script.

This script runs a complete ML trading pipeline:
1. Loads historical bars data
2. Generates multi-horizon return labels
3. Builds features using FeaturePipeline
4. Trains models (XGBoost/LightGBM)
5. Generates predictions
6. Converts to portfolio weights using LongTopKStrategy
7. Runs vectorized backtest
8. Computes performance metrics
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
from data.clock import SimulationClock
from data.universe import TradingCalendar
from data.maintenance import ensure_data_coverage
from labels.returns import ReturnLabelGenerator
from features.pipeline import FeaturePipeline
from models.tabular import XGBoostModel, LightGBMModel
from portfolio.strategies import LongTopKStrategy
from backtest.vectorized import VectorizedBacktester
from backtest.metrics import PerformanceMetrics
from backtest.benchmarks import BenchmarkStrategies
from configs.loader import get_config
from loguru import logger
import json


from portfolio.costs import TransactionCostModel
from labels.regimes import RegimeLabelGenerator
from models.training import WalkForwardRetrainer
import hashlib


def get_config_hash(config_dict: dict) -> str:
    """Generate a hash of config for versioning/tracking."""
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_universe_asset_ids(storage: StorageBackend, symbols: list) -> set:
    """Get asset_ids for given symbols."""
    symbol_list = "', '".join(symbols)
    query = f"SELECT asset_id FROM assets WHERE symbol IN ('{symbol_list}')"
    df = storage.query(query)
    return set(df['asset_id'].values) if len(df) > 0 else set()


def run_cost_sensitivity_analysis(
    prices_df: pd.DataFrame,
    target_weights_df: pd.DataFrame,
    cost_levels: list,
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Run backtest at different cost levels to analyze cost sensitivity.
    
    Args:
        prices_df: Price data for backtest
        target_weights_df: Target weights from strategy
        cost_levels: List of cost levels in basis points per side
        initial_capital: Starting capital
    
    Returns:
        DataFrame with metrics at each cost level
    """
    results = []
    
    for cost_bps in cost_levels:
        # Create cost model with this level
        cost_model = TransactionCostModel(
            commission_bps=cost_bps / 2,  # Half for commission
            slippage_bps=cost_bps / 2     # Half for slippage
        )
        
        # Run backtest
        backtester = VectorizedBacktester(
            initial_capital=initial_capital,
            cost_model=cost_model
        )
        
        equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
        metrics = PerformanceMetrics.compute_metrics(equity_curve)
        
        results.append({
            'cost_bps': cost_bps,
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'calmar_ratio': metrics['calmar_ratio']
        })
    
    return pd.DataFrame(results)


def run_walk_forward_evaluation(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: set,
    start_date: date,
    end_date: date,
    train_years: int = 3,
    test_years: int = 1,
    model_type: str = "xgboost",
    horizon: int = 20,
    top_k: int = 3,
    initial_capital: float = 100000.0,
    config = None
) -> dict:
    """
    Run walk-forward evaluation: train on rolling window, test on next period.
    
    Now uses the retraining policy from config for time-decay sample weighting.
    
    Returns:
        Dict with aggregated metrics and per-window results
    """
    calendar = TradingCalendar()
    
    # Get retraining config for time-decay weights
    retraining_config = config.retraining if config and hasattr(config, 'retraining') else None
    
    # Calculate windows
    windows = []
    current_train_start = start_date
    
    while True:
        train_end = (pd.Timestamp(current_train_start) + pd.DateOffset(years=train_years)).date()
        test_start = train_end
        test_end = (pd.Timestamp(test_start) + pd.DateOffset(years=test_years)).date()
        
        if test_end > end_date:
            break
        
        windows.append({
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_train_start = (pd.Timestamp(current_train_start) + pd.DateOffset(years=1)).date()
    
    if len(windows) == 0:
        logger.warning("No walk-forward windows generated (period too short)")
        return {'windows': [], 'aggregate_metrics': {}}
    
    logger.info(f"Running walk-forward with {len(windows)} windows")
    if retraining_config and retraining_config.time_decay.enabled:
        logger.info(f"Time-decay weighting enabled (lambda={retraining_config.time_decay.lambda_})")
    
    window_results = []
    all_equity_curves = []
    
    for i, window in enumerate(windows):
        logger.info(f"Window {i+1}/{len(windows)}: train {window['train_start']} to {window['train_end']}, test {window['test_start']} to {window['test_end']}")
        
        try:
            # Generate labels for training period
            label_generator = ReturnLabelGenerator(storage)
            labels_df = label_generator.generate_labels(
                start_date=window['train_start'],
                end_date=window['train_end'],
                horizons=[horizon],
                benchmark_symbol="SPY",
                universe=list(universe)
            )
            
            if len(labels_df) == 0:
                logger.warning(f"No labels for window {i+1}")
                continue
            
            # Build features
            feature_pipeline = FeaturePipeline(api, config.features if config else None)
            trading_days = [d.date() for d in calendar.get_trading_days(window['train_start'], window['train_end'])][::10]
            
            X_train_list = []
            y_train_list = []
            date_list = []  # Track dates for time-decay weighting
            
            for train_date in trading_days:
                try:
                    features_df = feature_pipeline.build_features_cross_sectional(
                        as_of_date=train_date,
                        universe=universe,
                        lookback_days=252
                    )
                    
                    if len(features_df) == 0:
                        continue
                    
                    date_labels = labels_df[labels_df['date'] == train_date]
                    if len(date_labels) == 0:
                        continue
                    
                    merged = features_df.merge(
                        date_labels[['asset_id', 'target_excess_log_return']],
                        on='asset_id',
                        how='inner'
                    )
                    
                    if len(merged) == 0:
                        continue
                    
                    feature_cols = [c for c in merged.columns if c not in ['asset_id', 'date', 'target_excess_log_return']]
                    X = merged[feature_cols].copy()
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    X = X.fillna(0)
                    y = merged['target_excess_log_return'].values
                    
                    X_train_list.append(X)
                    y_train_list.append(y)
                    # Track dates for each sample
                    date_list.extend([train_date] * len(merged))
                except Exception:
                    continue
            
            if len(X_train_list) == 0:
                logger.warning(f"No training data for window {i+1}")
                continue
            
            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = np.concatenate(y_train_list)
            
            # Compute time-decay sample weights
            sample_weights = None
            if retraining_config and retraining_config.time_decay.enabled:
                sample_weights = retraining_config.compute_sample_weights(date_list, window['train_end'])
                sample_weights = np.array(sample_weights)
                logger.debug(f"Sample weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
            
            # Train model with sample weights
            if model_type == "xgboost":
                model = XGBoostModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
            else:
                model = LightGBMModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
            
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Run test period
            strategy = LongTopKStrategy(k=top_k, min_score_threshold=-np.inf)
            
            test_trading_days = [d.date() for d in calendar.get_trading_days(window['test_start'], window['test_end'])]
            all_weights = []
            
            for test_date in test_trading_days:
                try:
                    features_df = feature_pipeline.build_features_cross_sectional(
                        as_of_date=test_date,
                        universe=universe,
                        lookback_days=252
                    )
                    
                    if len(features_df) == 0:
                        continue
                    
                    feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
                    X = features_df[feature_cols].copy()
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    X = X.fillna(0)
                    
                    predictions = model.predict(X)
                    
                    scores_df = pd.DataFrame({
                        'asset_id': features_df['asset_id'].values,
                        'score': predictions,
                        'confidence': np.ones(len(predictions))
                    })
                    
                    weights_df = strategy.compute_weights(scores_df, as_of_date=test_date)
                    if len(weights_df) > 0:
                        weights_df['date'] = test_date
                        all_weights.append(weights_df)
                except Exception:
                    continue
            
            if len(all_weights) == 0:
                logger.warning(f"No weights generated for window {i+1}")
                continue
            
            target_weights_df = pd.concat(all_weights, ignore_index=True)
            
            # Get prices for test period
            all_bars = api.get_bars_asof(window['test_end'], universe=universe)
            all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
            prices_df = all_bars[
                (all_bars['date'] >= window['test_start']) &
                (all_bars['date'] <= window['test_end'])
            ].copy()
            
            # Run backtest
            backtester = VectorizedBacktester(initial_capital=initial_capital)
            equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
            
            if len(equity_curve) > 0:
                metrics = PerformanceMetrics.compute_metrics(equity_curve)
                window_results.append({
                    'window': i + 1,
                    'train_start': str(window['train_start']),
                    'train_end': str(window['train_end']),
                    'test_start': str(window['test_start']),
                    'test_end': str(window['test_end']),
                    'metrics': metrics
                })
                all_equity_curves.append(equity_curve)
        
        except Exception as e:
            logger.warning(f"Error in window {i+1}: {e}")
            continue
    
    # Aggregate metrics
    if len(window_results) > 0:
        aggregate_metrics = {
            'avg_sharpe': np.mean([w['metrics']['sharpe_ratio'] for w in window_results]),
            'std_sharpe': np.std([w['metrics']['sharpe_ratio'] for w in window_results]),
            'avg_cagr': np.mean([w['metrics']['cagr'] for w in window_results]),
            'std_cagr': np.std([w['metrics']['cagr'] for w in window_results]),
            'avg_max_dd': np.mean([w['metrics']['max_drawdown'] for w in window_results]),
            'num_windows': len(window_results),
            'positive_sharpe_windows': sum(1 for w in window_results if w['metrics']['sharpe_ratio'] > 0)
        }
    else:
        aggregate_metrics = {}
    
    return {
        'windows': window_results,
        'aggregate_metrics': aggregate_metrics
    }


def run_policy_driven_backtest(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: set,
    start_date: date,
    end_date: date,
    model_type: str = "xgboost",
    horizon: int = 20,
    top_k: int = 3,
    initial_capital: float = 100000.0,
    config = None
) -> dict:
    """
    Run a backtest using the retraining policy from config.
    
    This simulates exactly what would happen in live trading:
    - Model is retrained according to cadence_days
    - Time-decay weighting is applied to training samples
    - Adaptive retraining triggers are checked
    
    Returns:
        Dict with equity curve, metrics, and retraining history
    """
    calendar = TradingCalendar()
    
    # Get retraining config
    if not config or not hasattr(config, 'retraining'):
        logger.warning("No retraining config found, using defaults")
        from configs.loader import RetrainingConfig
        retraining_config = RetrainingConfig()
    else:
        retraining_config = config.retraining
    
    logger.info(f"Running policy-driven backtest with cadence={retraining_config.cadence_days} days")
    logger.info(f"Window: {retraining_config.window_type}, {retraining_config.window_years} years")
    logger.info(f"Time-decay: enabled={retraining_config.time_decay.enabled}, lambda={retraining_config.time_decay.lambda_}")
    
    # Initialize components
    feature_pipeline = FeaturePipeline(api, config.features if config else None)
    label_generator = ReturnLabelGenerator(storage)
    
    # Model class and params
    if model_type == "xgboost":
        model_class = XGBoostModel
        model_params = {"task_type": "regression", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    else:
        model_class = LightGBMModel
        model_params = {"task_type": "regression", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    
    # Initialize walk-forward retrainer
    retrainer = WalkForwardRetrainer(
        retraining_config=retraining_config,
        model_class=model_class,
        model_params=model_params,
        feature_pipeline=feature_pipeline,
        label_generator=label_generator,
        storage=storage,
        api=api,
        universe=universe,
        horizon=horizon
    )
    
    # Get trading days
    trading_days = [d.date() for d in calendar.get_trading_days(start_date, end_date)]
    
    # Strategy
    strategy = LongTopKStrategy(k=top_k, min_score_threshold=-np.inf)
    
    # Run backtest day by day
    all_weights = []
    retraining_history = []
    
    for i, trading_date in enumerate(trading_days):
        try:
            # Get model for this date (may trigger retraining)
            model, was_retrained = retrainer.get_model_for_date(trading_date)
            
            if was_retrained:
                retraining_history.append({
                    'date': str(trading_date),
                    'day_index': i,
                    'model_version': retrainer.model_version
                })
            
            # Build features
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=trading_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            # Predict
            feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
            X = features_df[feature_cols].copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
            
            predictions = model.predict(X)
            
            # Compute weights
            scores_df = pd.DataFrame({
                'asset_id': features_df['asset_id'].values,
                'score': predictions,
                'confidence': np.ones(len(predictions))
            })
            
            weights_df = strategy.compute_weights(scores_df, as_of_date=trading_date)
            if len(weights_df) > 0:
                weights_df['date'] = trading_date
                all_weights.append(weights_df)
            
            # Update rolling metrics for adaptive retraining (if we have actuals)
            # This would require looking at next-day returns, which we'll skip for now
            # to avoid lookahead bias in the backtest
            
        except Exception as e:
            logger.warning(f"Error on {trading_date}: {e}")
            continue
    
    if len(all_weights) == 0:
        return {'error': 'No weights generated', 'retraining_history': retraining_history}
    
    target_weights_df = pd.concat(all_weights, ignore_index=True)
    
    # Get prices
    all_bars = api.get_bars_asof(end_date, universe=universe)
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    prices_df = all_bars[
        (all_bars['date'] >= start_date) &
        (all_bars['date'] <= end_date)
    ].copy()
    
    # Run backtest
    backtester = VectorizedBacktester(initial_capital=initial_capital)
    equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
    
    if len(equity_curve) == 0:
        return {'error': 'Empty equity curve', 'retraining_history': retraining_history}
    
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    return {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'retraining_history': retraining_history,
        'num_retrains': len(retraining_history),
        'config': {
            'cadence_days': retraining_config.cadence_days,
            'window_type': retraining_config.window_type,
            'window_years': retraining_config.window_years,
            'time_decay_enabled': retraining_config.time_decay.enabled,
            'time_decay_lambda': retraining_config.time_decay.lambda_,
        }
    }


def run_regime_aware_backtest(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: set,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    model_type: str = "xgboost",
    horizon: int = 20,
    top_k: int = 3,
    initial_capital: float = 100000.0,
    config = None
) -> dict:
    """
    Train separate models per market regime and blend predictions.
    
    Returns:
        Dict with per-regime metrics and blended results
    """
    logger.info("Running regime-aware backtest...")
    
    # Fit regimes
    regime_generator = RegimeLabelGenerator(storage, n_regimes=4)
    regimes_df = regime_generator.fit_regimes(train_start, train_end, method="kmeans")
    
    if len(regimes_df) == 0:
        logger.warning("Could not generate regime labels")
        return {'error': 'No regimes generated'}
    
    logger.info(f"Identified {regimes_df['regime_id'].nunique()} market regimes")
    
    # Generate labels
    label_generator = ReturnLabelGenerator(storage)
    labels_df = label_generator.generate_labels(
        start_date=train_start,
        end_date=train_end,
        horizons=[horizon],
        benchmark_symbol="SPY",
        universe=list(universe)
    )
    
    if len(labels_df) == 0:
        return {'error': 'No labels generated'}
    
    # Note: We don't merge regime_id into labels_df because features_df already has regime_id
    # from the FeaturePipeline's RegimeFeatureBuilder. Using the features_df regime_id
    # ensures consistency and avoids column conflicts during the merge.
    regimes_df['date'] = pd.to_datetime(regimes_df['date']).dt.date
    
    # Build features
    feature_pipeline = FeaturePipeline(api, config.features if config else None)
    calendar = TradingCalendar()
    
    trading_days = [d.date() for d in calendar.get_trading_days(train_start, train_end)][::10]
    
    # Collect training data with regime labels
    X_by_regime = {}
    y_by_regime = {}
    
    for train_date in trading_days:
        try:
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=train_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            date_labels = labels_df[labels_df['date'] == train_date]
            if len(date_labels) == 0:
                continue
            
            # Use regime_id from features_df (already computed by FeaturePipeline)
            # Don't merge regime_id from labels to avoid column conflict
            merged = features_df.merge(
                date_labels[['asset_id', 'target_excess_log_return']],
                on='asset_id',
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Use the regime_id already present in features_df
            # Exclude index column which is often non-numeric
            feature_cols = [c for c in merged.columns if c not in ['asset_id', 'date', 'index', 'target_excess_log_return', 'regime_id']]
            
            for regime_id in merged['regime_id'].unique():
                regime_data = merged[merged['regime_id'] == regime_id]
                X = regime_data[feature_cols].copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(0)
                y = regime_data['target_excess_log_return'].values
                
                if regime_id not in X_by_regime:
                    X_by_regime[regime_id] = []
                    y_by_regime[regime_id] = []
                
                X_by_regime[regime_id].append(X)
                y_by_regime[regime_id].append(y)
        except Exception:
            continue
    
    # Train one model per regime
    models = {}
    for regime_id in X_by_regime:
        if len(X_by_regime[regime_id]) == 0:
            continue
        
        X_train = pd.concat(X_by_regime[regime_id], ignore_index=True)
        y_train = np.concatenate(y_by_regime[regime_id])
        
        if len(X_train) < 100:  # Minimum samples
            logger.warning(f"Regime {regime_id} has only {len(X_train)} samples, skipping")
            continue
        
        logger.info(f"Training model for regime {regime_id} with {len(X_train)} samples")
        
        if model_type == "xgboost":
            model = XGBoostModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
        else:
            model = LightGBMModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
        
        model.fit(X_train, y_train)
        models[regime_id] = model
    
    if len(models) == 0:
        return {'error': 'No models trained'}
    
    logger.info(f"Trained {len(models)} regime-specific models")
    
    # Test period: use the trained regime model to predict regimes
    # (don't fit a new model, use predict_regime which uses the already-fitted model)
    strategy = LongTopKStrategy(k=top_k, min_score_threshold=-np.inf)
    test_trading_days = [d.date() for d in calendar.get_trading_days(test_start, test_end)]
    
    all_weights = []
    
    for test_date in test_trading_days:
        try:
            # Get current regime using the fitted model
            current_regime, regime_descriptor = regime_generator.predict_regime(test_date)
            
            # Use regime-specific model if available, else use fallback
            if current_regime in models:
                model = models[current_regime]
            else:
                # Fallback to first available model
                model = list(models.values())[0]
            
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=test_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
            # Exclude non-feature columns (must match training exclusions)
            exclude_cols = ['asset_id', 'date', 'index', 'regime_id']
            feature_cols = [c for c in features_df.columns if c not in exclude_cols]
            X = features_df[feature_cols].copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
            
            predictions = model.predict(X)
            
            scores_df = pd.DataFrame({
                'asset_id': features_df['asset_id'].values,
                'score': predictions,
                'confidence': np.ones(len(predictions))
            })
            
            weights_df = strategy.compute_weights(scores_df, as_of_date=test_date)
            if len(weights_df) > 0:
                weights_df['date'] = test_date
                all_weights.append(weights_df)
        except Exception:
            continue
    
    if len(all_weights) == 0:
        return {'error': 'No weights generated'}
    
    target_weights_df = pd.concat(all_weights, ignore_index=True)
    
    # Get prices and run backtest
    all_bars = api.get_bars_asof(test_end, universe=universe)
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    prices_df = all_bars[
        (all_bars['date'] >= test_start) &
        (all_bars['date'] <= test_end)
    ].copy()
    
    backtester = VectorizedBacktester(initial_capital=initial_capital)
    equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
    
    if len(equity_curve) == 0:
        return {'error': 'Empty equity curve'}
    
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    return {
        'metrics': metrics,
        'num_regimes': len(models),
        'regime_model_sizes': {r: len(X_by_regime.get(r, [[]])[0]) for r in models.keys()}
    }


def run_ablation_tests(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: set,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    model_type: str = "xgboost",
    horizon: int = 20,
    top_k: int = 3,
    initial_capital: float = 100000.0,
    config = None
) -> dict:
    """
    Run ablation tests by disabling feature groups one at a time.
    
    Returns:
        Dict with metrics for baseline and each ablated variant
    """
    logger.info("Running ablation tests...")
    
    feature_groups = ['technical', 'cross_sectional', 'fundamentals', 'calendar', 'sentiment']
    results = {}
    
    # Baseline: all features
    logger.info("Testing baseline (all features)...")
    baseline_metrics = run_single_ablation(
        storage, api, universe, train_start, train_end, test_start, test_end,
        model_type, horizon, top_k, initial_capital, config, disabled_features=[]
    )
    results['baseline'] = baseline_metrics
    
    # Ablate each feature group
    for feature_group in feature_groups:
        logger.info(f"Testing without {feature_group}...")
        ablated_metrics = run_single_ablation(
            storage, api, universe, train_start, train_end, test_start, test_end,
            model_type, horizon, top_k, initial_capital, config, disabled_features=[feature_group]
        )
        results[f'no_{feature_group}'] = ablated_metrics
    
    return results


def run_single_ablation(
    storage, api, universe, train_start, train_end, test_start, test_end,
    model_type, horizon, top_k, initial_capital, config, disabled_features
) -> dict:
    """Run a single ablation variant."""
    try:
        # Create modified config
        modified_config = dict(config.features) if config and hasattr(config, 'features') else {}
        for feature in disabled_features:
            if feature in modified_config:
                modified_config[feature] = {'enabled': False}
        
        # Build features
        feature_pipeline = FeaturePipeline(api, modified_config)
        calendar = TradingCalendar()
        
        # Generate labels
        label_generator = ReturnLabelGenerator(storage)
        labels_df = label_generator.generate_labels(
            start_date=train_start,
            end_date=train_end,
            horizons=[horizon],
            benchmark_symbol="SPY",
            universe=list(universe)
        )
        
        if len(labels_df) == 0:
            return {'error': 'No labels'}
        
        # Build training data
        trading_days = [d.date() for d in calendar.get_trading_days(train_start, train_end)][::10]
        
        X_train_list = []
        y_train_list = []
        
        for train_date in trading_days:
            try:
                features_df = feature_pipeline.build_features_cross_sectional(
                    as_of_date=train_date,
                    universe=universe,
                    lookback_days=252
                )
                
                if len(features_df) == 0:
                    continue
                
                date_labels = labels_df[labels_df['date'] == train_date]
                if len(date_labels) == 0:
                    continue
                
                merged = features_df.merge(
                    date_labels[['asset_id', 'target_excess_log_return']],
                    on='asset_id',
                    how='inner'
                )
                
                if len(merged) == 0:
                    continue
                
                feature_cols = [c for c in merged.columns if c not in ['asset_id', 'date', 'target_excess_log_return']]
                X = merged[feature_cols].copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(0)
                y = merged['target_excess_log_return'].values
                
                X_train_list.append(X)
                y_train_list.append(y)
            except Exception:
                continue
        
        if len(X_train_list) == 0:
            return {'error': 'No training data', 'disabled': disabled_features}
        
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = np.concatenate(y_train_list)
        
        # Train model
        if model_type == "xgboost":
            model = XGBoostModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
        else:
            model = LightGBMModel(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
        
        model.fit(X_train, y_train)
        
        # Test
        strategy = LongTopKStrategy(k=top_k, min_score_threshold=-np.inf)
        test_trading_days = [d.date() for d in calendar.get_trading_days(test_start, test_end)]
        
        all_weights = []
        
        for test_date in test_trading_days:
            try:
                features_df = feature_pipeline.build_features_cross_sectional(
                    as_of_date=test_date,
                    universe=universe,
                    lookback_days=252
                )
                
                if len(features_df) == 0:
                    continue
                
                feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
                X = features_df[feature_cols].copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(0)
                
                predictions = model.predict(X)
                
                scores_df = pd.DataFrame({
                    'asset_id': features_df['asset_id'].values,
                    'score': predictions,
                    'confidence': np.ones(len(predictions))
                })
                
                weights_df = strategy.compute_weights(scores_df, as_of_date=test_date)
                if len(weights_df) > 0:
                    weights_df['date'] = test_date
                    all_weights.append(weights_df)
            except Exception:
                continue
        
        if len(all_weights) == 0:
            return {'error': 'No weights', 'disabled': disabled_features}
        
        target_weights_df = pd.concat(all_weights, ignore_index=True)
        
        # Backtest
        all_bars = api.get_bars_asof(test_end, universe=universe)
        all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
        prices_df = all_bars[
            (all_bars['date'] >= test_start) &
            (all_bars['date'] <= test_end)
        ].copy()
        
        backtester = VectorizedBacktester(initial_capital=initial_capital)
        equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
        
        if len(equity_curve) == 0:
            return {'error': 'Empty equity curve', 'disabled': disabled_features}
        
        metrics = PerformanceMetrics.compute_metrics(equity_curve)
        return {
            'metrics': metrics,
            'disabled': disabled_features,
            'num_features': len(X_train.columns)
        }
    except Exception as e:
        return {'error': str(e), 'disabled': disabled_features}


def fetch_missing_data(
    storage: StorageBackend,
    symbols: list,
    start_date: date,
    end_date: date,
    vendor: str = "yahoo"
) -> bool:
    """
    Fetch missing data for the specified date range.
    
    Args:
        storage: Storage backend
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data
        vendor: Data vendor to use
    
    Returns:
        True if data was fetched successfully, False otherwise
    """
    from data.vendors.yahoo import YahooClient
    from data.vendors.tiingo import TiingoClient
    from data.normalize import DataNormalizer
    
    logger.info(f"Fetching missing data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Initialize vendor client
    if vendor == "yahoo":
        vendor_client = YahooClient()
    elif vendor == "tiingo":
        vendor_client = TiingoClient()
    else:
        logger.error(f"Unknown vendor: {vendor}")
        return False
    
    try:
        # Fetch bars
        bars_df = vendor_client.fetch_daily_bars(symbols, start_date, end_date)
        
        if len(bars_df) == 0:
            logger.warning("No bars fetched from vendor")
            return False
        
        logger.info(f"Fetched {len(bars_df)} bars from {vendor}")
        
        # Normalize
        normalizer = DataNormalizer(storage, vendor_client=vendor_client)
        normalized_bars = normalizer.normalize_bars(bars_df, vendor=vendor)
        
        logger.info(f"Normalized to {len(normalized_bars)} records")
        
        # Save to Parquet and DuckDB
        storage.save_parquet(normalized_bars, "bars_daily")
        storage.insert_dataframe("bars_daily", normalized_bars, if_exists="append")
        
        logger.info("Saved fetched data to storage")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return False


def check_and_fetch_missing_data(
    storage: StorageBackend,
    start_date: date,
    end_date: date,
    symbols: list = None,
    vendor: str = "yahoo",
    auto_fetch: bool = True
) -> bool:
    """
    Check if data exists for the date range and optionally fetch missing data.
    
    Args:
        storage: Storage backend
        start_date: Start date for backtest
        end_date: End date for backtest
        symbols: Optional list of symbols (if None, uses all in database)
        vendor: Data vendor to use for fetching
        auto_fetch: If True, automatically fetch missing data
    
    Returns:
        True if data is available (or was fetched), False otherwise
    """
    # Get current data range
    result = storage.query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM bars_daily")
    
    if len(result) == 0 or result['max_date'].iloc[0] is None:
        logger.warning("No data in database")
        if not auto_fetch:
            return False
        # Fetch all data
        if symbols is None:
            logger.error("No symbols specified and no data in database")
            return False
        return fetch_missing_data(storage, symbols, start_date, end_date, vendor)
    
    db_min_date = result['min_date'].iloc[0]
    db_max_date = result['max_date'].iloc[0]
    
    # Convert to date objects if needed
    if hasattr(db_min_date, 'date'):
        db_min_date = db_min_date.date()
    if hasattr(db_max_date, 'date'):
        db_max_date = db_max_date.date()
    
    logger.info(f"Database has data from {db_min_date} to {db_max_date}")
    logger.info(f"Requested range: {start_date} to {end_date}")
    
    # Check if we need to fetch data
    needs_fetch = False
    fetch_start = None
    fetch_end = None
    
    if end_date > db_max_date:
        needs_fetch = True
        fetch_start = db_max_date + pd.Timedelta(days=1)
        fetch_end = end_date
        logger.warning(f"Data ends at {db_max_date}, need data up to {end_date}")
    
    if start_date < db_min_date:
        needs_fetch = True
        if fetch_start is None:
            fetch_start = start_date
        else:
            fetch_start = start_date
        if fetch_end is None:
            fetch_end = db_min_date - pd.Timedelta(days=1)
        logger.warning(f"Data starts at {db_min_date}, need data from {start_date}")
    
    if not needs_fetch:
        logger.info("All required data is available")
        return True
    
    if not auto_fetch:
        logger.error(f"Missing data for range {fetch_start} to {fetch_end}. Use --auto-fetch to download.")
        return False
    
    # Get symbols to fetch
    if symbols is None:
        symbols_df = storage.query("SELECT DISTINCT symbol FROM assets ORDER BY symbol")
        symbols = symbols_df['symbol'].tolist()
    
    if len(symbols) == 0:
        logger.error("No symbols to fetch")
        return False
    
    # Convert fetch dates to date objects
    if hasattr(fetch_start, 'date'):
        fetch_start = fetch_start.date()
    elif isinstance(fetch_start, pd.Timestamp):
        fetch_start = fetch_start.date()
    
    if hasattr(fetch_end, 'date'):
        fetch_end = fetch_end.date()
    elif isinstance(fetch_end, pd.Timestamp):
        fetch_end = fetch_end.date()
    
    return fetch_missing_data(storage, symbols, fetch_start, fetch_end, vendor)


def main():
    parser = argparse.ArgumentParser(description="Run full ML backtest pipeline")
    parser.add_argument("--start-date", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to backtest (default: all in database)")
    parser.add_argument("--train-start", type=str, help="Training start date (default: same as start-date)")
    parser.add_argument("--train-end", type=str, help="Training end date (default: 80%% of backtest period)")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm"], help="Model type")
    parser.add_argument("--horizon", type=int, default=20, help="Primary prediction horizon (days)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top assets to hold")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run benchmark strategies for comparison")
    parser.add_argument("--benchmark", type=str, help="Single benchmark to use (e.g., 'sp500', 'dow', 'nasdaq', or ticker like 'SPY')")
    parser.add_argument("--benchmarks", type=str, help="Comma-separated list of benchmarks (e.g., 'sp500,dow,nasdaq' or 'SPY,QQQ')")
    parser.add_argument("--random-portfolio-runs", type=int, default=0, help="Number of random portfolio runs for distribution (0 to skip)")
    parser.add_argument("--cost-sensitivity", action="store_true", help="Run cost sensitivity analysis at different cost levels")
    parser.add_argument("--cost-levels", type=float, nargs="+", default=[0, 5, 10, 20], help="Cost levels in bps per side (default: 0 5 10 20)")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward evaluation")
    parser.add_argument("--walk-forward-train-years", type=int, default=3, help="Training window size in years")
    parser.add_argument("--walk-forward-test-years", type=int, default=1, help="Test window size in years")
    parser.add_argument("--regime-aware", action="store_true", help="Train separate models per market regime")
    parser.add_argument("--ablation-test", action="store_true", help="Run ablation tests by disabling feature groups")
    parser.add_argument("--auto-fetch", action="store_true", help="Automatically fetch missing data from vendor")
    parser.add_argument("--vendor", type=str, default="yahoo", choices=["yahoo", "tiingo"], help="Data vendor for auto-fetch")
    parser.add_argument("--policy-driven", action="store_true", 
                       help="Run policy-driven backtest using retraining config (cadence, time-decay)")
    parser.add_argument("--retrain-cadence", type=int, default=None,
                       help="Override retraining cadence (days). Default: use config value")
    parser.add_argument("--time-decay-lambda", type=float, default=None,
                       help="Override time-decay lambda. Higher = more emphasis on recent data")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    # Training period (use 80% of backtest period if not specified)
    if args.train_start:
        train_start = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    else:
        train_start = start_date
    
    if args.train_end:
        train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date()
    else:
        # Use 80% of period for training
        total_days = (end_date - start_date).days
        train_end = start_date + pd.Timedelta(days=int(total_days * 0.8))
    
    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Training period: {train_start} to {train_end}")
    
    # Initialize storage and API
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    api = AsOfQueryAPI(storage)
    
    # Check for missing data and optionally fetch it
    # Use config-driven auto_fetch or explicit --auto-fetch flag
    auto_fetch_enabled = args.auto_fetch or config.data.auto_fetch_on_backtest
    if auto_fetch_enabled:
        logger.info("Checking for missing data using maintenance module...")
        coverage_result = ensure_data_coverage(
            storage=storage,
            config=config.__dict__,
            mode="custom",
            target_start=start_date,
            target_end=end_date,
            symbols=args.symbols,
            vendor=args.vendor,
            auto_fetch=True
        )
        if coverage_result['status'] == 'error':
            logger.error(f"Failed to ensure data coverage: {coverage_result.get('message')}")
            storage.close()
            return
        elif coverage_result['gaps_identified']:
            logger.info(f"Data coverage result: {coverage_result.get('message')}")
    
    # Get universe
    if args.symbols:
        universe = get_universe_asset_ids(storage, args.symbols)
        logger.info(f"Using specified symbols: {args.symbols}")
    else:
        # Try to use universe_membership table (survivorship-bias-free)
        try:
            universe = api.get_universe_at_date(end_date, index_name="SP500")
            if len(universe) > 0:
                logger.info(f"Using S&P 500 universe from universe_membership table: {len(universe)} assets")
            else:
                # Fallback: get all asset_ids from bars_daily
                bars_df = api.get_bars_asof(end_date)
                universe = set(bars_df['asset_id'].unique())
                logger.info(f"Universe_membership table empty, using all assets in database: {len(universe)} assets")
        except Exception as e:
            # Fallback: get all asset_ids from bars_daily
            logger.warning(f"Could not load universe from universe_membership table: {e}")
            bars_df = api.get_bars_asof(end_date)
            universe = set(bars_df['asset_id'].unique())
            logger.info(f"Using all assets in database: {len(universe)} assets")
    
    if len(universe) == 0:
        logger.error("No assets found in universe")
        return
    
    logger.info(f"Universe size: {len(universe)} assets")
    
    # Get all bars for the period (need extended period for labels)
    max_horizon = args.horizon
    extended_end = pd.Timestamp(end_date) + pd.Timedelta(days=max_horizon * 2)
    all_bars = api.get_bars_asof(extended_end.date(), universe=universe)
    
    if len(all_bars) == 0:
        logger.error("No bars data found")
        return
    
    logger.info(f"Loaded {len(all_bars)} bars")
    
    # Filter bars to backtest period
    # Ensure date column is date type
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    bars_df = all_bars[
        (all_bars['date'] >= start_date) & 
        (all_bars['date'] <= end_date)
    ].copy()
    
    # Generate labels
    logger.info("Generating return labels...")
    label_generator = ReturnLabelGenerator(storage)
    labels_df = label_generator.generate_labels(
        start_date=train_start,
        end_date=end_date,
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
    # Get trading days for the full range (including training period)
    full_range_start = min(train_start, start_date)
    full_range_end = max(train_end, end_date)
    all_trading_days = calendar.get_trading_days(full_range_start, full_range_end)
    trading_days = calendar.get_trading_days(start_date, end_date)
    
    # Build features and train model
    logger.info("Building features for training...")
    train_features_list = []
    train_labels_list = []
    
    # Sample training dates from the training period (every 10 days to speed up)
    train_trading_days = [d.date() for d in all_trading_days if train_start <= d.date() <= train_end]
    train_dates = train_trading_days[::10]
    logger.info(f"Sampling {len(train_dates)} training dates from {len(train_trading_days)} total trading days in training period")
    
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
            
            # Get labels for this date
            date_labels = labels_df[
                (labels_df['date'] == train_date) & 
                (labels_df['horizon'] == args.horizon)
            ]
            
            if len(date_labels) == 0:
                continue
            
            # Ensure date and asset_id columns exist
            if 'date' not in features_df.columns:
                features_df['date'] = train_date
            if 'asset_id' not in features_df.columns:
                logger.warning(f"Features for {train_date} missing asset_id, skipping")
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
            if successful_dates == 0:  # Only log first error in detail
                logger.warning(f"Error building features for {train_date}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            continue
    
    logger.info(f"Successfully built features for {successful_dates} out of {len(train_dates)} training dates")
    
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
    
    # Run backtest
    logger.info("Running backtest...")
    
    # Initialize strategy
    strategy = LongTopKStrategy(k=args.top_k, min_score_threshold=-np.inf)
    
    # Generate predictions and weights for each trading day
    all_weights = []
    
    # Sample backtest dates (every day)
    backtest_dates = [d.date() for d in trading_days if start_date <= d.date() <= end_date]
    
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
            
            # Ensure date column exists
            if 'date' not in features_df.columns:
                features_df['date'] = backtest_date
            
            # Extract feature columns
            feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
            X = features_df[feature_cols].copy()
            # Convert to numeric and fill NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
            
            # Predict
            predictions = model.predict(X)
            
            # Create scores DataFrame
            scores_df = pd.DataFrame({
                'asset_id': features_df['asset_id'].values,
                'score': predictions,
                'confidence': np.ones(len(predictions))  # Placeholder
            })
            
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
    
    # Get symbol mapping for benchmarks (query database)
    symbol_map = {}
    try:
        assets_df = storage.query("SELECT asset_id, symbol FROM assets")
        if len(assets_df) > 0:
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
    logger.info("Computing performance metrics...")
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    # Store all results for comparison
    all_results = {
        'ml_strategy': {
            'name': f"{args.model}_{args.horizon}d_top{args.top_k}",
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    }
    
    # Run benchmarks if requested
    if args.run_benchmarks:
        logger.info("Running benchmark strategies...")
        benchmarks = BenchmarkStrategies(backtester)
        
        # Determine which benchmarks to run
        benchmark_symbols = []
        benchmark_config = getattr(config, 'benchmarks', {})
        benchmark_definitions = benchmark_config.get('definitions', {})
        default_benchmarks = benchmark_config.get('default', ['sp500'])
        
        if args.benchmark:
            # Single benchmark specified
            if args.benchmark in benchmark_definitions:
                ticker = benchmark_definitions[args.benchmark]['ticker']
                benchmark_symbols = [ticker]
            else:
                # Assume it's a ticker symbol directly
                benchmark_symbols = [args.benchmark.upper()]
        elif args.benchmarks:
            # Multiple benchmarks specified
            bench_names = [b.strip() for b in args.benchmarks.split(',')]
            for bench_name in bench_names:
                if bench_name in benchmark_definitions:
                    ticker = benchmark_definitions[bench_name]['ticker']
                    benchmark_symbols.append(ticker)
                else:
                    # Assume it's a ticker symbol directly
                    benchmark_symbols.append(bench_name.upper())
        else:
            # Use defaults from config
            for bench_name in default_benchmarks:
                if bench_name in benchmark_definitions:
                    ticker = benchmark_definitions[bench_name]['ticker']
                    benchmark_symbols.append(ticker)
        
        # Fallback to SPY if no benchmarks configured
        if not benchmark_symbols:
            benchmark_symbols = ['SPY']
        
        logger.info(f"Running benchmarks: {benchmark_symbols}")
        
        # Add benchmark symbols to prices_df if not present
        for symbol in benchmark_symbols:
            if 'symbol' not in prices_df.columns or len(prices_df[prices_df.get('symbol') == symbol]) == 0:
                # Try to add benchmark to prices_df
                symbol_df = storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{symbol}'")
                if len(symbol_df) > 0:
                    asset_id = symbol_df['asset_id'].iloc[0]
                    # Get bars for this benchmark
                    bench_bars = api.get_bars_asof(end_date, universe={asset_id})
                    if len(bench_bars) > 0:
                        bench_bars['date'] = pd.to_datetime(bench_bars['date']).dt.date
                        bench_bars = bench_bars[
                            (bench_bars['date'] >= start_date) & 
                            (bench_bars['date'] <= end_date)
                        ]
                        if len(bench_bars) > 0:
                            bench_bars['symbol'] = symbol
                            # Merge with prices_df
                            prices_df = pd.concat([prices_df, bench_bars[['date', 'asset_id', 'adj_close', 'symbol']]], ignore_index=True)
        
        # Run all benchmarks
        benchmark_results = benchmarks.run_benchmarks(prices_df, benchmark_symbols)
        
        for symbol, equity_curve in benchmark_results.items():
            try:
                metrics = PerformanceMetrics.compute_metrics(equity_curve)
                display_name = benchmark_definitions.get(
                    next((k for k, v in benchmark_definitions.items() if v['ticker'] == symbol), None),
                    {}
                ).get('name', symbol)
                
                result_key = f"{symbol.lower()}_buy_and_hold"
                all_results[result_key] = {
                    'name': f'{display_name} ({symbol}) Buy and Hold',
                    'equity_curve': equity_curve,
                    'metrics': metrics
                }
                logger.info(f"{symbol} buy-and-hold benchmark complete")
            except Exception as e:
                logger.warning(f"Failed to process {symbol} benchmark: {e}")
        
        # Equal-weight universe
        try:
            ew_equity = benchmarks.equal_weight_universe(prices_df, rebalance_frequency="monthly")
            ew_metrics = PerformanceMetrics.compute_metrics(ew_equity)
            all_results['equal_weight_universe'] = {
                'name': 'Equal Weight Universe (Monthly)',
                'equity_curve': ew_equity,
                'metrics': ew_metrics
            }
            logger.info("Equal-weight universe benchmark complete")
        except Exception as e:
            logger.warning(f"Failed to run equal-weight universe: {e}")
        
        # Random portfolios (if requested)
        if args.random_portfolio_runs > 0:
            logger.info(f"Running {args.random_portfolio_runs} random portfolio simulations...")
            random_results = []
            for seed in range(args.random_portfolio_runs):
                try:
                    random_equity = benchmarks.random_portfolio(prices_df, k=args.top_k, seed=seed)
                    random_metrics = PerformanceMetrics.compute_metrics(random_equity)
                    random_results.append({
                        'seed': seed,
                        'equity_curve': random_equity,
                        'metrics': random_metrics
                    })
                except Exception as e:
                    logger.warning(f"Failed to run random portfolio {seed}: {e}")
            
            if len(random_results) > 0:
                # Compute distribution statistics
                random_sharpes = [r['metrics']['sharpe_ratio'] for r in random_results]
                random_cagrs = [r['metrics']['cagr'] for r in random_results]
                
                all_results['random_portfolios'] = {
                    'name': f'Random Portfolios (K={args.top_k}, N={len(random_results)})',
                    'individual_results': random_results,
                    'distribution': {
                        'sharpe_mean': np.mean(random_sharpes),
                        'sharpe_std': np.std(random_sharpes),
                        'sharpe_1sigma_low': np.mean(random_sharpes) - np.std(random_sharpes),
                        'sharpe_1sigma_high': np.mean(random_sharpes) + np.std(random_sharpes),
                        'sharpe_3sigma_low': np.mean(random_sharpes) - 3 * np.std(random_sharpes),
                        'sharpe_3sigma_high': np.mean(random_sharpes) + 3 * np.std(random_sharpes),
                        'cagr_mean': np.mean(random_cagrs),
                        'cagr_std': np.std(random_cagrs),
                        'cagr_1sigma_low': np.mean(random_cagrs) - np.std(random_cagrs),
                        'cagr_1sigma_high': np.mean(random_cagrs) + np.std(random_cagrs),
                        'cagr_3sigma_low': np.mean(random_cagrs) - 3 * np.std(random_cagrs),
                        'cagr_3sigma_high': np.mean(random_cagrs) + 3 * np.std(random_cagrs),
                    }
                }
                
                # Check if ML strategy is above 3-sigma
                ml_sharpe = metrics['sharpe_ratio']
                ml_cagr = metrics['cagr']
                sharpe_3sigma_high = all_results['random_portfolios']['distribution']['sharpe_3sigma_high']
                cagr_3sigma_high = all_results['random_portfolios']['distribution']['cagr_3sigma_high']
                
                logger.info(f"Random portfolio distribution: Sharpe mean={np.mean(random_sharpes):.3f}±{np.std(random_sharpes):.3f}, "
                          f"CAGR mean={np.mean(random_cagrs):.2%}±{np.std(random_cagrs):.2%}")
                logger.info(f"ML strategy Sharpe={ml_sharpe:.3f} (3σ={sharpe_3sigma_high:.3f}), "
                          f"CAGR={ml_cagr:.2%} (3σ={cagr_3sigma_high:.2%})")
    
    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Universe: {len(universe)} assets")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon} days")
    print(f"Top K: {args.top_k}")
    print("\n" + "-"*80)
    print("ML STRATEGY PERFORMANCE:")
    print("-"*80)
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"  VaR (5%): {metrics['var_5pct']:.4f}")
    print(f"  CVaR (5%): {metrics['cvar_5pct']:.4f}")
    
    # Print benchmark comparisons
    if args.run_benchmarks:
        print("\n" + "-"*80)
        print("BENCHMARK COMPARISONS:")
        print("-"*80)
        
        # Comparison table
        comparison_data = []
        comparison_data.append({
            'Strategy': all_results['ml_strategy']['name'],
            'CAGR': metrics['cagr'],
            'Sharpe': metrics['sharpe_ratio'],
            'Max DD': metrics['max_drawdown'],
            'Calmar': metrics['calmar_ratio']
        })
        
        # Add all benchmark results (buy-and-hold benchmarks)
        for key in sorted(all_results.keys()):
            if key.endswith('_buy_and_hold') or key == 'equal_weight_universe':
                bm = all_results[key]
                comparison_data.append({
                    'Strategy': bm['name'],
                    'CAGR': bm['metrics']['cagr'],
                    'Sharpe': bm['metrics']['sharpe_ratio'],
                    'Max DD': bm['metrics']['max_drawdown'],
                    'Calmar': bm['metrics']['calmar_ratio']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Random portfolio distribution summary
        if 'random_portfolios' in all_results:
            rp_dist = all_results['random_portfolios']['distribution']
            print("\n" + "-"*80)
            print("RANDOM PORTFOLIO DISTRIBUTION:")
            print("-"*80)
            print(f"  Sharpe: {rp_dist['sharpe_mean']:.3f} ± {rp_dist['sharpe_std']:.3f}")
            print(f"    1σ range: [{rp_dist['sharpe_1sigma_low']:.3f}, {rp_dist['sharpe_1sigma_high']:.3f}]")
            print(f"    3σ range: [{rp_dist['sharpe_3sigma_low']:.3f}, {rp_dist['sharpe_3sigma_high']:.3f}]")
            print(f"  ML Strategy Sharpe: {ml_sharpe:.3f}")
            if ml_sharpe > rp_dist['sharpe_3sigma_high']:
                print(f"    ✓ ML strategy exceeds 3σ threshold (statistically significant)")
            elif ml_sharpe > rp_dist['sharpe_1sigma_high']:
                print(f"    ~ ML strategy exceeds 1σ threshold")
            else:
                print(f"    ✗ ML strategy within random distribution")
            
            print(f"\n  CAGR: {rp_dist['cagr_mean']:.2%} ± {rp_dist['cagr_std']:.2%}")
            print(f"    1σ range: [{rp_dist['cagr_1sigma_low']:.2%}, {rp_dist['cagr_1sigma_high']:.2%}]")
            print(f"    3σ range: [{rp_dist['cagr_3sigma_low']:.2%}, {rp_dist['cagr_3sigma_high']:.2%}]")
            print(f"  ML Strategy CAGR: {ml_cagr:.2%}")
            if ml_cagr > rp_dist['cagr_3sigma_high']:
                print(f"    ✓ ML strategy exceeds 3σ threshold (statistically significant)")
            elif ml_cagr > rp_dist['cagr_1sigma_high']:
                print(f"    ~ ML strategy exceeds 1σ threshold")
            else:
                print(f"    ✗ ML strategy within random distribution")
    
    print("="*80)
    
    # Cost sensitivity analysis
    if args.cost_sensitivity:
        logger.info("Running cost sensitivity analysis...")
        cost_results = run_cost_sensitivity_analysis(
            prices_df=prices_df,
            target_weights_df=target_weights_df,
            cost_levels=args.cost_levels,
            initial_capital=args.initial_capital
        )
        
        print("\n" + "-"*80)
        print("COST SENSITIVITY ANALYSIS:")
        print("-"*80)
        print(cost_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("-"*80)
        
        # Store for saving
        all_results['cost_sensitivity'] = cost_results.to_dict('records')
    
    # Walk-forward evaluation
    if args.walk_forward:
        logger.info("Running walk-forward evaluation...")
        wf_results = run_walk_forward_evaluation(
            storage=storage,
            api=api,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            train_years=args.walk_forward_train_years,
            test_years=args.walk_forward_test_years,
            model_type=args.model,
            horizon=args.horizon,
            top_k=args.top_k,
            initial_capital=args.initial_capital,
            config=config
        )
        
        print("\n" + "-"*80)
        print("WALK-FORWARD EVALUATION:")
        print("-"*80)
        
        if len(wf_results['windows']) > 0:
            for w in wf_results['windows']:
                print(f"Window {w['window']}: {w['test_start']} to {w['test_end']}")
                print(f"  Sharpe: {w['metrics']['sharpe_ratio']:.3f}, CAGR: {w['metrics']['cagr']:.2%}, Max DD: {w['metrics']['max_drawdown']:.2%}")
            
            agg = wf_results['aggregate_metrics']
            print("\nAggregate Results:")
            print(f"  Avg Sharpe: {agg['avg_sharpe']:.3f} ± {agg['std_sharpe']:.3f}")
            print(f"  Avg CAGR: {agg['avg_cagr']:.2%} ± {agg['std_cagr']:.2%}")
            print(f"  Avg Max DD: {agg['avg_max_dd']:.2%}")
            print(f"  Positive Sharpe Windows: {agg['positive_sharpe_windows']}/{agg['num_windows']}")
        else:
            print("  No walk-forward windows completed successfully.")
        print("-"*80)
        
        # Store for saving
        all_results['walk_forward'] = wf_results
    
    # Policy-driven backtest (uses retraining config)
    if args.policy_driven:
        logger.info("Running policy-driven backtest...")
        
        # Override config if command-line args provided
        if args.retrain_cadence is not None:
            config.retraining.cadence_days = args.retrain_cadence
            logger.info(f"Overriding retrain cadence to {args.retrain_cadence} days")
        if args.time_decay_lambda is not None:
            config.retraining.time_decay.lambda_ = args.time_decay_lambda
            logger.info(f"Overriding time-decay lambda to {args.time_decay_lambda}")
        
        policy_results = run_policy_driven_backtest(
            storage=storage,
            api=api,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            model_type=args.model,
            horizon=args.horizon,
            top_k=args.top_k,
            initial_capital=args.initial_capital,
            config=config
        )
        
        print("\n" + "-"*80)
        print("POLICY-DRIVEN BACKTEST:")
        print("-"*80)
        
        if 'error' not in policy_results:
            policy_metrics = policy_results['metrics']
            policy_config = policy_results['config']
            
            print(f"  Retraining Policy:")
            print(f"    Cadence: {policy_config['cadence_days']} days")
            print(f"    Window: {policy_config['window_type']}, {policy_config['window_years']} years")
            print(f"    Time-decay: {'enabled' if policy_config['time_decay_enabled'] else 'disabled'}")
            if policy_config['time_decay_enabled']:
                print(f"    Lambda: {policy_config['time_decay_lambda']}")
            print(f"  Number of Retrains: {policy_results['num_retrains']}")
            print(f"\n  Performance:")
            print(f"    Sharpe: {policy_metrics['sharpe_ratio']:.3f}")
            print(f"    CAGR: {policy_metrics['cagr']:.2%}")
            print(f"    Max DD: {policy_metrics['max_drawdown']:.2%}")
            print(f"    Calmar: {policy_metrics['calmar_ratio']:.3f}")
            
            # Compare with baseline (single-train) if available
            baseline_sharpe = metrics['sharpe_ratio']
            improvement = ((policy_metrics['sharpe_ratio'] / baseline_sharpe) - 1) * 100 if baseline_sharpe > 0 else 0
            print(f"\n  vs Single-Train Baseline:")
            print(f"    Sharpe improvement: {improvement:+.1f}%")
            
            # Show retraining dates
            if policy_results['retraining_history']:
                print(f"\n  Retraining History:")
                for rt in policy_results['retraining_history'][:5]:  # Show first 5
                    print(f"    {rt['date']} (day {rt['day_index']}, v{rt['model_version']})")
                if len(policy_results['retraining_history']) > 5:
                    print(f"    ... and {len(policy_results['retraining_history']) - 5} more")
        else:
            print(f"  Error: {policy_results['error']}")
        print("-"*80)
        
        all_results['policy_driven'] = policy_results
    
    # Regime-aware modeling
    if args.regime_aware:
        logger.info("Running regime-aware backtest...")
        regime_results = run_regime_aware_backtest(
            storage=storage,
            api=api,
            universe=universe,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=end_date,
            model_type=args.model,
            horizon=args.horizon,
            top_k=args.top_k,
            initial_capital=args.initial_capital,
            config=config
        )
        
        print("\n" + "-"*80)
        print("REGIME-AWARE MODELING:")
        print("-"*80)
        
        if 'error' not in regime_results:
            regime_metrics = regime_results['metrics']
            print(f"  Num Regimes: {regime_results['num_regimes']}")
            print(f"  Sharpe: {regime_metrics['sharpe_ratio']:.3f}")
            print(f"  CAGR: {regime_metrics['cagr']:.2%}")
            print(f"  Max DD: {regime_metrics['max_drawdown']:.2%}")
            
            # Compare with baseline
            baseline_sharpe = metrics['sharpe_ratio']
            improvement = ((regime_metrics['sharpe_ratio'] / baseline_sharpe) - 1) * 100 if baseline_sharpe > 0 else 0
            print(f"  vs Baseline Sharpe: {improvement:+.1f}%")
        else:
            print(f"  Error: {regime_results['error']}")
        print("-"*80)
        
        all_results['regime_aware'] = regime_results
    
    # Ablation tests
    if args.ablation_test:
        logger.info("Running ablation tests...")
        ablation_results = run_ablation_tests(
            storage=storage,
            api=api,
            universe=universe,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=end_date,
            model_type=args.model,
            horizon=args.horizon,
            top_k=args.top_k,
            initial_capital=args.initial_capital,
            config=config
        )
        
        print("\n" + "-"*80)
        print("ABLATION TESTS:")
        print("-"*80)
        
        baseline_sharpe = ablation_results.get('baseline', {}).get('metrics', {}).get('sharpe_ratio', 0)
        print(f"  Baseline Sharpe: {baseline_sharpe:.3f}")
        print("\n  Feature Importance (by Sharpe degradation when removed):")
        
        importance_scores = []
        for key, result in ablation_results.items():
            if key == 'baseline':
                continue
            if 'metrics' in result:
                ablated_sharpe = result['metrics']['sharpe_ratio']
                importance = baseline_sharpe - ablated_sharpe
                importance_scores.append((key.replace('no_', ''), importance, ablated_sharpe))
        
        # Sort by importance (most important = biggest drop when removed)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance, sharpe in importance_scores:
            sign = "+" if importance > 0 else ""
            print(f"    {feature:20s}: Sharpe={sharpe:.3f} (impact: {sign}{importance:.3f})")
        print("-"*80)
        
        all_results['ablation'] = ablation_results
    
    # Save results
    output_path = Path("artifacts") / "backtest_results"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save ML equity curve
    equity_curve.to_csv(output_path / f"equity_curve_{args.model}_{args.horizon}d.csv", index=False)
    logger.info(f"ML equity curve saved to {output_path / f'equity_curve_{args.model}_{args.horizon}d.csv'}")
    
    # Save benchmark equity curves
    if args.run_benchmarks:
        for key in sorted(all_results.keys()):
            if (key.endswith('_buy_and_hold') or key == 'equal_weight_universe') and key in all_results:
                bm = all_results[key]
                safe_name = bm['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
                bm['equity_curve'].to_csv(output_path / f"equity_curve_{safe_name}.csv", index=False)
                logger.info(f"Benchmark equity curve saved: {safe_name}")
    
    # Save comparison metrics summary with full experiment metadata
    import subprocess
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()[:8]
    except Exception:
        git_commit = 'unknown'
    
    comparison_summary = {
        'experiment_metadata': {
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_commit,
            'config_hash': get_config_hash(dict(config.features)) if hasattr(config, 'features') else None,
        },
        'backtest_config': {
            'start_date': str(start_date),
            'end_date': str(end_date),
            'train_start': str(train_start),
            'train_end': str(train_end),
            'universe_size': len(universe),
            'model': args.model,
            'horizon': args.horizon,
            'top_k': args.top_k,
            'initial_capital': args.initial_capital,
            'run_benchmarks': args.run_benchmarks,
            'random_portfolio_runs': args.random_portfolio_runs,
            'cost_sensitivity': args.cost_sensitivity,
            'walk_forward': args.walk_forward,
            'regime_aware': args.regime_aware,
            'ablation_test': args.ablation_test,
        },
        'feature_config': dict(config.features) if hasattr(config, 'features') else {},
        'cost_config': dict(config.costs) if hasattr(config, 'costs') else {},
        'strategies': {}
    }
    
    for key, result in all_results.items():
        if key == 'random_portfolios':
            # Store distribution summary only
            comparison_summary['strategies'][key] = {
                'name': result['name'],
                'distribution': result['distribution']
            }
        elif key in ['cost_sensitivity', 'walk_forward', 'regime_aware', 'ablation']:
            # Store these special results directly
            comparison_summary[key] = result
        elif 'name' in result and 'metrics' in result:
            comparison_summary['strategies'][key] = {
                'name': result['name'],
                'metrics': result['metrics']
            }
    
    summary_path = output_path / f"comparison_summary_{args.model}_{args.horizon}d.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    logger.info(f"Comparison summary saved to {summary_path}")
    
    storage.close()


if __name__ == "__main__":
    main()

