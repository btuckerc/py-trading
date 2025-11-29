"""Experiment orchestration for backtesting.

This module handles:
- Walk-forward evaluation
- Policy-driven backtests (using retraining config)
- Regime-aware modeling
- Ablation tests
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np
from loguru import logger

from data.asof_api import AsOfQueryAPI
from data.storage import StorageBackend
from data.universe import TradingCalendar
from labels.returns import ReturnLabelGenerator
from labels.regimes import RegimeLabelGenerator
from features.pipeline import FeaturePipeline
from models.tabular import XGBoostModel, LightGBMModel
from models.training import WalkForwardRetrainer
from models.tabular_trainer import TabularTrainer, TrainingConfig, SamplingStrategy
from portfolio.strategies import LongTopKStrategy
from backtest.vectorized import VectorizedBacktester
from backtest.metrics import PerformanceMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a backtest experiment."""
    
    # Date ranges
    start_date: date
    end_date: date
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    
    # Model settings
    model_type: str = "xgboost"
    horizon: int = 20
    
    # Strategy settings
    top_k: int = 3
    
    # Capital
    initial_capital: float = 100000.0
    
    # Feature config (from base.yaml)
    feature_config: Dict = field(default_factory=dict)
    
    # Retraining config
    retraining_config: Any = None


@dataclass
class ExperimentResult:
    """Result of a backtest experiment."""
    
    name: str
    metrics: Dict[str, float]
    equity_curve: Optional[pd.DataFrame] = None
    additional_data: Dict = field(default_factory=dict)
    error: Optional[str] = None


def get_model_class(model_type: str):
    """Get model class from type string."""
    if model_type == "xgboost":
        return XGBoostModel
    elif model_type == "lightgbm":
        return LightGBMModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_walk_forward_evaluation(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: Set[int],
    config: ExperimentConfig,
    train_years: int = 3,
    test_years: int = 1,
) -> ExperimentResult:
    """
    Run walk-forward evaluation: train on rolling window, test on next period.
    
    Args:
        storage: StorageBackend instance
        api: AsOfQueryAPI instance
        universe: Set of asset_ids
        config: ExperimentConfig
        train_years: Training window size in years
        test_years: Test window size in years
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    calendar = TradingCalendar()
    
    # Calculate windows
    windows = []
    current_train_start = config.start_date
    
    while True:
        train_end = (pd.Timestamp(current_train_start) + pd.DateOffset(years=train_years)).date()
        test_start = train_end
        test_end = (pd.Timestamp(test_start) + pd.DateOffset(years=test_years)).date()
        
        if test_end > config.end_date:
            break
        
        windows.append({
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_train_start = (pd.Timestamp(current_train_start) + pd.DateOffset(years=1)).date()
    
    if len(windows) == 0:
        return ExperimentResult(
            name="walk_forward",
            metrics={},
            error="No walk-forward windows generated (period too short)"
        )
    
    logger.info(f"Running walk-forward with {len(windows)} windows")
    
    # Initialize components
    feature_pipeline = FeaturePipeline(api, config.feature_config)
    label_generator = ReturnLabelGenerator(storage)
    model_class = get_model_class(config.model_type)
    model_params = {"task_type": "regression", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    
    window_results = []
    
    for i, window in enumerate(windows):
        logger.info(f"Window {i+1}/{len(windows)}: train {window['train_start']} to {window['train_end']}, test {window['test_start']} to {window['test_end']}")
        
        try:
            # Build TrainingConfig
            training_config = TrainingConfig(
                window_start=window['train_start'],
                window_end=window['train_end'],
                horizons=[config.horizon],
                sampling=SamplingStrategy(sample_every_n_days=10),
                time_decay_enabled=config.retraining_config.time_decay.enabled if config.retraining_config else False,
                time_decay_lambda=config.retraining_config.time_decay.lambda_ if config.retraining_config else 0.001,
            )
            
            # Train using TabularTrainer
            trainer = TabularTrainer(
                feature_pipeline=feature_pipeline,
                label_generator=label_generator,
                storage=storage,
                api=api,
            )
            
            training_result = trainer.train(
                model_class=model_class,
                model_params=model_params,
                config=training_config,
                universe=universe,
            )
            
            model = training_result.model
            
            # Run test period
            strategy = LongTopKStrategy(k=config.top_k, min_score_threshold=-np.inf)
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
            backtester = VectorizedBacktester(initial_capital=config.initial_capital)
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
    
    return ExperimentResult(
        name="walk_forward",
        metrics=aggregate_metrics,
        additional_data={'windows': window_results}
    )


def run_policy_driven_backtest(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: Set[int],
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    Run a backtest using the retraining policy from config.
    
    This simulates exactly what would happen in live trading:
    - Model is retrained according to cadence_days
    - Time-decay weighting is applied to training samples
    - Adaptive retraining triggers are checked
    
    Args:
        storage: StorageBackend instance
        api: AsOfQueryAPI instance
        universe: Set of asset_ids
        config: ExperimentConfig
        
    Returns:
        ExperimentResult with metrics and retraining history
    """
    calendar = TradingCalendar()
    
    # Get retraining config
    retraining_config = config.retraining_config
    if retraining_config is None:
        return ExperimentResult(
            name="policy_driven",
            metrics={},
            error="No retraining config provided"
        )
    
    logger.info(f"Running policy-driven backtest with cadence={retraining_config.cadence_days} days")
    
    # Initialize components
    feature_pipeline = FeaturePipeline(api, config.feature_config)
    label_generator = ReturnLabelGenerator(storage)
    model_class = get_model_class(config.model_type)
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
        horizon=config.horizon
    )
    
    # Get trading days
    trading_days = [d.date() for d in calendar.get_trading_days(config.start_date, config.end_date)]
    
    # Strategy
    strategy = LongTopKStrategy(k=config.top_k, min_score_threshold=-np.inf)
    
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
            
        except Exception as e:
            logger.warning(f"Error on {trading_date}: {e}")
            continue
    
    if len(all_weights) == 0:
        return ExperimentResult(
            name="policy_driven",
            metrics={},
            additional_data={'retraining_history': retraining_history},
            error="No weights generated"
        )
    
    target_weights_df = pd.concat(all_weights, ignore_index=True)
    
    # Get prices
    all_bars = api.get_bars_asof(config.end_date, universe=universe)
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    prices_df = all_bars[
        (all_bars['date'] >= config.start_date) &
        (all_bars['date'] <= config.end_date)
    ].copy()
    
    # Run backtest
    backtester = VectorizedBacktester(initial_capital=config.initial_capital)
    equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
    
    if len(equity_curve) == 0:
        return ExperimentResult(
            name="policy_driven",
            metrics={},
            additional_data={'retraining_history': retraining_history},
            error="Empty equity curve"
        )
    
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    return ExperimentResult(
        name="policy_driven",
        metrics=metrics,
        equity_curve=equity_curve,
        additional_data={
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
    )


def run_regime_aware_backtest(
    storage: StorageBackend,
    api: AsOfQueryAPI,
    universe: Set[int],
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    Train separate models per market regime and blend predictions.
    
    Args:
        storage: StorageBackend instance
        api: AsOfQueryAPI instance
        universe: Set of asset_ids
        config: ExperimentConfig
        
    Returns:
        ExperimentResult with per-regime metrics
    """
    logger.info("Running regime-aware backtest...")
    
    train_start = config.train_start or config.start_date
    train_end = config.train_end or (pd.Timestamp(config.start_date) + pd.DateOffset(years=3)).date()
    test_start = train_end
    test_end = config.end_date
    
    # Fit regimes
    regime_generator = RegimeLabelGenerator(storage, n_regimes=4)
    regimes_df = regime_generator.fit_regimes(train_start, train_end, method="kmeans")
    
    if len(regimes_df) == 0:
        return ExperimentResult(
            name="regime_aware",
            metrics={},
            error="Could not generate regime labels"
        )
    
    logger.info(f"Identified {regimes_df['regime_id'].nunique()} market regimes")
    
    # Generate labels
    label_generator = ReturnLabelGenerator(storage)
    labels_df = label_generator.generate_labels(
        start_date=train_start,
        end_date=train_end,
        horizons=[config.horizon],
        benchmark_symbol="SPY",
        universe=list(universe)
    )
    
    if len(labels_df) == 0:
        return ExperimentResult(
            name="regime_aware",
            metrics={},
            error="No labels generated"
        )
    
    regimes_df['date'] = pd.to_datetime(regimes_df['date']).dt.date
    
    # Build features
    feature_pipeline = FeaturePipeline(api, config.feature_config)
    calendar = TradingCalendar()
    
    trading_days = [d.date() for d in calendar.get_trading_days(train_start, train_end)][::10]
    
    # Collect training data with regime labels
    X_by_regime: Dict[int, List] = {}
    y_by_regime: Dict[int, List] = {}
    
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
    model_class = get_model_class(config.model_type)
    models = {}
    
    for regime_id in X_by_regime:
        if len(X_by_regime[regime_id]) == 0:
            continue
        
        X_train = pd.concat(X_by_regime[regime_id], ignore_index=True)
        y_train = np.concatenate(y_by_regime[regime_id])
        
        if len(X_train) < 100:
            logger.warning(f"Regime {regime_id} has only {len(X_train)} samples, skipping")
            continue
        
        logger.info(f"Training model for regime {regime_id} with {len(X_train)} samples")
        
        model = model_class(task_type="regression", n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X_train, y_train)
        models[regime_id] = model
    
    if len(models) == 0:
        return ExperimentResult(
            name="regime_aware",
            metrics={},
            error="No models trained"
        )
    
    logger.info(f"Trained {len(models)} regime-specific models")
    
    # Test period
    strategy = LongTopKStrategy(k=config.top_k, min_score_threshold=-np.inf)
    test_trading_days = [d.date() for d in calendar.get_trading_days(test_start, test_end)]
    
    all_weights = []
    
    for test_date in test_trading_days:
        try:
            current_regime, _ = regime_generator.predict_regime(test_date)
            
            if current_regime in models:
                model = models[current_regime]
            else:
                model = list(models.values())[0]
            
            features_df = feature_pipeline.build_features_cross_sectional(
                as_of_date=test_date,
                universe=universe,
                lookback_days=252
            )
            
            if len(features_df) == 0:
                continue
            
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
        return ExperimentResult(
            name="regime_aware",
            metrics={},
            error="No weights generated"
        )
    
    target_weights_df = pd.concat(all_weights, ignore_index=True)
    
    # Get prices and run backtest
    all_bars = api.get_bars_asof(test_end, universe=universe)
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    prices_df = all_bars[
        (all_bars['date'] >= test_start) &
        (all_bars['date'] <= test_end)
    ].copy()
    
    backtester = VectorizedBacktester(initial_capital=config.initial_capital)
    equity_curve = backtester.run_backtest(prices_df, target_weights_df, execution_lag=1)
    
    if len(equity_curve) == 0:
        return ExperimentResult(
            name="regime_aware",
            metrics={},
            error="Empty equity curve"
        )
    
    metrics = PerformanceMetrics.compute_metrics(equity_curve)
    
    return ExperimentResult(
        name="regime_aware",
        metrics=metrics,
        equity_curve=equity_curve,
        additional_data={
            'num_regimes': len(models),
            'regime_model_sizes': {r: len(X_by_regime.get(r, [[]])[0]) for r in models.keys()}
        }
    )

