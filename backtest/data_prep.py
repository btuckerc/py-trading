"""Data preparation utilities for backtesting.

This module handles:
- Sampling training dates
- Building features and labels
- Preparing data for model training via TabularTrainer
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from data.asof_api import AsOfQueryAPI
from data.storage import StorageBackend
from data.universe import TradingCalendar
from labels.returns import ReturnLabelGenerator
from features.pipeline import FeaturePipeline
from models.tabular_trainer import TabularTrainer, TrainingConfig, TrainingResult, SamplingStrategy


@dataclass
class BacktestDataConfig:
    """Configuration for backtest data preparation."""
    
    # Date ranges
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    
    # Horizons
    horizons: List[int]
    
    # Sampling
    sample_every_n_days: int = 10
    
    # Feature settings
    feature_lookback_days: int = 252
    
    # Label settings
    benchmark_symbol: str = "SPY"


def get_universe_asset_ids(storage: StorageBackend, symbols: List[str]) -> Set[int]:
    """Get asset_ids for given symbols."""
    if not symbols:
        return set()
    symbol_list = "', '".join(symbols)
    query = f"SELECT asset_id FROM assets WHERE symbol IN ('{symbol_list}')"
    df = storage.query(query)
    return set(df['asset_id'].values) if len(df) > 0 else set()


def sample_trading_dates(
    start_date: date,
    end_date: date,
    sample_every_n: int = 10,
) -> List[date]:
    """
    Sample trading dates from a date range.
    
    Args:
        start_date: Start of range
        end_date: End of range
        sample_every_n: Sample every N-th trading day
        
    Returns:
        List of sampled dates
    """
    calendar = TradingCalendar()
    trading_days = calendar.get_trading_days(start_date, end_date)
    all_dates = [d.date() for d in trading_days]
    return all_dates[::sample_every_n]


def build_training_data(
    api: AsOfQueryAPI,
    storage: StorageBackend,
    config: BacktestDataConfig,
    universe: Set[int],
    feature_config: Optional[dict] = None,
    time_decay_enabled: bool = False,
    time_decay_lambda: float = 0.001,
) -> TrainingResult:
    """
    Build training data using TabularTrainer.
    
    Args:
        api: AsOfQueryAPI instance
        storage: StorageBackend instance
        config: BacktestDataConfig
        universe: Set of asset_ids
        feature_config: Optional feature configuration
        time_decay_enabled: Whether to use time-decay weighting
        time_decay_lambda: Time-decay lambda parameter
        
    Returns:
        TrainingResult with trained model placeholder (model=None)
        Use this to get feature names and training data statistics
    """
    from models.tabular import XGBoostModel
    
    # Initialize components
    feature_pipeline = FeaturePipeline(api, feature_config)
    label_generator = ReturnLabelGenerator(storage)
    
    # Build TrainingConfig
    training_config = TrainingConfig(
        window_start=config.train_start,
        window_end=config.train_end,
        horizons=config.horizons,
        sampling=SamplingStrategy(sample_every_n_days=config.sample_every_n_days),
        time_decay_enabled=time_decay_enabled,
        time_decay_lambda=time_decay_lambda,
        feature_lookback_days=config.feature_lookback_days,
        benchmark_symbol=config.benchmark_symbol,
    )
    
    # Initialize trainer
    trainer = TabularTrainer(
        feature_pipeline=feature_pipeline,
        label_generator=label_generator,
        storage=storage,
        api=api,
    )
    
    # Train a simple model to get data statistics
    # The actual model training is done by the experiment functions
    result = trainer.train(
        model_class=XGBoostModel,
        model_params={"task_type": "regression", "n_estimators": 10},  # Minimal model
        config=training_config,
        universe=universe,
    )
    
    return result


def prepare_test_data(
    api: AsOfQueryAPI,
    config: BacktestDataConfig,
    universe: Set[int],
    feature_config: Optional[dict] = None,
) -> Tuple[List[date], pd.DataFrame]:
    """
    Prepare test period data.
    
    Args:
        api: AsOfQueryAPI instance
        config: BacktestDataConfig
        universe: Set of asset_ids
        feature_config: Optional feature configuration
        
    Returns:
        Tuple of (test_dates, prices_df)
    """
    calendar = TradingCalendar()
    
    # Get test trading days
    test_trading_days = calendar.get_trading_days(config.test_start, config.test_end)
    test_dates = [d.date() for d in test_trading_days]
    
    # Get prices for test period
    all_bars = api.get_bars_asof(config.test_end, universe=universe)
    all_bars['date'] = pd.to_datetime(all_bars['date']).dt.date
    prices_df = all_bars[
        (all_bars['date'] >= config.test_start) &
        (all_bars['date'] <= config.test_end)
    ].copy()
    
    return test_dates, prices_df


def check_data_coverage(
    storage: StorageBackend,
    start_date: date,
    end_date: date,
    symbols: Optional[List[str]] = None,
) -> dict:
    """
    Check data coverage for a date range.
    
    Args:
        storage: StorageBackend instance
        start_date: Start date
        end_date: End date
        symbols: Optional list of symbols to check
        
    Returns:
        Dict with coverage statistics
    """
    # Get current data range
    result = storage.query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM bars_daily")
    
    if len(result) == 0 or result['max_date'].iloc[0] is None:
        return {
            'has_data': False,
            'min_date': None,
            'max_date': None,
            'covers_range': False,
        }
    
    db_min_date = result['min_date'].iloc[0]
    db_max_date = result['max_date'].iloc[0]
    
    # Convert to date objects if needed
    if hasattr(db_min_date, 'date'):
        db_min_date = db_min_date.date()
    if hasattr(db_max_date, 'date'):
        db_max_date = db_max_date.date()
    
    covers_range = db_min_date <= start_date and db_max_date >= end_date
    
    return {
        'has_data': True,
        'min_date': db_min_date,
        'max_date': db_max_date,
        'covers_range': covers_range,
        'missing_start': start_date < db_min_date,
        'missing_end': end_date > db_max_date,
    }

