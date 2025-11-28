"""Naive baseline strategies (no ML, just rules)."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date


def persistence_strategy(
    bars_df: pd.DataFrame,
    horizon: int = 1
) -> pd.Series:
    """
    Naive persistence: predict tomorrow's return = today's return.
    
    Returns:
        Series of predicted returns indexed by (date, asset_id)
    """
    bars_df = bars_df.set_index(['date', 'asset_id']).sort_index()
    returns = np.log(bars_df['adj_close'] / bars_df['adj_close'].shift(1))
    
    # Predict next return = current return
    predictions = returns.shift(-horizon)
    
    return predictions


def momentum_strategy(
    bars_df: pd.DataFrame,
    lookback_window: int = 252,
    top_decile: bool = True
) -> pd.Series:
    """
    Cross-sectional momentum: long top decile, short bottom decile.
    
    Returns:
        Series of predicted excess returns
    """
    bars_df = bars_df.set_index(['date', 'asset_id']).sort_index()
    
    # Compute momentum (12-month return)
    prices = bars_df['adj_close']
    momentum = np.log(prices / prices.shift(lookback_window))
    
    # Rank by momentum
    momentum_ranks = momentum.groupby('date').rank(pct=True)
    
    # Top decile = 1, bottom decile = -1, else 0
    predictions = pd.Series(index=momentum.index, dtype=float)
    for date_idx, date_group in momentum_ranks.groupby('date'):
        top_threshold = 0.9
        bottom_threshold = 0.1
        
        top_assets = date_group[date_group >= top_threshold].index
        bottom_assets = date_group[date_group <= bottom_threshold].index
        
        predictions.loc[top_assets] = 1.0
        predictions.loc[bottom_assets] = -1.0
        predictions.loc[date_group.index.difference(top_assets.union(bottom_assets))] = 0.0
    
    return predictions


def mean_reversion_strategy(
    bars_df: pd.DataFrame,
    lookback_window: int = 20,
    z_threshold: float = 2.0
) -> pd.Series:
    """
    Mean reversion: predict reversal after extreme moves.
    
    Returns:
        Series of predicted returns
    """
    bars_df = bars_df.set_index(['date', 'asset_id']).sort_index()
    
    # Compute returns
    returns = np.log(bars_df['adj_close'] / bars_df['adj_close'].shift(1))
    
    # Z-score of returns
    mean_returns = returns.groupby('asset_id').rolling(lookback_window).mean()
    std_returns = returns.groupby('asset_id').rolling(lookback_window).std()
    z_scores = (returns - mean_returns) / (std_returns + 1e-10)
    
    # Predict reversal: if z-score > threshold, predict negative return
    predictions = pd.Series(index=returns.index, dtype=float)
    predictions[z_scores > z_threshold] = -0.01  # Predict negative return
    predictions[z_scores < -z_threshold] = 0.01   # Predict positive return
    predictions[(z_scores >= -z_threshold) & (z_scores <= z_threshold)] = 0.0
    
    return predictions

