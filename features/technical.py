"""Technical/price-based feature engineering."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import date
from data.asof_api import AsOfQueryAPI


class TechnicalFeatureBuilder:
    """Builds technical indicators from price/volume data."""
    
    def __init__(self, api: AsOfQueryAPI):
        self.api = api
    
    def compute_returns(
        self,
        prices: pd.Series,
        windows: List[int] = [1, 5, 20, 60, 120]
    ) -> pd.DataFrame:
        """Compute log returns over multiple windows."""
        features = pd.DataFrame(index=prices.index)
        
        for window in windows:
            features[f'log_return_{window}d'] = np.log(prices / prices.shift(window))
        
        # Rolling mean and std of daily returns
        daily_returns = np.log(prices / prices.shift(1))
        features['return_mean_20d'] = daily_returns.rolling(20).mean()
        features['return_std_20d'] = daily_returns.rolling(20).std()
        features['return_mean_60d'] = daily_returns.rolling(60).mean()
        features['return_std_60d'] = daily_returns.rolling(60).std()
        
        return features
    
    def compute_moving_averages(
        self,
        prices: pd.Series,
        ma_windows: List[int] = [5, 10, 20, 50, 100, 200],
        ema_windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """Compute simple and exponential moving averages."""
        features = pd.DataFrame(index=prices.index)
        
        # Simple MAs
        for window in ma_windows:
            features[f'ma_{window}d'] = prices.rolling(window).mean()
            features[f'price_to_ma_{window}d'] = prices / features[f'ma_{window}d']
        
        # EMAs
        for window in ema_windows:
            features[f'ema_{window}d'] = prices.ewm(span=window, adjust=False).mean()
            features[f'price_to_ema_{window}d'] = prices / features[f'ema_{window}d']
        
        # MA ratios
        if 10 in ma_windows and 50 in ma_windows:
            features['ma_10_to_ma_50'] = features['ma_10d'] / features['ma_50d']
        if 20 in ma_windows and 200 in ma_windows:
            features['ma_20_to_ma_200'] = features['ma_20d'] / features['ma_200d']
        
        return features
    
    def compute_momentum(
        self,
        prices: pd.Series,
        windows: List[int] = [20, 60, 120]
    ) -> pd.DataFrame:
        """Compute momentum measures."""
        features = pd.DataFrame(index=prices.index)
        
        for window in windows:
            features[f'momentum_{window}d'] = (prices / prices.shift(window)) - 1
        
        return features
    
    def compute_volatility(
        self,
        prices: pd.Series,
        windows: List[int] = [20, 60]
    ) -> pd.DataFrame:
        """Compute volatility metrics."""
        features = pd.DataFrame(index=prices.index)
        
        returns = np.log(prices / prices.shift(1))
        
        for window in windows:
            features[f'volatility_{window}d'] = returns.rolling(window).std()
        
        # True Range / ATR-like
        # Would need high/low data for true ATR, here we approximate with returns
        features['atr_20d'] = returns.rolling(20).std()  # Approximation
        
        return features
    
    def compute_oscillators(
        self,
        bars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute RSI, MACD, Stochastic oscillators."""
        features = pd.DataFrame(index=bars_df.index)
        
        if 'close' not in bars_df.columns:
            return features
        
        prices = bars_df['close']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Stochastic (requires high/low)
        if 'high' in bars_df.columns and 'low' in bars_df.columns:
            high_14 = bars_df['high'].rolling(14).max()
            low_14 = bars_df['low'].rolling(14).min()
            features['stoch_k'] = 100 * ((prices - low_14) / (high_14 - low_14))
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        return features
    
    def compute_volume_features(
        self,
        bars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute volume-based features."""
        features = pd.DataFrame(index=bars_df.index)
        
        if 'volume' not in bars_df.columns:
            return features
        
        volume = bars_df['volume']
        
        # Volume z-score vs history
        volume_mean = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        features['volume_zscore'] = (volume - volume_mean) / (volume_std + 1e-10)
        
        # Volume ratios
        features['volume_to_ma20'] = volume / volume.rolling(20).mean()
        features['volume_to_ma60'] = volume / volume.rolling(60).mean()
        
        return features
    
    def build_features(
        self,
        bars_df: pd.DataFrame,
        return_windows: List[int] = [1, 5, 20, 60, 120],
        ma_windows: List[int] = [5, 10, 20, 50, 100, 200],
        ema_windows: List[int] = [10, 20, 50],
        momentum_windows: List[int] = [20, 60, 120]
    ) -> pd.DataFrame:
        """
        Build all technical features for a bars DataFrame.
        
        Args:
            bars_df: DataFrame with columns: date, asset_id, open, high, low, close, adj_close, volume
        
        Returns:
            DataFrame with date and asset_id index and feature columns
        """
        if len(bars_df) == 0:
            return pd.DataFrame()
        
        bars_df = bars_df.set_index(['date', 'asset_id']).sort_index()
        
        all_features = []
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            asset_bars = asset_bars.sort_index()
            
            # Reset index to get date as a column
            asset_bars_reset = asset_bars.reset_index()
            prices = asset_bars_reset.set_index('date')['adj_close']
            
            # Compute all feature groups
            returns_features = self.compute_returns(prices, return_windows)
            ma_features = self.compute_moving_averages(prices, ma_windows, ema_windows)
            momentum_features = self.compute_momentum(prices, momentum_windows)
            volatility_features = self.compute_volatility(prices)
            oscillator_features = self.compute_oscillators(asset_bars_reset)
            volume_features = self.compute_volume_features(asset_bars_reset)
            
            # Combine all features
            asset_features = pd.concat([
                returns_features,
                ma_features,
                momentum_features,
                volatility_features,
                oscillator_features,
                volume_features
            ], axis=1)
            
            # Reset index to get date as column
            asset_features = asset_features.reset_index()
            asset_features['asset_id'] = asset_id
            
            all_features.append(asset_features)
        
        if len(all_features) == 0:
            return pd.DataFrame()
        
        result = pd.concat(all_features, ignore_index=True)
        # Ensure date and asset_id columns exist
        if 'date' not in result.columns:
            # Try to get date from index if it's a MultiIndex
            if isinstance(result.index, pd.MultiIndex) and 'date' in result.index.names:
                result = result.reset_index()
            else:
                # Create a dummy date column (shouldn't happen, but handle gracefully)
                result['date'] = None
        
        return result

