"""Regime label generation using clustering."""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import date
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI


class RegimeLabelGenerator:
    """Generates regime labels using unsupervised clustering."""
    
    def __init__(self, storage: StorageBackend, n_regimes: int = 4):
        self.storage = storage
        self.api = AsOfQueryAPI(storage)
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.model = None
    
    def build_regime_features(
        self,
        start_date: date,
        end_date: date,
        benchmark_symbol: str = "SPY"
    ) -> pd.DataFrame:
        """
        Build feature matrix for regime detection.
        
        Features:
        - Index returns (1d, 5d, 20d)
        - Realized volatility (20d rolling)
        - Drawdown from recent high
        - VIX proxy (if available) or volatility of volatility
        
        Returns:
            DataFrame with date index and feature columns
        """
        # Get benchmark bars
        benchmark_df = self.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{benchmark_symbol}'")
        if len(benchmark_df) == 0:
            raise ValueError(f"Benchmark symbol {benchmark_symbol} not found")
        benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
        
        bars_df = self.api.get_bars_asof(end_date)
        benchmark_bars = bars_df[bars_df['asset_id'] == benchmark_asset_id].copy()
        benchmark_bars = benchmark_bars.set_index('date').sort_index()
        
        if len(benchmark_bars) == 0:
            return pd.DataFrame()
        
        prices = benchmark_bars['adj_close']
        returns = np.log(prices / prices.shift(1))
        
        # Compute features
        features = pd.DataFrame(index=benchmark_bars.index)
        
        # Returns over different horizons
        features['return_1d'] = returns
        features['return_5d'] = np.log(prices / prices.shift(5))
        features['return_20d'] = np.log(prices / prices.shift(20))
        
        # Realized volatility (20-day rolling std of returns)
        features['realized_vol_20d'] = returns.rolling(20).std()
        
        # Drawdown from recent high (20-day lookback)
        rolling_high = prices.rolling(20).max()
        features['drawdown'] = (prices - rolling_high) / rolling_high
        
        # Volatility of volatility (proxy for VIX-like measure)
        vol_20d = returns.rolling(20).std()
        features['vol_of_vol'] = vol_20d.rolling(20).std()
        
        # Filter to date range
        features = features[
            (features.index >= pd.Timestamp(start_date)) &
            (features.index <= pd.Timestamp(end_date))
        ]
        
        # Drop rows with NaN
        features = features.dropna()
        
        return features
    
    def fit_regimes(
        self,
        start_date: date,
        end_date: date,
        method: str = "kmeans"
    ) -> pd.DataFrame:
        """
        Fit regime model and assign regime labels to dates.
        
        Args:
            start_date: Start date for training
            end_date: End date for training
            method: "kmeans" or "gmm"
        
        Returns:
            DataFrame with columns: date, regime_id, regime_descriptor
        """
        # Build features
        features_df = self.build_regime_features(start_date, end_date)
        
        if len(features_df) == 0:
            return pd.DataFrame(columns=['date', 'regime_id', 'regime_descriptor'])
        
        # Scale features
        feature_matrix = self.scaler.fit_transform(features_df.values)
        
        # Fit clustering model
        if method == "kmeans":
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        elif method == "gmm":
            self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        regime_ids = self.model.fit_predict(feature_matrix)
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'date': features_df.index.date,
            'regime_id': regime_ids
        })
        
        # Add regime descriptors based on feature centroids/means
        labels_df['regime_descriptor'] = labels_df['regime_id'].apply(
            lambda rid: f"regime_{rid}"
        )
        
        # Optionally, label regimes based on average return/volatility
        regime_stats = []
        for rid in range(self.n_regimes):
            regime_features = features_df[regime_ids == rid]
            avg_return = regime_features['return_20d'].mean()
            avg_vol = regime_features['realized_vol_20d'].mean()
            
            if avg_return > 0.05 and avg_vol < 0.15:
                descriptor = "bull_low_vol"
            elif avg_return > 0.05 and avg_vol >= 0.15:
                descriptor = "bull_high_vol"
            elif avg_return <= 0.05 and avg_vol < 0.15:
                descriptor = "bear_low_vol"
            else:
                descriptor = "bear_high_vol"
            
            regime_stats.append({'regime_id': rid, 'descriptor': descriptor})
        
        regime_map = {r['regime_id']: r['descriptor'] for r in regime_stats}
        labels_df['regime_descriptor'] = labels_df['regime_id'].map(regime_map)
        
        return labels_df
    
    def predict_regime(self, as_of_date: date) -> int:
        """Predict regime for a specific date using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_regimes first.")
        
        # Build features for this date
        features_df = self.build_regime_features(as_of_date, as_of_date)
        
        if len(features_df) == 0:
            return -1
        
        # Scale and predict
        feature_matrix = self.scaler.transform(features_df.values)
        regime_id = self.model.predict(feature_matrix)[0]
        
        return regime_id
    
    def save_regimes(self, regimes_df: pd.DataFrame, table_name: str = "regimes"):
        """Save regime labels to storage."""
        self.storage.save_parquet(regimes_df, table_name)
        # Also save to DuckDB
        try:
            self.storage.insert_dataframe(table_name, regimes_df, if_exists="append")
        except Exception:
            # Create table if it doesn't exist
            self.storage.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date DATE NOT NULL PRIMARY KEY,
                    regime_id INTEGER NOT NULL,
                    regime_descriptor VARCHAR
                )
            """)
            self.storage.insert_dataframe(table_name, regimes_df, if_exists="append")

