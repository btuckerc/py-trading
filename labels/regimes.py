"""Regime label generation using clustering."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import date
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from loguru import logger
import pickle
from pathlib import Path


# Standard regime descriptors used throughout the system
REGIME_DESCRIPTORS = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]


class RegimeLabelGenerator:
    """
    Generates regime labels using unsupervised clustering.
    
    Regime Descriptors:
    - bull_low_vol: Positive returns with low volatility (best regime)
    - bull_high_vol: Positive returns with high volatility (recovery/momentum)
    - bear_low_vol: Negative/flat returns with low volatility (grinding bear)
    - bear_high_vol: Negative returns with high volatility (crisis)
    
    These descriptors are used throughout the system for:
    - Feature engineering (one-hot encoding)
    - Exposure scaling (regime_policy in config)
    - Sector rotation (sector_policy in config)
    """
    
    def __init__(self, storage: StorageBackend, n_regimes: int = 4):
        self.storage = storage
        self.api = AsOfQueryAPI(storage)
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.model = None
        self.regime_stats: Dict[int, Dict] = {}  # Stores stats for each regime_id
    
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
        - Realized volatility (20d rolling, annualized)
        - Drawdown from recent high
        - VIX proxy (volatility of volatility)
        
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
        
        # Realized volatility (20-day rolling std of returns, annualized)
        features['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Drawdown from recent high (20-day lookback)
        rolling_high = prices.rolling(20).max()
        features['drawdown'] = (prices - rolling_high) / rolling_high
        
        # Volatility of volatility (proxy for VIX-like measure)
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        features['vol_of_vol'] = vol_20d.rolling(20).std()
        
        # Filter to date range
        features = features[
            (features.index >= pd.Timestamp(start_date)) &
            (features.index <= pd.Timestamp(end_date))
        ]
        
        # Drop rows with NaN
        features = features.dropna()
        
        return features
    
    def _assign_regime_descriptors(
        self,
        features_df: pd.DataFrame,
        regime_ids: np.ndarray
    ) -> Dict[int, str]:
        """
        Assign descriptive labels to regime IDs based on cluster characteristics.
        
        Uses median return and volatility to classify each cluster into one of:
        - bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
        
        Returns:
            Dictionary mapping regime_id -> descriptor
        """
        regime_stats = []
        
        for rid in range(self.n_regimes):
            mask = regime_ids == rid
            if not mask.any():
                continue
                
            regime_features = features_df[mask]
            
            # Use median for robustness against outliers
            median_return_20d = regime_features['return_20d'].median()
            median_vol = regime_features['realized_vol_20d'].median()
            mean_drawdown = regime_features['drawdown'].mean()
            
            regime_stats.append({
                'regime_id': rid,
                'median_return_20d': median_return_20d,
                'median_vol': median_vol,
                'mean_drawdown': mean_drawdown,
                'count': mask.sum()
            })
        
        # Sort regimes by return (descending) then by volatility (ascending)
        regime_stats.sort(key=lambda x: (-x['median_return_20d'], x['median_vol']))
        
        # Compute thresholds based on all regime stats
        all_returns = [r['median_return_20d'] for r in regime_stats]
        all_vols = [r['median_vol'] for r in regime_stats]
        
        # Use median of medians as threshold (more robust)
        return_threshold = np.median(all_returns)
        vol_threshold = np.median(all_vols)
        
        logger.info(f"Regime classification thresholds: return={return_threshold:.4f}, vol={vol_threshold:.4f}")
        
        # Assign descriptors based on quadrant
        regime_map = {}
        used_descriptors = set()
        
        for stats in regime_stats:
            rid = stats['regime_id']
            ret = stats['median_return_20d']
            vol = stats['median_vol']
            
            # Classify into quadrant
            is_bull = ret > return_threshold
            is_low_vol = vol < vol_threshold
            
            if is_bull and is_low_vol:
                descriptor = "bull_low_vol"
            elif is_bull and not is_low_vol:
                descriptor = "bull_high_vol"
            elif not is_bull and is_low_vol:
                descriptor = "bear_low_vol"
            else:
                descriptor = "bear_high_vol"
            
            # Handle duplicate descriptors by adjusting based on secondary characteristics
            if descriptor in used_descriptors:
                # Find alternative based on drawdown severity
                if "bear" in descriptor and stats['mean_drawdown'] < -0.10:
                    descriptor = "bear_high_vol"
                elif "bull" in descriptor and stats['mean_drawdown'] > -0.03:
                    descriptor = "bull_low_vol"
            
            regime_map[rid] = descriptor
            used_descriptors.add(descriptor)
            
            # Store stats for later use
            self.regime_stats[rid] = {
                'descriptor': descriptor,
                'median_return_20d': ret,
                'median_vol': vol,
                'mean_drawdown': stats['mean_drawdown'],
                'count': stats['count']
            }
            
            logger.info(
                f"Regime {rid} -> {descriptor}: "
                f"return={ret:.4f}, vol={vol:.4f}, dd={stats['mean_drawdown']:.4f}, "
                f"n={stats['count']}"
            )
        
        return regime_map
    
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
        
        logger.info(f"Built regime features for {len(features_df)} dates")
        
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
        
        # Assign descriptive labels based on cluster characteristics
        regime_map = self._assign_regime_descriptors(features_df, regime_ids)
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'date': features_df.index.date,
            'regime_id': regime_ids
        })
        
        # Map regime_id to descriptor
        labels_df['regime_descriptor'] = labels_df['regime_id'].map(regime_map)
        
        return labels_df
    
    def predict_regime(self, as_of_date: date) -> Tuple[int, str]:
        """
        Predict regime for a specific date using fitted model.
        
        Returns:
            Tuple of (regime_id, regime_descriptor)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_regimes first.")
        
        # Build features for this date (need lookback for rolling calculations)
        lookback_start = pd.Timestamp(as_of_date) - pd.Timedelta(days=60)
        features_df = self.build_regime_features(lookback_start.date(), as_of_date)
        
        if len(features_df) == 0:
            return -1, "unknown"
        
        # Get the most recent feature row
        latest_features = features_df.iloc[[-1]]
        
        # Scale and predict
        feature_matrix = self.scaler.transform(latest_features.values)
        regime_id = int(self.model.predict(feature_matrix)[0])
        
        # Get descriptor from stored stats
        descriptor = self.regime_stats.get(regime_id, {}).get('descriptor', f'regime_{regime_id}')
        
        return regime_id, descriptor
    
    def get_regime_exposure_factor(self, regime_descriptor: str, exposure_policy: Dict[str, float]) -> float:
        """
        Get the exposure scaling factor for a given regime.
        
        Args:
            regime_descriptor: One of bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
            exposure_policy: Dict mapping descriptor -> exposure factor
        
        Returns:
            Exposure factor (0.0 to 1.0)
        """
        return exposure_policy.get(regime_descriptor, 1.0)
    
    def get_current_regime_features(self, as_of_date: date) -> Dict[str, float]:
        """
        Get the raw regime features for a specific date.
        
        Useful for VIX-style position sizing based on realized volatility.
        
        Returns:
            Dict with feature values (realized_vol_20d, vol_of_vol, drawdown, etc.)
        """
        lookback_start = pd.Timestamp(as_of_date) - pd.Timedelta(days=60)
        features_df = self.build_regime_features(lookback_start.date(), as_of_date)
        
        if len(features_df) == 0:
            return {}
        
        latest = features_df.iloc[-1]
        return {
            'return_1d': float(latest['return_1d']),
            'return_5d': float(latest['return_5d']),
            'return_20d': float(latest['return_20d']),
            'realized_vol_20d': float(latest['realized_vol_20d']),
            'drawdown': float(latest['drawdown']),
            'vol_of_vol': float(latest['vol_of_vol']),
        }
    
    def save_regimes(self, regimes_df: pd.DataFrame, table_name: str = "regimes"):
        """Save regime labels to storage (both Parquet and DuckDB)."""
        self.storage.save_parquet(regimes_df, table_name)
        
        # Also save to DuckDB with proper table creation
        try:
            # Try to create table first
            self.storage.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date DATE NOT NULL PRIMARY KEY,
                    regime_id INTEGER NOT NULL,
                    regime_descriptor VARCHAR
                )
            """)
            
            # Clear existing data and insert new
            self.storage.conn.execute(f"DELETE FROM {table_name}")
            self.storage.insert_dataframe(table_name, regimes_df, if_exists="append")
            logger.info(f"Saved {len(regimes_df)} regime labels to {table_name}")
        except Exception as e:
            logger.error(f"Error saving regimes to DuckDB: {e}")
            raise
    
    def save_model(self, path: str):
        """
        Save the fitted regime model and scaler for later use.
        
        Args:
            path: Path to save the model pickle file
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_regimes first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'regime_stats': self.regime_stats,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved regime model to {path}")
    
    def load_model(self, path: str):
        """
        Load a previously fitted regime model.
        
        Args:
            path: Path to the model pickle file
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.n_regimes = model_data['n_regimes']
        self.regime_stats = model_data['regime_stats']
        logger.info(f"Loaded regime model from {path}")

