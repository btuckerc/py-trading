"""Unified feature pipeline."""

import pandas as pd
import numpy as np
from typing import List, Optional, Set, Dict, Tuple
from datetime import date
from loguru import logger
from data.asof_api import AsOfQueryAPI
from data.clock import SimulationClock
from features.technical import TechnicalFeatureBuilder
from features.cross_sectional import CrossSectionalFeatureBuilder
from features.fundamentals import FundamentalsFeatureBuilder
from features.calendar import CalendarFeatureBuilder
from features.sentiment import SentimentFeatureBuilder
from features.options import OptionsFeatureBuilder
from features.microstructure import MicrostructureFeatureBuilder
from features.scaling import ScalerManager, FeatureScaler
from configs.loader import get_config


class RegimeFeatureBuilder:
    """
    Builds regime features from pre-computed regime labels.
    
    Features include:
    - regime_id: Numeric cluster ID
    - One-hot encoded regime descriptors
    - Raw regime indicators (vol, drawdown) for VIX-style sizing
    """
    
    def __init__(self, api: AsOfQueryAPI, include_raw_features: bool = True):
        self.api = api
        self.include_raw_features = include_raw_features
    
    def build_features(self, as_of_date: date, universe: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Build regime features for all assets on a date.
        
        Queries the regimes table for the current regime and returns it as a feature
        for each asset in the universe.
        
        Returns:
            DataFrame with columns: asset_id, regime_id, regime_bull_low_vol, etc.
        """
        try:
            # Query the regimes table for this date
            result = self.api.storage.conn.execute("""
                SELECT date, regime_id, regime_descriptor
                FROM regimes
                WHERE date = ?
            """, [as_of_date]).df()
            
            if len(result) == 0:
                # No regime for this date, try to find most recent
                result = self.api.storage.conn.execute("""
                    SELECT date, regime_id, regime_descriptor
                    FROM regimes
                    WHERE date <= ?
                    ORDER BY date DESC
                    LIMIT 1
                """, [as_of_date]).df()
                
                if len(result) == 0:
                    return pd.DataFrame(columns=['asset_id'])
            
            regime_id = result['regime_id'].iloc[0]
            regime_descriptor = result['regime_descriptor'].iloc[0]
            
            # Create features for each asset
            if universe is None:
                universe = self.api.get_universe_at_date(as_of_date)
            
            # Get raw regime features if enabled (for VIX-style sizing)
            raw_features = {}
            if self.include_raw_features:
                raw_features = self._get_raw_regime_features(as_of_date)
            
            records = []
            for asset_id in universe:
                record = {
                    'asset_id': asset_id,
                    'regime_id': regime_id,
                    # One-hot encode regime descriptors
                    'regime_bull_low_vol': 1 if regime_descriptor == 'bull_low_vol' else 0,
                    'regime_bull_high_vol': 1 if regime_descriptor == 'bull_high_vol' else 0,
                    'regime_bear_low_vol': 1 if regime_descriptor == 'bear_low_vol' else 0,
                    'regime_bear_high_vol': 1 if regime_descriptor == 'bear_high_vol' else 0,
                }
                
                # Add raw features (same for all assets - market-level)
                if raw_features:
                    record['market_realized_vol_20d'] = raw_features.get('realized_vol_20d', 0.15)
                    record['market_drawdown'] = raw_features.get('drawdown', 0.0)
                    record['market_vol_of_vol'] = raw_features.get('vol_of_vol', 0.0)
                
                records.append(record)
            
            return pd.DataFrame(records)
        except Exception as e:
            # Table may not exist or other error
            logger.debug(f"RegimeFeatureBuilder error: {e}")
            return pd.DataFrame(columns=['asset_id'])
    
    def _get_raw_regime_features(self, as_of_date: date) -> Dict:
        """Get raw market-level regime features for VIX-style sizing."""
        try:
            # Get SPY bars for volatility calculation
            benchmark_df = self.api.storage.query("SELECT asset_id FROM assets WHERE symbol = 'SPY'")
            if len(benchmark_df) == 0:
                return {}
            
            benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
            
            # Get recent bars
            bars_df = self.api.get_bars_asof(as_of_date, lookback_days=60, universe={benchmark_asset_id})
            if len(bars_df) == 0:
                return {}
            
            bars_df = bars_df.sort_values('date')
            prices = bars_df['adj_close'].values
            
            if len(prices) < 21:
                return {}
            
            # Calculate features
            returns = np.log(prices[1:] / prices[:-1])
            
            # Realized vol (annualized)
            realized_vol_20d = np.std(returns[-20:]) * np.sqrt(252)
            
            # Drawdown
            rolling_max = np.maximum.accumulate(prices)
            drawdowns = (prices - rolling_max) / rolling_max
            current_drawdown = drawdowns[-1]
            
            # Vol of vol
            if len(returns) >= 40:
                rolling_vols = pd.Series(returns).rolling(20).std().dropna().values * np.sqrt(252)
                vol_of_vol = np.std(rolling_vols) if len(rolling_vols) > 1 else 0.0
            else:
                vol_of_vol = 0.0
            
            return {
                'realized_vol_20d': float(realized_vol_20d),
                'drawdown': float(current_drawdown),
                'vol_of_vol': float(vol_of_vol),
            }
        except Exception as e:
            logger.debug(f"Error getting raw regime features: {e}")
            return {}
    
    def get_current_regime(self, as_of_date: date) -> Tuple[int, str]:
        """
        Get the current regime ID and descriptor for a date.
        
        Returns:
            Tuple of (regime_id, regime_descriptor)
        """
        try:
            result = self.api.storage.conn.execute("""
                SELECT regime_id, regime_descriptor
                FROM regimes
                WHERE date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, [as_of_date]).df()
            
            if len(result) == 0:
                return -1, "unknown"
            
            return int(result['regime_id'].iloc[0]), str(result['regime_descriptor'].iloc[0])
        except Exception:
            return -1, "unknown"


class FeaturePipeline:
    """
    Unified feature pipeline that builds features as-of a specific date.
    
    Ensures point-in-time correctness by querying data through the as-of API.
    """
    
    def __init__(self, api: AsOfQueryAPI, config: Optional[Dict] = None):
        self.api = api
        self.config = config or get_config().features
        
        # Initialize feature builders
        self.technical_builder = TechnicalFeatureBuilder(api) if self.config.get('technical', {}).get('enabled', True) else None
        self.cross_sectional_builder = CrossSectionalFeatureBuilder(api) if self.config.get('cross_sectional', {}).get('enabled', True) else None
        self.fundamentals_builder = FundamentalsFeatureBuilder(api) if self.config.get('fundamentals', {}).get('enabled', True) else None
        self.calendar_builder = CalendarFeatureBuilder() if self.config.get('calendar', {}).get('enabled', True) else None
        self.sentiment_builder = SentimentFeatureBuilder(api) if self.config.get('sentiment', {}).get('enabled', False) else None
        self.options_builder = OptionsFeatureBuilder(api) if self.config.get('options', {}).get('enabled', False) else None
        self.microstructure_builder = MicrostructureFeatureBuilder(api) if self.config.get('microstructure', {}).get('enabled', False) else None
        self.regime_builder = RegimeFeatureBuilder(api) if self.config.get('regimes', {}).get('enabled', False) else None
        
        # Scaler manager
        self.scaler_manager = ScalerManager()
    
    def build_features_cross_sectional(
        self,
        as_of_date: date,
        universe: Optional[Set[int]] = None,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Build cross-sectional features (one row per asset).
        
        Returns:
            DataFrame with shape (num_assets, num_features)
        """
        # Get universe if not provided
        if universe is None:
            universe = self.api.get_universe_at_date(as_of_date)
        
        if len(universe) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        # Get bars with lookback
        bars_df = self.api.get_bars_asof(as_of_date, lookback_days=lookback_days, universe=universe)
        
        if len(bars_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        all_features = []
        
        # Technical features
        if self.technical_builder:
            technical_features = self.technical_builder.build_features(bars_df)
            if len(technical_features) > 0:
                # Get latest features per asset
                technical_features = technical_features.groupby('asset_id').last().reset_index()
                technical_features['date'] = as_of_date
                all_features.append(technical_features)
        
        # Cross-sectional features
        if self.cross_sectional_builder:
            cross_features = self.cross_sectional_builder.build_features(bars_df, as_of_date)
            if len(cross_features) > 0:
                all_features.append(cross_features)
        
        # Fundamentals
        if self.fundamentals_builder:
            fund_features = self.fundamentals_builder.build_features(as_of_date, universe)
            if len(fund_features) > 0:
                all_features.append(fund_features)
        
        # Calendar features
        if self.calendar_builder:
            dates = pd.Series([as_of_date] * len(universe))
            calendar_features = self.calendar_builder.build_features(dates)
            calendar_features['asset_id'] = list(universe)
            all_features.append(calendar_features)
        
        # Sentiment
        if self.sentiment_builder:
            sentiment_features = self.sentiment_builder.build_features(as_of_date, universe=universe)
            if len(sentiment_features) > 0:
                all_features.append(sentiment_features)
        
        # Options (placeholder)
        if self.options_builder:
            options_features = self.options_builder.build_features(as_of_date, universe)
            if len(options_features) > 0:
                all_features.append(options_features)
        
        # Microstructure (placeholder)
        if self.microstructure_builder:
            micro_features = self.microstructure_builder.build_features(as_of_date, universe)
            if len(micro_features) > 0:
                all_features.append(micro_features)
        
        # Regime features
        if self.regime_builder:
            regime_features = self.regime_builder.build_features(as_of_date, universe)
            if len(regime_features) > 0:
                all_features.append(regime_features)
        
        # Merge all features
        if len(all_features) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        # Ensure all feature DataFrames have date and asset_id columns
        for i, features in enumerate(all_features):
            if 'date' not in features.columns:
                features['date'] = as_of_date
            else:
                # Normalize date column to date type (not datetime)
                features['date'] = pd.to_datetime(features['date']).dt.date
            if 'asset_id' not in features.columns:
                # Skip this feature set if it doesn't have asset_id
                all_features[i] = None
        
        # Filter out None entries
        all_features = [f for f in all_features if f is not None and len(f) > 0]
        
        if len(all_features) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        result = all_features[0]
        # Ensure result date is date type
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date']).dt.date
        
        for features in all_features[1:]:
            # Ensure both have date and asset_id for merge
            if 'date' in features.columns and 'asset_id' in features.columns:
                # Normalize date type
                features['date'] = pd.to_datetime(features['date']).dt.date
                result = result.merge(features, on=['asset_id', 'date'], how='outer')
        
        # Fill NaN with 0 for numeric columns (or use forward fill for time series)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        return result
    
    def build_features_sequence(
        self,
        as_of_date: date,
        sequence_length: int,
        universe: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Build sequence features (L days of history per asset).
        
        Returns:
            Array with shape (num_assets, sequence_length, num_features)
        """
        # Get lookback dates
        clock = SimulationClock(
            pd.Timestamp(as_of_date) - pd.Timedelta(days=sequence_length * 2),
            as_of_date
        )
        lookback_dates = clock.get_lookback_dates(sequence_length)
        
        if len(lookback_dates) < sequence_length:
            return np.array([])
        
        # Build features for each date in sequence
        sequence_features = []
        for seq_date in lookback_dates[-sequence_length:]:
            date_features = self.build_features_cross_sectional(seq_date, universe)
            sequence_features.append(date_features)
        
        if len(sequence_features) == 0:
            return np.array([])
        
        # Stack into sequence
        # Align asset_ids across all dates
        all_asset_ids = set()
        for features in sequence_features:
            all_asset_ids.update(features['asset_id'].unique())
        
        all_asset_ids = sorted(all_asset_ids)
        
        # Extract feature columns (exclude asset_id, date)
        feature_cols = [c for c in sequence_features[0].columns if c not in ['asset_id', 'date']]
        
        # Build sequence array
        num_assets = len(all_asset_ids)
        num_features = len(feature_cols)
        
        sequence_array = np.zeros((num_assets, sequence_length, num_features))
        
        for t, features in enumerate(sequence_features):
            features_indexed = features.set_index('asset_id')
            for i, asset_id in enumerate(all_asset_ids):
                if asset_id in features_indexed.index:
                    sequence_array[i, t, :] = features_indexed.loc[asset_id, feature_cols].values
        
        return sequence_array
    
    def scale_features(
        self,
        features_df: pd.DataFrame,
        window_name: str,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Scale features using scaler for a training window.
        
        Args:
            features_df: DataFrame with features
            window_name: Name of training window (e.g., "split_1")
            fit: If True, fit scaler on this data; if False, use existing scaler
        
        Returns:
            Scaled features DataFrame
        """
        scaler = self.scaler_manager.get_scaler(window_name)
        
        if fit:
            scaler.fit(features_df)
            self.scaler_manager.save_scaler(window_name, scaler)
        
        return scaler.transform(features_df)

