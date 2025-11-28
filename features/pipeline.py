"""Unified feature pipeline."""

import pandas as pd
import numpy as np
from typing import List, Optional, Set, Dict
import pandas as pd
from datetime import date
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
        
        # Merge all features
        if len(all_features) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        result = all_features[0]
        for features in all_features[1:]:
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

