"""Portfolio construction strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Set
from datetime import date
from portfolio.risk import RiskManager, ExposureManager


class LongTopKStrategy:
    """Long-only strategy: pick top K assets by score."""
    
    def __init__(
        self,
        k: int = 20,
        min_score_threshold: float = 0.0,
        min_confidence: float = 0.7,
        risk_manager: Optional[RiskManager] = None,
        exposure_manager: Optional[ExposureManager] = None
    ):
        self.k = k
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence
        self.risk_manager = risk_manager or RiskManager()
        self.exposure_manager = exposure_manager  # Optional, can be None
    
    def compute_weights(
        self,
        scores_df: pd.DataFrame,
        current_positions: Optional[Dict[int, float]] = None,
        as_of_date: date = None,
        exposure_scale: Optional[float] = None,
        current_regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute target weights.
        
        Args:
            scores_df: DataFrame with columns: asset_id, score, confidence (optional)
            current_positions: Current portfolio weights {asset_id: weight}
            as_of_date: Date for risk checks
            exposure_scale: Optional explicit exposure scale factor (0.0 to 1.0).
                           If provided, overrides ExposureManager.
            current_regime: Current regime descriptor for sector tilts
        
        Returns:
            DataFrame with columns: asset_id, weight
        """
        # Filter by score threshold
        filtered = scores_df[scores_df['score'] >= self.min_score_threshold].copy()
        
        # Filter by confidence if available
        if 'confidence' in filtered.columns:
            filtered = filtered[filtered['confidence'] >= self.min_confidence]
        
        # Sort by score and take top K
        filtered = filtered.sort_values('score', ascending=False).head(self.k)
        
        if len(filtered) == 0:
            return pd.DataFrame(columns=['asset_id', 'weight'])
        
        # Compute weights proportional to score
        filtered['weight_raw'] = np.maximum(filtered['score'], 0)
        total_score = filtered['weight_raw'].sum()
        
        if total_score > 0:
            filtered['weight'] = filtered['weight_raw'] / total_score
        else:
            filtered['weight'] = 1.0 / len(filtered)  # Equal weight
        
        # Determine regime for sector tilts
        regime = current_regime
        if regime is None and self.exposure_manager is not None:
            regime = self.exposure_manager.current_regime
        
        # Apply risk constraints (including sector tilts)
        weights_dict = dict(zip(filtered['asset_id'], filtered['weight']))
        weights_dict = self.risk_manager.apply_constraints(
            weights_dict, as_of_date, current_regime=regime
        )
        
        # Normalize to sum to 1 (before exposure scaling)
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
        
        # Apply exposure scaling (regime + vol + drawdown)
        if exposure_scale is not None:
            # Explicit scale provided
            if exposure_scale < 1.0:
                weights_dict = {k: v * exposure_scale for k, v in weights_dict.items()}
        elif self.exposure_manager is not None:
            # Use exposure manager's combined scale
            scale = self.exposure_manager.get_combined_scale()
            if scale < 1.0:
                weights_dict = {k: v * scale for k, v in weights_dict.items()}
        
        result = pd.DataFrame({
            'asset_id': list(weights_dict.keys()),
            'weight': list(weights_dict.values())
        })
        
        return result


class LongShortDecilesStrategy:
    """
    Long top decile, short bottom decile.
    
    Supports regime-aware operation:
    - Can reduce gross exposure in bear/high-vol regimes
    - Can optionally disable shorts in crisis regimes
    """
    
    def __init__(
        self,
        long_decile: int = 10,
        short_decile: int = 1,
        market_neutral: bool = True,
        risk_manager: Optional[RiskManager] = None,
        exposure_manager: Optional[ExposureManager] = None,
        regime_aware: bool = True,
        disable_shorts_in_crisis: bool = False
    ):
        self.long_decile = long_decile
        self.short_decile = short_decile
        self.market_neutral = market_neutral
        self.risk_manager = risk_manager or RiskManager()
        self.exposure_manager = exposure_manager
        self.regime_aware = regime_aware
        self.disable_shorts_in_crisis = disable_shorts_in_crisis
    
    def compute_weights(
        self,
        scores_df: pd.DataFrame,
        current_positions: Optional[Dict[int, float]] = None,
        as_of_date: date = None,
        exposure_scale: Optional[float] = None,
        current_regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute long/short weights.
        
        Args:
            scores_df: DataFrame with columns: asset_id, score
            current_positions: Current portfolio weights
            as_of_date: Date for risk checks
            exposure_scale: Optional explicit exposure scale factor
            current_regime: Current regime descriptor for crisis detection
        
        Returns:
            DataFrame with columns: asset_id, weight
        """
        # Compute deciles
        scores_df = scores_df.copy()
        scores_df['decile'] = pd.qcut(
            scores_df['score'],
            q=10,
            labels=False,
            duplicates='drop'
        ) + 1
        
        # Long top decile
        long_assets = scores_df[scores_df['decile'] == self.long_decile]
        # Short bottom decile
        short_assets = scores_df[scores_df['decile'] == self.short_decile]
        
        # Check if we should disable shorts in crisis
        disable_shorts = False
        if self.disable_shorts_in_crisis and current_regime == 'bear_high_vol':
            disable_shorts = True
        
        weights_dict = {}
        
        # Long positions
        if len(long_assets) > 0:
            long_weight = 0.5 / len(long_assets) if self.market_neutral else 1.0 / len(long_assets)
            for asset_id in long_assets['asset_id']:
                weights_dict[asset_id] = long_weight
        
        # Short positions (unless disabled)
        if len(short_assets) > 0 and not disable_shorts:
            short_weight = -0.5 / len(short_assets) if self.market_neutral else 0.0
            for asset_id in short_assets['asset_id']:
                weights_dict[asset_id] = short_weight
        
        # Determine regime for sector tilts
        regime = current_regime
        if regime is None and self.exposure_manager is not None:
            regime = self.exposure_manager.current_regime
        
        # Apply risk constraints (including sector tilts)
        weights_dict = self.risk_manager.apply_constraints(
            weights_dict, as_of_date, current_regime=regime
        )
        
        # Apply exposure scaling
        if exposure_scale is not None:
            if exposure_scale < 1.0:
                weights_dict = {k: v * exposure_scale for k, v in weights_dict.items()}
        elif self.exposure_manager is not None and self.regime_aware:
            scale = self.exposure_manager.get_combined_scale()
            if scale < 1.0:
                weights_dict = {k: v * scale for k, v in weights_dict.items()}
        
        result = pd.DataFrame({
            'asset_id': list(weights_dict.keys()),
            'weight': list(weights_dict.values())
        })
        
        return result


class EqualWeightUniverseStrategy:
    """Equal-weight all assets in universe."""
    
    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.risk_manager = risk_manager or RiskManager()
    
    def compute_weights(
        self,
        universe: Set[int],
        current_positions: Optional[Dict[int, float]] = None,
        as_of_date: date = None
    ) -> pd.DataFrame:
        """Compute equal weights."""
        num_assets = len(universe)
        if num_assets == 0:
            return pd.DataFrame(columns=['asset_id', 'weight'])
        
        weight_per_asset = 1.0 / num_assets
        
        weights_dict = {asset_id: weight_per_asset for asset_id in universe}
        
        # Apply risk constraints
        weights_dict = self.risk_manager.apply_constraints(weights_dict, as_of_date)
        
        result = pd.DataFrame({
            'asset_id': list(weights_dict.keys()),
            'weight': list(weights_dict.values())
        })
        
        return result

