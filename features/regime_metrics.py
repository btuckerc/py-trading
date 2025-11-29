"""Public regime metrics service.

This module provides a clean public API for accessing regime-related metrics,
avoiding the need to reach into private methods of RegimeFeatureBuilder.

Usage:
    from features.regime_metrics import RegimeMetricsService
    
    service = RegimeMetricsService(api)
    
    # Get current regime
    regime_id, descriptor = service.get_current_regime(trading_date)
    
    # Get volatility for position sizing
    vol = service.get_current_volatility(trading_date)
    
    # Get full regime metrics
    metrics = service.get_regime_metrics(trading_date)
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RegimeMetrics:
    """
    Container for regime-related metrics.
    
    Provides all the information needed for regime-aware trading:
    - Current regime identification
    - Volatility metrics for position sizing
    - Market stress indicators
    """
    
    # Regime identification
    regime_id: int
    regime_descriptor: str
    
    # Volatility metrics (annualized)
    realized_vol_20d: float
    realized_vol_60d: Optional[float] = None
    vol_of_vol: Optional[float] = None
    
    # Market stress indicators
    current_drawdown: float = 0.0
    max_drawdown_60d: Optional[float] = None
    
    # VIX-like implied vol (if available)
    implied_vol: Optional[float] = None
    
    # Regime confidence/stability
    regime_age_days: int = 0
    regime_confidence: float = 1.0
    
    def is_high_vol(self, threshold: float = 0.25) -> bool:
        """Check if current volatility is above threshold."""
        return self.realized_vol_20d > threshold
    
    def is_in_drawdown(self, threshold: float = 0.10) -> bool:
        """Check if in significant drawdown."""
        return abs(self.current_drawdown) > threshold
    
    def get_vol_scale_factor(
        self,
        target_vol: float = 0.15,
        vol_floor: float = 0.10,
        vol_ceiling: float = 0.60,
    ) -> float:
        """
        Compute volatility-based position scale factor.
        
        Args:
            target_vol: Target portfolio volatility
            vol_floor: Minimum volatility assumption
            vol_ceiling: Maximum volatility assumption
            
        Returns:
            Scale factor (1.0 = full position, <1.0 = reduced)
        """
        # Clamp realized vol
        clamped_vol = max(vol_floor, min(vol_ceiling, self.realized_vol_20d))
        
        # Scale factor = target / realized
        return target_vol / clamped_vol
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime_id': self.regime_id,
            'regime_descriptor': self.regime_descriptor,
            'realized_vol_20d': self.realized_vol_20d,
            'realized_vol_60d': self.realized_vol_60d,
            'vol_of_vol': self.vol_of_vol,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_60d': self.max_drawdown_60d,
            'implied_vol': self.implied_vol,
            'regime_age_days': self.regime_age_days,
            'regime_confidence': self.regime_confidence,
        }


class RegimeMetricsService:
    """
    Public service for accessing regime-related metrics.
    
    This provides a clean API that LiveEngine and other components can use
    without reaching into private methods of feature builders.
    """
    
    def __init__(self, api):
        """
        Initialize the regime metrics service.
        
        Args:
            api: AsOfQueryAPI instance for data access
        """
        self.api = api
        self._cache: Dict[date, RegimeMetrics] = {}
        self._benchmark_asset_id: Optional[int] = None
    
    def get_current_regime(self, as_of_date: date) -> Tuple[int, str]:
        """
        Get the current regime ID and descriptor.
        
        Args:
            as_of_date: Date to query
            
        Returns:
            Tuple of (regime_id, regime_descriptor)
        """
        metrics = self.get_regime_metrics(as_of_date)
        return metrics.regime_id, metrics.regime_descriptor
    
    def get_current_volatility(self, as_of_date: date) -> float:
        """
        Get current realized volatility (20-day, annualized).
        
        Args:
            as_of_date: Date to query
            
        Returns:
            Annualized volatility
        """
        metrics = self.get_regime_metrics(as_of_date)
        return metrics.realized_vol_20d
    
    def get_current_drawdown(self, as_of_date: date) -> float:
        """
        Get current market drawdown.
        
        Args:
            as_of_date: Date to query
            
        Returns:
            Current drawdown (negative number)
        """
        metrics = self.get_regime_metrics(as_of_date)
        return metrics.current_drawdown
    
    def get_regime_metrics(self, as_of_date: date) -> RegimeMetrics:
        """
        Get full regime metrics for a date.
        
        Results are cached for efficiency.
        
        Args:
            as_of_date: Date to query
            
        Returns:
            RegimeMetrics instance
        """
        # Check cache
        if as_of_date in self._cache:
            return self._cache[as_of_date]
        
        # Build metrics
        metrics = self._build_regime_metrics(as_of_date)
        
        # Cache result
        self._cache[as_of_date] = metrics
        
        return metrics
    
    def get_vol_scale_factor(
        self,
        as_of_date: date,
        target_vol: float = 0.15,
        vol_floor: float = 0.10,
        vol_ceiling: float = 0.60,
    ) -> float:
        """
        Get volatility-based position scale factor.
        
        Args:
            as_of_date: Date to query
            target_vol: Target portfolio volatility
            vol_floor: Minimum volatility assumption
            vol_ceiling: Maximum volatility assumption
            
        Returns:
            Scale factor
        """
        metrics = self.get_regime_metrics(as_of_date)
        return metrics.get_vol_scale_factor(target_vol, vol_floor, vol_ceiling)
    
    def clear_cache(self):
        """Clear the metrics cache."""
        self._cache.clear()
    
    def _build_regime_metrics(self, as_of_date: date) -> RegimeMetrics:
        """Build regime metrics from data."""
        # Get regime from database
        regime_id, regime_descriptor, regime_age = self._query_regime(as_of_date)
        
        # Get market data for volatility calculation
        vol_metrics = self._compute_volatility_metrics(as_of_date)
        
        return RegimeMetrics(
            regime_id=regime_id,
            regime_descriptor=regime_descriptor,
            realized_vol_20d=vol_metrics.get('realized_vol_20d', 0.15),
            realized_vol_60d=vol_metrics.get('realized_vol_60d'),
            vol_of_vol=vol_metrics.get('vol_of_vol'),
            current_drawdown=vol_metrics.get('current_drawdown', 0.0),
            max_drawdown_60d=vol_metrics.get('max_drawdown_60d'),
            regime_age_days=regime_age,
        )
    
    def _query_regime(self, as_of_date: date) -> Tuple[int, str, int]:
        """Query regime from database."""
        try:
            result = self.api.storage.conn.execute("""
                SELECT regime_id, regime_descriptor, date
                FROM regimes
                WHERE date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, [as_of_date]).df()
            
            if len(result) == 0:
                return -1, "unknown", 0
            
            regime_id = int(result['regime_id'].iloc[0])
            regime_descriptor = str(result['regime_descriptor'].iloc[0])
            regime_date = result['date'].iloc[0]
            
            # Calculate regime age
            if hasattr(regime_date, 'date'):
                regime_date = regime_date.date()
            regime_age = (as_of_date - regime_date).days if regime_date else 0
            
            return regime_id, regime_descriptor, regime_age
            
        except Exception as e:
            logger.debug(f"Could not query regime: {e}")
            return -1, "unknown", 0
    
    def _compute_volatility_metrics(self, as_of_date: date) -> Dict:
        """Compute volatility metrics from market data."""
        try:
            # Get benchmark asset_id (SPY)
            if self._benchmark_asset_id is None:
                benchmark_df = self.api.storage.query(
                    "SELECT asset_id FROM assets WHERE symbol = 'SPY'"
                )
                if len(benchmark_df) == 0:
                    return {'realized_vol_20d': 0.15}
                self._benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
            
            # Get recent bars
            bars_df = self.api.get_bars_asof(
                as_of_date, 
                lookback_days=90, 
                universe={self._benchmark_asset_id}
            )
            
            if len(bars_df) < 21:
                return {'realized_vol_20d': 0.15}
            
            bars_df = bars_df.sort_values('date')
            prices = bars_df['adj_close'].values
            
            # Calculate returns
            returns = np.log(prices[1:] / prices[:-1])
            
            # Realized volatility (annualized)
            realized_vol_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.15
            realized_vol_60d = np.std(returns[-60:]) * np.sqrt(252) if len(returns) >= 60 else None
            
            # Vol of vol
            vol_of_vol = None
            if len(returns) >= 40:
                rolling_vols = pd.Series(returns).rolling(20).std().dropna().values * np.sqrt(252)
                vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) > 1 else None
            
            # Drawdown
            rolling_max = np.maximum.accumulate(prices)
            drawdowns = (prices - rolling_max) / rolling_max
            current_drawdown = float(drawdowns[-1])
            max_drawdown_60d = float(np.min(drawdowns[-60:])) if len(drawdowns) >= 60 else None
            
            return {
                'realized_vol_20d': float(realized_vol_20d),
                'realized_vol_60d': realized_vol_60d,
                'vol_of_vol': vol_of_vol,
                'current_drawdown': current_drawdown,
                'max_drawdown_60d': max_drawdown_60d,
            }
            
        except Exception as e:
            logger.debug(f"Error computing volatility metrics: {e}")
            return {'realized_vol_20d': 0.15}

