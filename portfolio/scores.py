"""Convert predictions to risk-adjusted scores."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field


# =============================================================================
# Configuration for Multi-Horizon Score Blending
# =============================================================================

@dataclass
class MultiHorizonConfig:
    """
    Configuration for multi-horizon score blending.
    
    This should be loaded from configs/base.yaml to ensure consistency
    between backtests and live trading.
    """
    
    # Horizons to use (in days)
    horizons: List[int] = field(default_factory=lambda: [5, 20])
    
    # Weights for each horizon (should sum to 1.0)
    weights: Dict[int, float] = field(default_factory=lambda: {5: 0.4, 20: 0.6})
    
    # Combination method: "weighted_average", "best_horizon", "uncertainty_weighted"
    method: str = "weighted_average"
    
    # Risk adjustment method: "sharpe_like", "mean_variance", "rank_based"
    risk_adjustment: str = "sharpe_like"
    
    # Risk aversion parameter (for mean_variance method)
    risk_aversion: float = 1.0
    
    @classmethod
    def from_config(cls, config_dict: Optional[Dict] = None) -> 'MultiHorizonConfig':
        """Create from config dictionary."""
        if config_dict is None:
            return cls()
        
        return cls(
            horizons=config_dict.get('horizons', [5, 20]),
            weights=config_dict.get('weights', {5: 0.4, 20: 0.6}),
            method=config_dict.get('method', 'weighted_average'),
            risk_adjustment=config_dict.get('risk_adjustment', 'sharpe_like'),
            risk_aversion=config_dict.get('risk_aversion', 1.0),
        )


# Default config (can be overridden via configs/base.yaml)
DEFAULT_MULTI_HORIZON_CONFIG = MultiHorizonConfig()


class ScoreConverter:
    """Converts multi-horizon predictions to scores."""
    
    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        """
        Initialize ScoreConverter.
        
        Args:
            config: Multi-horizon configuration. If None, uses defaults.
        """
        self.config = config or DEFAULT_MULTI_HORIZON_CONFIG
    
    @staticmethod
    def convert_to_daily_growth_rate(
        mu_h: float,
        horizon_days: int
    ) -> float:
        """
        Convert horizon return to daily-equivalent growth rate.
        
        Uses geometric mean: g = (exp(mu_h))^(1/h) - 1
        """
        if horizon_days <= 0:
            return 0.0
        
        return (np.exp(mu_h) ** (1.0 / horizon_days)) - 1
    
    @staticmethod
    def compute_risk_adjusted_score(
        mu: float,
        sigma: float,
        method: str = "sharpe_like",
        risk_aversion: float = 1.0,
        epsilon: float = 1e-6
    ) -> float:
        """
        Compute risk-adjusted score.
        
        Args:
            mu: Expected return
            sigma: Uncertainty/volatility
            method: "sharpe_like", "mean_variance", or "rank_based"
            risk_aversion: Risk aversion parameter (for mean_variance)
            epsilon: Small constant to avoid division by zero
        """
        if method == "sharpe_like":
            return mu / (sigma + epsilon)
        elif method == "mean_variance":
            return mu - risk_aversion * sigma
        elif method == "rank_based":
            # For rank-based, just return mu (ranking happens later)
            return mu
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def combine_multi_horizon_scores(
        predictions: Dict[int, Dict[str, float]],
        method: str = "weighted_average",
        weights: Optional[Dict[int, float]] = None,
        risk_adjustment: str = "sharpe_like",
        risk_aversion: float = 1.0,
    ) -> float:
        """
        Combine scores from multiple horizons.
        
        Args:
            predictions: Dict mapping horizon -> {mu, sigma}
            method: "weighted_average", "best_horizon", or "uncertainty_weighted"
            weights: Optional weights for each horizon
            risk_adjustment: Risk adjustment method
            risk_aversion: Risk aversion parameter
        
        Returns:
            Combined score
        """
        if method == "best_horizon":
            # Pick horizon with highest adjusted score
            best_score = -np.inf
            for horizon, pred in predictions.items():
                daily_growth = ScoreConverter.convert_to_daily_growth_rate(pred['mu'], horizon)
                score = ScoreConverter.compute_risk_adjusted_score(
                    daily_growth, pred['sigma'], method=risk_adjustment, risk_aversion=risk_aversion
                )
                if score > best_score:
                    best_score = score
            return best_score
        
        elif method == "uncertainty_weighted":
            # Weight by inverse uncertainty (more weight to more certain predictions)
            total_score = 0.0
            total_weight = 0.0
            
            for horizon, pred in predictions.items():
                daily_growth = ScoreConverter.convert_to_daily_growth_rate(pred['mu'], horizon)
                score = ScoreConverter.compute_risk_adjusted_score(
                    daily_growth, pred['sigma'], method=risk_adjustment, risk_aversion=risk_aversion
                )
                # Inverse uncertainty weighting
                uncertainty_weight = 1.0 / (pred['sigma'] + 1e-6)
                total_score += score * uncertainty_weight
                total_weight += uncertainty_weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
        
        elif method == "weighted_average":
            # Weighted average of daily growth rates
            if weights is None:
                # Default: more weight on intermediate horizons
                weights = {1: 0.1, 5: 0.3, 20: 0.4, 120: 0.2}
            
            total_score = 0.0
            total_weight = 0.0
            
            for horizon, pred in predictions.items():
                daily_growth = ScoreConverter.convert_to_daily_growth_rate(pred['mu'], horizon)
                score = ScoreConverter.compute_risk_adjusted_score(
                    daily_growth, pred['sigma'], method=risk_adjustment, risk_aversion=risk_aversion
                )
                weight = weights.get(horizon, 0.0)
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def combine_scores(
        self,
        predictions: Dict[int, Dict[str, float]],
    ) -> float:
        """
        Combine scores using instance config.
        
        Args:
            predictions: Dict mapping horizon -> {mu, sigma}
        
        Returns:
            Combined score
        """
        return self.combine_multi_horizon_scores(
            predictions=predictions,
            method=self.config.method,
            weights=self.config.weights,
            risk_adjustment=self.config.risk_adjustment,
            risk_aversion=self.config.risk_aversion,
        )

