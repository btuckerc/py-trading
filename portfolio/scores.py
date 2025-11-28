"""Convert predictions to risk-adjusted scores."""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class ScoreConverter:
    """Converts multi-horizon predictions to scores."""
    
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
        weights: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Combine scores from multiple horizons.
        
        Args:
            predictions: Dict mapping horizon -> {mu, sigma}
            method: "weighted_average" or "best_horizon"
            weights: Optional weights for each horizon
        
        Returns:
            Combined score
        """
        if method == "best_horizon":
            # Pick horizon with highest adjusted score
            best_score = -np.inf
            for horizon, pred in predictions.items():
                daily_growth = ScoreConverter.convert_to_daily_growth_rate(pred['mu'], horizon)
                score = ScoreConverter.compute_risk_adjusted_score(daily_growth, pred['sigma'])
                if score > best_score:
                    best_score = score
            return best_score
        
        elif method == "weighted_average":
            # Weighted average of daily growth rates
            if weights is None:
                # Default: more weight on intermediate horizons
                weights = {1: 0.1, 5: 0.3, 20: 0.4, 120: 0.2}
            
            total_score = 0.0
            total_weight = 0.0
            
            for horizon, pred in predictions.items():
                daily_growth = ScoreConverter.convert_to_daily_growth_rate(pred['mu'], horizon)
                score = ScoreConverter.compute_risk_adjusted_score(daily_growth, pred['sigma'])
                weight = weights.get(horizon, 0.0)
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
        
        else:
            raise ValueError(f"Unknown method: {method}")

