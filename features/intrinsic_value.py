"""Intrinsic value estimation module (Graham-style)."""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import date
from sklearn.linear_model import LinearRegression
from data.asof_api import AsOfQueryAPI


class IntrinsicValueEstimator:
    """Estimates intrinsic value using fundamentals and growth."""
    
    def __init__(self, api: AsOfQueryAPI):
        self.api = api
    
    def estimate_growth_rate(
        self,
        fundamentals_df: pd.DataFrame,
        metric: str = "eps",
        quarters: int = 8
    ) -> float:
        """
        Estimate growth rate using point-in-time regression.
        
        Only uses data available at each point in time.
        """
        if len(fundamentals_df) < 2:
            return 0.0
        
        fundamentals_df = fundamentals_df.sort_values('report_release_date')
        values = fundamentals_df[metric].values
        
        if len(values) < 2 or np.any(values <= 0):
            return 0.0
        
        # Fit regression on log scale
        log_values = np.log(values)
        X = np.arange(len(log_values)).reshape(-1, 1)
        y = log_values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Annualized growth rate
        growth = model.coef_[0] * 4  # Quarterly to annual
        
        return growth
    
    def compute_intrinsic_value(
        self,
        asset_id: int,
        as_of_date: date,
        required_return: float = 0.15
    ) -> Optional[float]:
        """
        Compute intrinsic value using Graham formula.
        
        Simplified: IV = EPS * (8.5 + 2 * growth_rate) / required_return
        """
        # Get fundamentals as-of date
        fundamentals_df = self.api.get_fundamentals_asof(as_of_date, universe={asset_id})
        
        if len(fundamentals_df) == 0 or 'eps' not in fundamentals_df.columns:
            return None
        
        eps = fundamentals_df['eps'].iloc[0]
        if eps <= 0:
            return None
        
        # Estimate growth rate
        growth_rate = self.estimate_growth_rate(fundamentals_df, metric='eps')
        
        # Graham formula (simplified)
        intrinsic_value = eps * (8.5 + 2 * growth_rate * 100) / (required_return * 100)
        
        return intrinsic_value
    
    def compute_intrinsic_value_ratio(
        self,
        asset_id: int,
        as_of_date: date,
        current_price: float
    ) -> Optional[float]:
        """Compute price / intrinsic_value ratio."""
        intrinsic_value = self.compute_intrinsic_value(asset_id, as_of_date)
        
        if intrinsic_value is None or intrinsic_value <= 0:
            return None
        
        return current_price / intrinsic_value

