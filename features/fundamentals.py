"""Fundamentals and valuation feature engineering."""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import date
from data.asof_api import AsOfQueryAPI
from sklearn.linear_model import LinearRegression


class FundamentalsFeatureBuilder:
    """Builds fundamental and valuation features."""
    
    def __init__(self, api: AsOfQueryAPI):
        self.api = api
    
    def compute_growth_rates(
        self,
        fundamentals_df: pd.DataFrame,
        metric: str = "eps",
        quarters: int = 8
    ) -> pd.Series:
        """
        Compute growth rate using only historical data available at each point.
        
        Uses linear regression on log scale over last N quarters.
        """
        growth_rates = []
        
        for asset_id, asset_fundamentals in fundamentals_df.groupby('asset_id'):
            asset_fundamentals = asset_fundamentals.sort_values('report_release_date')
            
            if metric not in asset_fundamentals.columns:
                continue
            
            # For each report release date, use only data released before that date
            for idx, row in asset_fundamentals.iterrows():
                release_date = row['report_release_date']
                available_data = asset_fundamentals[
                    asset_fundamentals['report_release_date'] <= release_date
                ].tail(quarters)
                
                if len(available_data) < 2:
                    growth_rates.append({
                        'asset_id': asset_id,
                        'report_release_date': release_date,
                        f'{metric}_growth': np.nan
                    })
                    continue
                
                # Fit regression on log scale
                values = available_data[metric].values
                if np.any(values <= 0):
                    growth_rates.append({
                        'asset_id': asset_id,
                        'report_release_date': release_date,
                        f'{metric}_growth': np.nan
                    })
                    continue
                
                log_values = np.log(values)
                X = np.arange(len(log_values)).reshape(-1, 1)
                y = log_values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Annualized growth rate
                growth = model.coef_[0] * 4  # Quarterly to annual
                
                growth_rates.append({
                    'asset_id': asset_id,
                    'report_release_date': release_date,
                    f'{metric}_growth': growth
                })
        
        if len(growth_rates) == 0:
            return pd.DataFrame(columns=['asset_id', 'report_release_date', f'{metric}_growth'])
        
        return pd.DataFrame(growth_rates)
    
    def compute_valuation_ratios(
        self,
        fundamentals_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        as_of_date: date
    ) -> pd.DataFrame:
        """
        Compute valuation ratios (P/E, P/B, EV/EBITDA, etc.).
        
        Args:
            fundamentals_df: DataFrame with fundamental metrics
            bars_df: DataFrame with prices (should have latest price as-of as_of_date)
            as_of_date: Date to compute ratios as-of
        """
        # Get latest prices
        latest_bars = bars_df[bars_df['date'] == as_of_date][['asset_id', 'adj_close']]
        
        # Merge fundamentals with prices
        merged = fundamentals_df.merge(latest_bars, on='asset_id', how='left')
        
        ratios = pd.DataFrame()
        ratios['asset_id'] = merged['asset_id']
        ratios['date'] = as_of_date
        
        # P/E ratio
        if 'eps' in merged.columns:
            ratios['pe_ratio'] = merged['adj_close'] / (merged['eps'] + 1e-10)
        
        # Forward P/E (if estimates available)
        if 'eps_estimate' in merged.columns:
            ratios['forward_pe'] = merged['adj_close'] / (merged['eps_estimate'] + 1e-10)
        
        # P/B ratio (would need book value, approximated here)
        if 'total_equity' in merged.columns:
            # Assume shares outstanding from market cap / price (rough approximation)
            ratios['pb_ratio'] = merged['adj_close'] / (merged['total_equity'] + 1e-10)
        
        # EV/EBITDA (would need enterprise value, simplified here)
        if 'ebitda' in merged.columns:
            # EV approximation: market cap + debt - cash (simplified)
            ratios['ev_ebitda'] = merged['adj_close'] / (merged['ebitda'] + 1e-10)
        
        # FCF yield
        if 'free_cash_flow' in merged.columns:
            ratios['fcf_yield'] = merged['free_cash_flow'] / (merged['adj_close'] + 1e-10)
        
        return ratios
    
    def compute_quality_metrics(
        self,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute quality flags (stable profitability, low earnings vol, etc.).
        """
        quality = pd.DataFrame()
        quality['asset_id'] = fundamentals_df['asset_id'].unique()
        
        for asset_id, asset_fundamentals in fundamentals_df.groupby('asset_id'):
            asset_fundamentals = asset_fundamentals.sort_values('report_release_date')
            
            # Earnings stability (coefficient of variation)
            if 'eps' in asset_fundamentals.columns:
                eps_values = asset_fundamentals['eps'].values
                if len(eps_values) > 1 and np.std(eps_values) > 0:
                    quality.loc[quality['asset_id'] == asset_id, 'eps_stability'] = (
                        1.0 / (1.0 + np.std(eps_values) / (np.abs(np.mean(eps_values)) + 1e-10))
                    )
            
            # Profit margin stability
            if 'revenue' in asset_fundamentals.columns and 'eps' in asset_fundamentals.columns:
                margins = asset_fundamentals['eps'] / (asset_fundamentals['revenue'] + 1e-10)
                if len(margins) > 1:
                    quality.loc[quality['asset_id'] == asset_id, 'margin_stability'] = (
                        1.0 / (1.0 + margins.std())
                    )
        
        return quality
    
    def build_features(
        self,
        as_of_date: date,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Build all fundamental features as-of a specific date.
        """
        # Get fundamentals as-of date
        fundamentals_df = self.api.get_fundamentals_asof(as_of_date, universe)
        
        if len(fundamentals_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        # Get latest bars for prices
        bars_df = self.api.get_bars_asof(as_of_date, lookback_days=1, universe=universe)
        
        # Compute growth rates
        eps_growth = self.compute_growth_rates(fundamentals_df, metric='eps')
        revenue_growth = self.compute_growth_rates(fundamentals_df, metric='revenue')
        
        # Compute valuation ratios
        valuation_ratios = self.compute_valuation_ratios(fundamentals_df, bars_df, as_of_date)
        
        # Compute quality metrics
        quality_metrics = self.compute_quality_metrics(fundamentals_df)
        
        # Merge all features
        result = fundamentals_df[['asset_id', 'report_release_date']].copy()
        result['date'] = as_of_date
        
        if len(eps_growth) > 0:
            result = result.merge(eps_growth, on=['asset_id', 'report_release_date'], how='left')
        if len(revenue_growth) > 0:
            result = result.merge(revenue_growth, on=['asset_id', 'report_release_date'], how='left')
        if len(valuation_ratios) > 0:
            result = result.merge(valuation_ratios, on=['asset_id', 'date'], how='left')
        if len(quality_metrics) > 0:
            result = result.merge(quality_metrics, on='asset_id', how='left')
        
        return result

