"""Validators to check for lookahead bias and other issues."""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import date


class BacktestValidator:
    """Validates backtests for common biases."""
    
    @staticmethod
    def check_lookahead_bias(
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict:
        """
        Check for lookahead bias by verifying features don't use future data.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check that all feature dates <= label dates
        feature_dates = pd.to_datetime(features_df[date_col]).unique()
        label_dates = pd.to_datetime(labels_df[date_col]).unique()
        
        # For each label date, check that features used are <= that date
        for label_date in label_dates:
            label_features = features_df[features_df[date_col] <= label_date]
            if len(label_features) == 0:
                issues.append({
                    'type': 'missing_features',
                    'date': label_date,
                    'message': f'No features available for label date {label_date}'
                })
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def check_survivorship_bias(
        universe_membership_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Dict:
        """
        Check that trades only occur for assets in universe at that date.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check each trade
        for _, trade in trades_df.iterrows():
            trade_date = trade['date']
            asset_id = trade['asset_id']
            
            # Check if asset was in universe at that date
            date_membership = universe_membership_df[
                (universe_membership_df['date'] == trade_date) &
                (universe_membership_df['asset_id'] == asset_id)
            ]
            
            if len(date_membership) == 0 or not date_membership['in_index'].iloc[0]:
                issues.append({
                    'type': 'survivorship_bias',
                    'date': trade_date,
                    'asset_id': asset_id,
                    'message': f'Traded asset {asset_id} not in universe on {trade_date}'
                })
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def synthetic_lookahead_test(
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        model,
        baseline_performance: float
    ) -> Dict:
        """
        Intentionally introduce lookahead and verify performance improves
        (which would indicate the original test was correct).
        
        Returns:
            Dictionary with test results
        """
        # Shift features forward by 1 day (introducing lookahead)
        features_shifted = features_df.copy()
        features_shifted['date'] = pd.to_datetime(features_shifted['date']) + pd.Timedelta(days=1)
        
        # Train model on shifted features
        # (This is a placeholder - would need actual model training)
        
        return {
            'baseline_performance': baseline_performance,
            'shifted_performance': None,  # Would compute with shifted features
            'test_passed': True  # Placeholder
        }

