"""Data quality checks and validation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import date
from .storage import StorageBackend
from .universe import TradingCalendar


class DataQualityChecker:
    """Performs data quality checks."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.calendar = TradingCalendar()
    
    def check_bar_gaps(
        self,
        bars_df: pd.DataFrame,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Check for gaps in daily bars.
        
        Returns DataFrame with columns: asset_id, gap_start, gap_end, gap_days
        """
        gaps = []
        
        if len(bars_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'gap_start', 'gap_end', 'gap_days'])
        
        # Get all trading days in range
        trading_days = self.calendar.get_trading_days(start_date, end_date)
        trading_days_set = set(trading_days.date)
        
        # Check each asset
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            asset_dates = set(pd.to_datetime(asset_bars['date']).dt.date)
            missing_dates = trading_days_set - asset_dates
            
            if len(missing_dates) == 0:
                continue
            
            # Group consecutive missing dates
            missing_sorted = sorted(missing_dates)
            gap_start = missing_sorted[0]
            gap_end = missing_sorted[0]
            
            for i in range(1, len(missing_sorted)):
                if (missing_sorted[i] - gap_end).days == 1:
                    gap_end = missing_sorted[i]
                else:
                    # End current gap, start new one
                    gap_days = (gap_end - gap_start).days + 1
                    gaps.append({
                        'asset_id': asset_id,
                        'gap_start': gap_start,
                        'gap_end': gap_end,
                        'gap_days': gap_days
                    })
                    gap_start = missing_sorted[i]
                    gap_end = missing_sorted[i]
            
            # Add final gap
            gap_days = (gap_end - gap_start).days + 1
            gaps.append({
                'asset_id': asset_id,
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap_days': gap_days
            })
        
        if len(gaps) == 0:
            return pd.DataFrame(columns=['asset_id', 'gap_start', 'gap_end', 'gap_days'])
        
        return pd.DataFrame(gaps)
    
    def check_price_outliers(
        self,
        bars_df: pd.DataFrame,
        z_threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Check for extreme price outliers (likely data errors).
        
        Returns DataFrame with columns: asset_id, date, field, value, z_score, issue_type
        """
        outliers = []
        
        if len(bars_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'field', 'value', 'z_score', 'issue_type'])
        
        price_fields = ['open', 'high', 'low', 'close', 'adj_close']
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            asset_bars = asset_bars.sort_values('date')
            
            for field in price_fields:
                if field not in asset_bars.columns:
                    continue
                
                values = asset_bars[field].values
                
                # Compute log returns
                log_prices = np.log(values + 1e-10)  # Avoid log(0)
                log_returns = np.diff(log_prices)
                
                # Compute z-scores of returns
                mean_return = np.mean(log_returns)
                std_return = np.std(log_returns)
                
                if std_return == 0:
                    continue
                
                z_scores = np.abs((log_returns - mean_return) / std_return)
                
                # Find outliers
                outlier_indices = np.where(z_scores > z_threshold)[0]
                
                for idx in outlier_indices:
                    date_idx = idx + 1  # +1 because diff reduces length by 1
                    if date_idx < len(asset_bars):
                        outliers.append({
                            'asset_id': asset_id,
                            'date': asset_bars.iloc[date_idx]['date'],
                            'field': field,
                            'value': values[date_idx],
                            'z_score': z_scores[idx],
                            'issue_type': 'price_outlier'
                        })
        
        if len(outliers) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'field', 'value', 'z_score', 'issue_type'])
        
        return pd.DataFrame(outliers)
    
    def check_duplicates(
        self,
        bars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Check for duplicate (asset_id, date) pairs.
        
        Returns DataFrame with duplicate records
        """
        if len(bars_df) == 0:
            return pd.DataFrame()
        
        duplicates = bars_df[bars_df.duplicated(subset=['asset_id', 'date'], keep=False)]
        return duplicates
    
    def check_volume_outliers(
        self,
        bars_df: pd.DataFrame,
        volume_threshold_multiplier: float = 10.0
    ) -> pd.DataFrame:
        """
        Check for extreme volume spikes (possible data errors or corporate events).
        
        Returns DataFrame with columns: asset_id, date, volume, median_volume, ratio, issue_type
        """
        outliers = []
        
        if len(bars_df) == 0 or 'volume' not in bars_df.columns:
            return pd.DataFrame(columns=['asset_id', 'date', 'volume', 'median_volume', 'ratio', 'issue_type'])
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            # Reset index to ensure positional indexing works correctly
            asset_bars = asset_bars.sort_values('date').reset_index(drop=True)
            volumes = asset_bars['volume'].values
            
            # Compute rolling median (30-day window)
            window = min(30, len(volumes))
            if window < 2:
                continue
            
            median_volumes = pd.Series(volumes).rolling(window=window, center=False).median()
            
            # Iterate using positional index
            for pos_idx in range(window - 1, len(asset_bars)):
                row = asset_bars.iloc[pos_idx]
                volume = row['volume']
                median_vol = median_volumes.iloc[pos_idx]
                
                if pd.notna(median_vol) and median_vol > 0:
                    ratio = volume / median_vol
                    if ratio > volume_threshold_multiplier:
                        outliers.append({
                            'asset_id': asset_id,
                            'date': row['date'],
                            'volume': volume,
                            'median_volume': median_vol,
                            'ratio': ratio,
                            'issue_type': 'volume_outlier'
                        })
        
        if len(outliers) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'volume', 'median_volume', 'ratio', 'issue_type'])
        
        return pd.DataFrame(outliers)
    
    def check_currency_consistency(
        self,
        bars_df: pd.DataFrame,
        assets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Check that all bars for an asset use the same currency.
        
        Returns DataFrame with inconsistencies
        """
        # This is a placeholder - would need currency column in bars_df
        # For now, assume USD for all US equities
        return pd.DataFrame(columns=['asset_id', 'issue_type', 'message'])
    
    def generate_quality_report(
        self,
        bars_df: pd.DataFrame,
        actions_df: Optional[pd.DataFrame] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict:
        """
        Generate comprehensive quality report.
        
        Returns dictionary with summary statistics and issue counts
        """
        report = {
            'total_assets': bars_df['asset_id'].nunique() if len(bars_df) > 0 else 0,
            'total_bars': len(bars_df),
            'date_range': {
                'start': bars_df['date'].min() if len(bars_df) > 0 else None,
                'end': bars_df['date'].max() if len(bars_df) > 0 else None
            },
            'issues': {}
        }
        
        # Check gaps
        if start_date and end_date:
            gaps = self.check_bar_gaps(bars_df, start_date, end_date)
            report['issues']['gaps'] = {
                'count': len(gaps),
                'total_gap_days': gaps['gap_days'].sum() if len(gaps) > 0 else 0
            }
        
        # Check duplicates
        duplicates = self.check_duplicates(bars_df)
        report['issues']['duplicates'] = {
            'count': len(duplicates)
        }
        
        # Check price outliers
        price_outliers = self.check_price_outliers(bars_df)
        report['issues']['price_outliers'] = {
            'count': len(price_outliers)
        }
        
        # Check volume outliers
        volume_outliers = self.check_volume_outliers(bars_df)
        report['issues']['volume_outliers'] = {
            'count': len(volume_outliers)
        }
        
        # Check adj_close consistency if actions provided
        if actions_df is not None and len(actions_df) > 0:
            from .normalize import DataNormalizer
            normalizer = DataNormalizer(self.storage)
            adj_issues = normalizer.validate_adj_close_consistency(bars_df, actions_df)
            report['issues']['adj_close_inconsistencies'] = {
                'count': len(adj_issues)
            }
        
        return report

