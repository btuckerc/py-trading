"""Cross-sectional and market/sector features."""

import pandas as pd
import numpy as np
from typing import Optional, Set, List
from datetime import date
from data.asof_api import AsOfQueryAPI


class CrossSectionalFeatureBuilder:
    """Builds cross-sectional and market context features."""
    
    def __init__(self, api: AsOfQueryAPI):
        self.api = api
    
    def compute_index_returns(
        self,
        as_of_date: date,
        benchmark_symbol: str = "SPY",
        windows: List[int] = [1, 5, 20]
    ) -> pd.Series:
        """Compute index returns over multiple horizons."""
        # Get benchmark bars
        benchmark_df = self.api.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{benchmark_symbol}'")
        if len(benchmark_df) == 0:
            return pd.Series()
        
        benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
        bars_df = self.api.get_bars_asof(as_of_date, lookback_days=max(windows) + 10)
        benchmark_bars = bars_df[bars_df['asset_id'] == benchmark_asset_id].copy()
        
        if len(benchmark_bars) == 0:
            return pd.Series()
        
        benchmark_bars = benchmark_bars.set_index('date').sort_index()
        prices = benchmark_bars['adj_close']
        
        index_returns = {}
        for window in windows:
            returns = np.log(prices / prices.shift(window))
            if len(returns) > 0:
                index_returns[f'index_return_{window}d'] = returns.iloc[-1]
        
        return pd.Series(index_returns)
    
    def compute_sector_returns(
        self,
        bars_df: pd.DataFrame,
        sectors_df: pd.DataFrame,
        windows: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        Compute sector returns for each asset.
        
        Args:
            bars_df: DataFrame with asset_id, date, adj_close
            sectors_df: DataFrame with asset_id, sector
            windows: Return windows to compute
        """
        # Merge sectors
        bars_with_sector = bars_df.merge(sectors_df, on='asset_id', how='left')
        
        sector_returns = []
        
        for sector, sector_bars in bars_with_sector.groupby('sector'):
            sector_bars = sector_bars.set_index('date').sort_index()
            sector_prices = sector_bars.groupby('date')['adj_close'].mean()  # Equal-weight sector
            
            for window in windows:
                returns = np.log(sector_prices / sector_prices.shift(window))
                sector_returns.append({
                    'sector': sector,
                    'date': sector_prices.index,
                    f'sector_return_{window}d': returns
                })
        
        # Merge back to bars_df
        sector_returns_df = pd.DataFrame(sector_returns)
        if len(sector_returns_df) > 0:
            bars_with_sector = bars_with_sector.merge(
                sector_returns_df,
                on=['sector', 'date'],
                how='left'
            )
        
        return bars_with_sector
    
    def compute_cross_sectional_ranks(
        self,
        features_df: pd.DataFrame,
        rank_features: List[str],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute cross-sectional ranks/z-scores of features.
        
        Args:
            features_df: DataFrame with date, asset_id, and feature columns
            rank_features: List of feature names to rank
            group_by: Optional column to group by (e.g., 'sector')
        """
        result = features_df.copy()
        
        for feature in rank_features:
            if feature not in result.columns:
                continue
            
            if group_by:
                # Rank within groups
                result[f'{feature}_rank'] = result.groupby(['date', group_by])[feature].rank(pct=True)
                result[f'{feature}_zscore'] = result.groupby(['date', group_by])[feature].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
            else:
                # Rank across all assets
                result[f'{feature}_rank'] = result.groupby('date')[feature].rank(pct=True)
                result[f'{feature}_zscore'] = result.groupby('date')[feature].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
        
        return result
    
    def compute_correlations(
        self,
        bars_df: pd.DataFrame,
        benchmark_symbol: str = "SPY",
        window: int = 60
    ) -> pd.DataFrame:
        """
        Compute rolling correlation to benchmark.
        
        Args:
            bars_df: DataFrame with asset_id, date, adj_close
            benchmark_symbol: Benchmark symbol
            window: Rolling window in days
        """
        # Get benchmark returns
        benchmark_df = self.api.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{benchmark_symbol}'")
        if len(benchmark_df) == 0:
            return bars_df
        
        benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
        all_bars = self.api.get_bars_asof(bars_df['date'].max(), lookback_days=window + 10)
        
        benchmark_bars = all_bars[all_bars['asset_id'] == benchmark_asset_id].copy()
        benchmark_bars = benchmark_bars.set_index('date').sort_index()
        benchmark_returns = np.log(benchmark_bars['adj_close'] / benchmark_bars['adj_close'].shift(1))
        
        correlations = []
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            asset_bars = asset_bars.set_index('date').sort_index()
            asset_returns = np.log(asset_bars['adj_close'] / asset_bars['adj_close'].shift(1))
            
            # Align dates
            aligned = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            # Rolling correlation
            rolling_corr = aligned['asset'].rolling(window).corr(aligned['benchmark'])
            
            # Convert index to date type for merging
            corr_dates = pd.to_datetime(rolling_corr.index).date if hasattr(rolling_corr.index, 'date') else rolling_corr.index
            if not isinstance(corr_dates, pd.Series):
                corr_dates = pd.Series(corr_dates)
            
            correlations.append(pd.DataFrame({
                'asset_id': asset_id,
                'date': corr_dates,
                'correlation_to_benchmark': rolling_corr.values
            }))
        
        if len(correlations) > 0:
            corr_df = pd.concat(correlations, ignore_index=True)
            # Ensure date types match
            corr_df['date'] = pd.to_datetime(corr_df['date']).dt.date
            bars_df['date'] = pd.to_datetime(bars_df['date']).dt.date
            result = bars_df.merge(corr_df, on=['asset_id', 'date'], how='left')
        else:
            result = bars_df.copy()
            result['correlation_to_benchmark'] = np.nan
        
        return result
    
    def build_features(
        self,
        bars_df: pd.DataFrame,
        as_of_date: date,
        sectors_df: Optional[pd.DataFrame] = None,
        benchmark_symbol: str = "SPY"
    ) -> pd.DataFrame:
        """
        Build all cross-sectional features.
        
        Returns:
            DataFrame with cross-sectional features added
        """
        result = bars_df.copy()
        
        # Index returns (add as constant per date)
        index_returns = self.compute_index_returns(as_of_date, benchmark_symbol)
        if len(index_returns) > 0:
            for col in index_returns.index:
                result[col] = index_returns[col]
        
        # Sector returns (if sectors available)
        if sectors_df is not None and len(sectors_df) > 0:
            result = self.compute_sector_returns(result, sectors_df)
        
        # Cross-sectional ranks of key features
        rank_features = ['log_return_20d', 'momentum_20d', 'volatility_20d']
        available_rank_features = [f for f in rank_features if f in result.columns]
        if len(available_rank_features) > 0:
            result = self.compute_cross_sectional_ranks(
                result,
                available_rank_features,
                group_by='sector' if 'sector' in result.columns else None
            )
        
        # Correlations
        result = self.compute_correlations(result, benchmark_symbol)
        
        return result

