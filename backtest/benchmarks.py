"""Benchmark strategies for comparison."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import date
from backtest.vectorized import VectorizedBacktester


class BenchmarkStrategies:
    """Collection of benchmark strategies."""
    
    def __init__(self, backtester: VectorizedBacktester):
        self.backtester = backtester
    
    def buy_and_hold(
        self,
        prices_df: pd.DataFrame,
        symbol: str = "SPY"
    ) -> pd.DataFrame:
        """
        Buy and hold benchmark.
        
        Args:
            prices_df: DataFrame with columns: date, asset_id, adj_close
            symbol: Symbol to buy and hold
        
        Returns:
            Equity curve DataFrame
        """
        # Get asset_id for symbol
        asset_id = prices_df[prices_df.get('symbol') == symbol]['asset_id'].iloc[0] if 'symbol' in prices_df.columns else None
        
        if asset_id is None:
            # Assume first asset_id if symbol column doesn't exist
            asset_id = prices_df['asset_id'].iloc[0]
        
        # Create target weights: 100% in this asset
        all_dates = sorted(prices_df['date'].unique())
        target_weights = []
        
        for date_val in all_dates:
            target_weights.append({
                'date': date_val,
                'asset_id': asset_id,
                'weight': 1.0
            })
        
        target_weights_df = pd.DataFrame(target_weights)
        
        return self.backtester.run_backtest(prices_df, target_weights_df)
    
    def equal_weight_universe(
        self,
        prices_df: pd.DataFrame,
        rebalance_frequency: str = "monthly"
    ) -> pd.DataFrame:
        """
        Equal-weight portfolio of all assets in universe.
        
        Args:
            prices_df: DataFrame with columns: date, asset_id, adj_close
            rebalance_frequency: "daily", "weekly", or "monthly"
        
        Returns:
            Equity curve DataFrame
        """
        # Get all unique assets
        all_assets = prices_df['asset_id'].unique()
        num_assets = len(all_assets)
        weight_per_asset = 1.0 / num_assets
        
        # Determine rebalance dates
        all_dates = sorted(prices_df['date'].unique())
        dates_df = pd.DataFrame({'date': all_dates})
        dates_df['date'] = pd.to_datetime(dates_df['date'])
        
        if rebalance_frequency == "daily":
            rebalance_dates = all_dates
        elif rebalance_frequency == "weekly":
            dates_df['week'] = dates_df['date'].dt.isocalendar().week
            rebalance_dates = dates_df.groupby('week')['date'].first().dt.date.tolist()
        elif rebalance_frequency == "monthly":
            dates_df['year_month'] = dates_df['date'].dt.to_period('M')
            rebalance_dates = dates_df.groupby('year_month')['date'].first().dt.date.tolist()
        else:
            rebalance_dates = all_dates
        
        # Create target weights
        target_weights = []
        for date_val in all_dates:
            # Check if this is a rebalance date
            if date_val in rebalance_dates:
                # Equal weight all assets
                for asset_id in all_assets:
                    target_weights.append({
                        'date': date_val,
                        'asset_id': asset_id,
                        'weight': weight_per_asset
                    })
            else:
                # Hold previous weights (will be handled by backtester)
                pass
        
        target_weights_df = pd.DataFrame(target_weights)
        
        return self.backtester.run_backtest(prices_df, target_weights_df)
    
    def random_portfolio(
        self,
        prices_df: pd.DataFrame,
        k: int = 20,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Random portfolio: randomly select K assets each day.
        
        Args:
            prices_df: DataFrame with columns: date, asset_id, adj_close
            k: Number of assets to hold
            seed: Random seed
        
        Returns:
            Equity curve DataFrame
        """
        if seed is not None:
            np.random.seed(seed)
        
        all_assets = prices_df['asset_id'].unique()
        all_dates = sorted(prices_df['date'].unique())
        
        target_weights = []
        weight_per_asset = 1.0 / k
        
        for date_val in all_dates:
            # Randomly select K assets
            selected_assets = np.random.choice(all_assets, size=min(k, len(all_assets)), replace=False)
            
            for asset_id in selected_assets:
                target_weights.append({
                    'date': date_val,
                    'asset_id': asset_id,
                    'weight': weight_per_asset
                })
        
        target_weights_df = pd.DataFrame(target_weights)
        
        return self.backtester.run_backtest(prices_df, target_weights_df)
    
    def random_portfolio_distribution(
        self,
        prices_df: pd.DataFrame,
        k: int = 20,
        n_runs: int = 100
    ) -> pd.DataFrame:
        """
        Run multiple random portfolios to get performance distribution.
        
        Returns:
            DataFrame with columns: run_id, date, equity
        """
        all_results = []
        
        for run_id in range(n_runs):
            equity_curve = self.random_portfolio(prices_df, k, seed=run_id)
            equity_curve['run_id'] = run_id
            all_results.append(equity_curve)
        
        return pd.concat(all_results, ignore_index=True)

