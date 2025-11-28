"""Multi-horizon return label generation."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import date
from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI


class ReturnLabelGenerator:
    """Generates multi-horizon return labels."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.api = AsOfQueryAPI(storage)
    
    def compute_log_returns(
        self,
        prices: pd.Series,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Compute log returns for multiple horizons.
        
        Args:
            prices: Series with date index and price values
            horizons: List of horizon days (e.g., [1, 5, 20, 120])
        
        Returns:
            DataFrame with columns: date, horizon, log_return
        """
        results = []
        
        for horizon in horizons:
            # Forward-looking: price at t+h / price at t
            future_prices = prices.shift(-horizon)
            log_returns = np.log(future_prices / prices)
            
            # Create DataFrame
            returns_df = pd.DataFrame({
                'date': prices.index,
                'horizon': horizon,
                'log_return': log_returns
            })
            
            # Remove rows where we don't have future prices
            returns_df = returns_df.dropna()
            
            results.append(returns_df)
        
        if len(results) == 0:
            return pd.DataFrame(columns=['date', 'horizon', 'log_return'])
        
        return pd.concat(results, ignore_index=True)
    
    def compute_excess_returns(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
        horizon: int
    ) -> pd.Series:
        """
        Compute excess returns vs benchmark.
        
        Args:
            asset_returns: Log returns for asset
            benchmark_returns: Log returns for benchmark
            horizon: Horizon in days
        
        Returns:
            Series of excess log returns
        """
        # Align indices
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        excess = aligned['asset'] - aligned['benchmark']
        return excess
    
    def generate_labels(
        self,
        start_date: date,
        end_date: date,
        horizons: List[int] = [1, 5, 20, 120],
        benchmark_symbol: str = "SPY",
        universe: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate multi-horizon return labels for all assets in universe.
        
        Args:
            start_date: Start date for label generation
            end_date: End date for label generation
            horizons: List of horizon days
            benchmark_symbol: Symbol for benchmark (e.g., "SPY")
            universe: Optional list of asset_ids to generate labels for
        
        Returns:
            DataFrame with columns: date, asset_id, horizon, target_log_return, 
            target_excess_log_return, target_up
        """
        # Get benchmark asset_id
        benchmark_df = self.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{benchmark_symbol}'")
        if len(benchmark_df) == 0:
            raise ValueError(f"Benchmark symbol {benchmark_symbol} not found in assets table")
        benchmark_asset_id = benchmark_df['asset_id'].iloc[0]
        
        # Get all bars up to end_date + max horizon
        max_horizon = max(horizons)
        extended_end = pd.Timestamp(end_date) + pd.Timedelta(days=max_horizon * 2)  # Buffer
        
        # Build universe set that includes the benchmark
        query_universe = None
        if universe:
            query_universe = set(universe)
            # Always include benchmark in the query to compute excess returns
            query_universe.add(benchmark_asset_id)
        
        bars_df = self.api.get_bars_asof(extended_end.date(), universe=query_universe)
        
        if len(bars_df) == 0:
            return pd.DataFrame(columns=[
                'date', 'asset_id', 'horizon', 'target_log_return',
                'target_excess_log_return', 'target_up'
            ])
        
        # Get benchmark bars
        benchmark_bars = bars_df[bars_df['asset_id'] == benchmark_asset_id].copy()
        benchmark_bars = benchmark_bars.set_index('date').sort_index()
        
        # Generate labels for each asset
        all_labels = []
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            if asset_id == benchmark_asset_id:
                continue  # Skip benchmark itself
            
            asset_bars = asset_bars.set_index('date').sort_index()
            asset_prices = asset_bars['adj_close']
            
            # Compute log returns for each horizon
            for horizon in horizons:
                # Forward shift to get future prices
                future_prices = asset_prices.shift(-horizon)
                log_returns = np.log(future_prices / asset_prices)
                
                # Compute excess returns vs benchmark
                benchmark_prices = benchmark_bars['adj_close']
                benchmark_future = benchmark_prices.shift(-horizon)
                benchmark_returns = np.log(benchmark_future / benchmark_prices)
                
                # Align and compute excess
                aligned = pd.DataFrame({
                    'asset_return': log_returns,
                    'benchmark_return': benchmark_returns
                }).dropna()
                
                aligned['excess_return'] = aligned['asset_return'] - aligned['benchmark_return']
                
                # Filter to date range
                aligned = aligned[
                    (aligned.index >= pd.Timestamp(start_date)) &
                    (aligned.index <= pd.Timestamp(end_date))
                ]
                
                # Create label records
                for date_idx, row in aligned.iterrows():
                    all_labels.append({
                        'date': date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                        'asset_id': asset_id,
                        'horizon': horizon,
                        'target_log_return': row['asset_return'],
                        'target_excess_log_return': row['excess_return'],
                        'target_up': 1 if row['excess_return'] > 0 else 0
                    })
        
        if len(all_labels) == 0:
            return pd.DataFrame(columns=[
                'date', 'asset_id', 'horizon', 'target_log_return',
                'target_excess_log_return', 'target_up'
            ])
        
        labels_df = pd.DataFrame(all_labels)
        
        # Remove any rows with NaN (where we don't have future prices)
        labels_df = labels_df.dropna()
        
        return labels_df
    
    def save_labels(self, labels_df: pd.DataFrame, table_name: str = "labels_returns"):
        """Save labels to storage."""
        self.storage.save_parquet(labels_df, table_name)
        # Also save to DuckDB if table exists
        try:
            self.storage.insert_dataframe(table_name, labels_df, if_exists="append")
        except Exception:
            # Table might not exist, create it
            self.storage.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    date DATE NOT NULL,
                    asset_id INTEGER NOT NULL,
                    horizon INTEGER NOT NULL,
                    target_log_return DOUBLE,
                    target_excess_log_return DOUBLE,
                    target_up INTEGER,
                    PRIMARY KEY (date, asset_id, horizon)
                )
            """)
            self.storage.insert_dataframe(table_name, labels_df, if_exists="append")

