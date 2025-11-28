"""Triple-barrier event-based labels (Lopez de Prado style)."""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import date
from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI


class TripleBarrierLabelGenerator:
    """
    Generates triple-barrier labels for event-based trading.
    
    For each time t, defines:
    - Upper barrier: +X%
    - Lower barrier: -Y%
    - Time barrier: h days
    
    Label is based on which barrier hits first.
    """
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.api = AsOfQueryAPI(storage)
    
    def generate_labels(
        self,
        start_date: date,
        end_date: date,
        upper_barrier_pct: float = 0.05,  # 5%
        lower_barrier_pct: float = 0.03,   # 3%
        time_barrier_days: int = 20,
        universe: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate triple-barrier labels.
        
        Returns:
            DataFrame with columns: date, asset_id, label, hit_barrier, 
            holding_period, max_excursion, min_excursion
        """
        # Get bars with sufficient lookahead
        extended_end = pd.Timestamp(end_date) + pd.Timedelta(days=time_barrier_days * 2)
        bars_df = self.api.get_bars_asof(extended_end.date(), universe=set(universe) if universe else None)
        
        if len(bars_df) == 0:
            return pd.DataFrame(columns=[
                'date', 'asset_id', 'label', 'hit_barrier',
                'holding_period', 'max_excursion', 'min_excursion'
            ])
        
        all_labels = []
        
        for asset_id, asset_bars in bars_df.groupby('asset_id'):
            asset_bars = asset_bars.set_index('date').sort_index()
            prices = asset_bars['adj_close']
            
            # Filter to date range
            asset_bars = asset_bars[
                (asset_bars.index >= pd.Timestamp(start_date)) &
                (asset_bars.index <= pd.Timestamp(end_date))
            ]
            
            for entry_date, entry_price in asset_bars['adj_close'].items():
                # Look forward to see which barrier hits first
                future_bars = asset_bars[asset_bars.index > entry_date]
                
                if len(future_bars) == 0:
                    continue
                
                # Compute price excursions
                future_prices = future_bars['adj_close']
                max_price = future_prices.max()
                min_price = future_prices.min()
                
                max_excursion = (max_price - entry_price) / entry_price
                min_excursion = (min_price - entry_price) / entry_price
                
                # Check barriers
                hit_barrier = None
                label = 0  # Timeout/default
                holding_period = None
                
                # Check upper barrier
                upper_hit = future_prices[future_prices >= entry_price * (1 + upper_barrier_pct)]
                if len(upper_hit) > 0:
                    upper_hit_date = upper_hit.index[0]
                    upper_days = (upper_hit_date - entry_date).days
                    
                    # Check lower barrier
                    lower_hit = future_prices[future_prices <= entry_price * (1 - lower_barrier_pct)]
                    if len(lower_hit) > 0:
                        lower_hit_date = lower_hit.index[0]
                        lower_days = (lower_hit_date - entry_date).days
                        
                        # Which hits first?
                        if upper_days < lower_days:
                            hit_barrier = "upper"
                            label = 1
                            holding_period = upper_days
                        else:
                            hit_barrier = "lower"
                            label = -1
                            holding_period = lower_days
                    else:
                        # Only upper hit
                        hit_barrier = "upper"
                        label = 1
                        holding_period = upper_days
                else:
                    # Check lower barrier
                    lower_hit = future_prices[future_prices <= entry_price * (1 - lower_barrier_pct)]
                    if len(lower_hit) > 0:
                        lower_hit_date = lower_hit.index[0]
                        hit_barrier = "lower"
                        label = -1
                        holding_period = (lower_hit_date - entry_date).days
                
                # Check time barrier
                if holding_period is None or holding_period > time_barrier_days:
                    # Timeout
                    hit_barrier = "time"
                    label = 0
                    holding_period = time_barrier_days
                    # Use price at time barrier
                    barrier_date = entry_date + pd.Timedelta(days=time_barrier_days)
                    if barrier_date in future_bars.index:
                        barrier_price = future_bars.loc[barrier_date, 'adj_close']
                        final_excursion = (barrier_price - entry_price) / entry_price
                        if final_excursion > 0:
                            label = 1
                        else:
                            label = -1
                
                all_labels.append({
                    'date': entry_date.date() if hasattr(entry_date, 'date') else entry_date,
                    'asset_id': asset_id,
                    'label': label,
                    'hit_barrier': hit_barrier,
                    'holding_period': holding_period,
                    'max_excursion': max_excursion,
                    'min_excursion': min_excursion
                })
        
        if len(all_labels) == 0:
            return pd.DataFrame(columns=[
                'date', 'asset_id', 'label', 'hit_barrier',
                'holding_period', 'max_excursion', 'min_excursion'
            ])
        
        return pd.DataFrame(all_labels)

