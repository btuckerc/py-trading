"""Point-in-time as-of query API."""

import pandas as pd
from typing import Optional, Set
from datetime import date
from .storage import StorageBackend


class AsOfQueryAPI:
    """
    Provides point-in-time queries that ensure no lookahead bias.
    
    All queries return data as-of a specific date, using only information
    that would have been available at that time.
    """
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
    
    def get_bars_asof(
        self,
        as_of_date: date,
        lookback_days: Optional[int] = None,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Get bars up to and including as_of_date.
        
        Args:
            as_of_date: Date to query as-of
            lookback_days: If provided, only return this many days of history
            universe: If provided, filter to these asset_ids
        
        Returns:
            DataFrame with columns: asset_id, date, open, high, low, close, adj_close, volume
        """
        query = """
            SELECT asset_id, date, open, high, low, close, adj_close, volume
            FROM bars_daily
            WHERE date <= ?
        """
        params = [as_of_date]
        
        if universe is not None:
            asset_list = ",".join(map(str, universe))
            query += f" AND asset_id IN ({asset_list})"
        
        query += " ORDER BY asset_id, date"
        
        # DuckDB parameterized queries
        bars_df = self.storage.conn.execute(query, params).df()
        
        if lookback_days is not None:
            # Filter to last lookback_days per asset
            result = []
            for asset_id, asset_bars in bars_df.groupby('asset_id'):
                asset_bars = asset_bars.sort_values('date')
                if len(asset_bars) > lookback_days:
                    asset_bars = asset_bars.tail(lookback_days)
                result.append(asset_bars)
            if len(result) > 0:
                bars_df = pd.concat(result, ignore_index=True)
        
        return bars_df
    
    def get_fundamentals_asof(
        self,
        as_of_date: date,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Get last-known fundamentals for each asset as-of as_of_date.
        
        Uses report_release_date to ensure point-in-time correctness:
        only fundamentals with report_release_date <= as_of_date are considered.
        
        Returns:
            DataFrame with columns: asset_id, period_end_date, report_release_date, eps, revenue, etc.
        """
        query = """
            SELECT DISTINCT ON (asset_id)
                asset_id, period_end_date, report_release_date,
                eps, eps_estimate, revenue, ebitda, total_debt,
                total_equity, free_cash_flow
            FROM fundamentals
            WHERE report_release_date <= ?
        """
        params = [as_of_date]
        
        if universe is not None:
            asset_list = ",".join(map(str, universe))
            query += f" AND asset_id IN ({asset_list})"
        
        query += " ORDER BY asset_id, report_release_date DESC"
        
        # DuckDB doesn't support DISTINCT ON, so use window function instead
        query = """
            SELECT * FROM (
                SELECT 
                    asset_id, period_end_date, report_release_date,
                    eps, eps_estimate, revenue, ebitda, total_debt,
                    total_equity, free_cash_flow,
                    ROW_NUMBER() OVER (PARTITION BY asset_id ORDER BY report_release_date DESC) as rn
                FROM fundamentals
                WHERE report_release_date <= ?
        """
        if universe is not None:
            asset_list = ",".join(map(str, universe))
            query += f" AND asset_id IN ({asset_list})"
        
        query += ") WHERE rn = 1"
        
        return self.storage.conn.execute(query, params).df()
    
    def get_news_asof(
        self,
        as_of_date: date,
        lookback_window_days: int = 30,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Get news events up to and including as_of_date.
        
        Args:
            as_of_date: Date to query as-of
            lookback_window_days: Only return news within this many days
            universe: If provided, filter to these asset_ids
        
        Returns:
            DataFrame with columns: asset_id, timestamp, headline, source, url, vendor_sentiment_score
        """
        from datetime import timedelta
        start_date = as_of_date - timedelta(days=lookback_window_days)
        
        query = """
            SELECT asset_id, timestamp, headline, source, url, vendor_sentiment_score
            FROM news_events
            WHERE DATE(timestamp) >= ? AND DATE(timestamp) <= ?
        """
        params = [start_date, as_of_date]
        
        if universe is not None:
            asset_list = ",".join(map(str, universe))
            query += f" AND asset_id IN ({asset_list})"
        
        query += " ORDER BY asset_id, timestamp"
        
        return self.storage.conn.execute(query, params).df()
    
    def get_universe_at_date(
        self,
        as_of_date: date,
        index_name: str = "SP500"
    ) -> Set[int]:
        """
        Get set of asset_ids in universe at as_of_date.
        
        Returns:
            Set of asset_ids
        """
        query = """
            SELECT DISTINCT asset_id
            FROM universe_membership
            WHERE date = ? AND index_name = ? AND in_index = TRUE
        """
        params = [as_of_date, index_name]
        
        result = self.storage.conn.execute(query, params).df()
        return set(result['asset_id'].values) if len(result) > 0 else set()
    
    def get_corporate_actions_asof(
        self,
        as_of_date: date,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Get corporate actions up to and including as_of_date.
        
        Returns:
            DataFrame with columns: asset_id, date, split_factor, dividend_amount, special_dividend
        """
        query = """
            SELECT asset_id, date, split_factor, dividend_amount, special_dividend
            FROM corporate_actions
            WHERE date <= ?
        """
        params = [as_of_date]
        
        if universe is not None:
            asset_list = ",".join(map(str, universe))
            query += f" AND asset_id IN ({asset_list})"
        
        query += " ORDER BY asset_id, date"
        
        return self.storage.conn.execute(query, params).df()

