"""Universe definition and calendar utilities."""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime, date
import pandas_market_calendars as mcal


class TradingCalendar:
    """Trading calendar for NYSE/EQUITY markets."""
    
    def __init__(self):
        self.nyse = mcal.get_calendar('NYSE')
    
    def get_trading_days(self, start_date: date, end_date: date) -> pd.DatetimeIndex:
        """Get all trading days between start and end dates."""
        schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
        return schedule.index
    
    def is_trading_day(self, check_date: date) -> bool:
        """Check if a date is a trading day."""
        schedule = self.nyse.schedule(start_date=check_date, end_date=check_date)
        return len(schedule) > 0
    
    def next_trading_day(self, from_date: date) -> date:
        """Get the next trading day after from_date."""
        schedule = self.nyse.schedule(
            start_date=from_date,
            end_date=pd.Timestamp(from_date) + pd.Timedelta(days=10)
        )
        if len(schedule) > 0:
            return schedule.index[0].date()
        raise ValueError(f"No trading days found after {from_date}")
    
    def previous_trading_day(self, from_date: date) -> date:
        """Get the previous trading day before from_date."""
        schedule = self.nyse.schedule(
            start_date=pd.Timestamp(from_date) - pd.Timedelta(days=10),
            end_date=from_date
        )
        if len(schedule) > 0:
            return schedule.index[-1].date()
        raise ValueError(f"No trading days found before {from_date}")


class UniverseManager:
    """Manages universe membership over time."""
    
    def __init__(self, db_path: str = "data/market.duckdb"):
        self.db_path = db_path
        self.calendar = TradingCalendar()
    
    def load_sp500_constituents(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load S&P 500 historical constituents.
        
        Expected CSV format:
        - date: date when membership changed
        - symbol: ticker symbol
        - action: 'added' or 'removed'
        """
        if csv_path is None:
            csv_path = "data/sp500_constituents.csv"
        
        path = Path(csv_path)
        if not path.exists():
            # Return empty DataFrame with expected schema
            return pd.DataFrame(columns=['date', 'symbol', 'action'])
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    
    def build_universe_membership(
        self,
        start_date: date,
        end_date: date,
        index_name: str = "SP500"
    ) -> pd.DataFrame:
        """
        Build universe_membership table from historical constituents.
        
        Returns DataFrame with columns: date, asset_id, index_name, in_index
        """
        constituents_df = self.load_sp500_constituents()
        
        if len(constituents_df) == 0:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=['date', 'asset_id', 'index_name', 'in_index'])
        
        # Get all trading days
        trading_days = self.calendar.get_trading_days(start_date, end_date)
        
        # Build membership over time
        membership_records = []
        current_members: Set[str] = set()
        
        # Sort by date
        constituents_df = constituents_df.sort_values('date')
        
        for trading_day in trading_days:
            trading_date = trading_day.date()
            
            # Process all changes up to this date
            changes = constituents_df[constituents_df['date'] <= trading_date]
            for _, row in changes.iterrows():
                if row['action'] == 'added':
                    current_members.add(row['symbol'])
                elif row['action'] == 'removed':
                    current_members.discard(row['symbol'])
            
            # Record membership for this date
            for symbol in current_members:
                membership_records.append({
                    'date': trading_date,
                    'symbol': symbol,
                    'index_name': index_name,
                    'in_index': True
                })
        
        membership_df = pd.DataFrame(membership_records)
        
        # Return with symbol column - will be mapped to asset_id later
        return membership_df
    
    def get_universe_at_date(self, as_of_date: date, index_name: str = "SP500") -> Set[int]:
        """
        Get set of asset_ids in universe at a specific date.
        
        Queries the universe_membership table in DuckDB.
        
        Returns:
            Set of asset_ids that were in the index on as_of_date
        """
        import duckdb
        
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            result = conn.execute("""
                SELECT DISTINCT asset_id
                FROM universe_membership
                WHERE date = ? AND index_name = ? AND in_index = TRUE
            """, [as_of_date, index_name]).fetchdf()
            conn.close()
            
            if len(result) > 0:
                return set(result['asset_id'].values)
            return set()
        except Exception:
            # Table may not exist or be empty
            return set()
    
    def filter_universe(
        self,
        symbols: Set[str],
        as_of_date: date,
        min_price: Optional[float] = None,
        min_dollar_volume: Optional[float] = None,
        bars_df: Optional[pd.DataFrame] = None
    ) -> Set[str]:
        """
        Filter universe by price and volume constraints.
        
        Args:
            symbols: Set of candidate symbols
            as_of_date: Date to check constraints
            min_price: Minimum price threshold
            min_dollar_volume: Minimum dollar volume threshold
            bars_df: DataFrame with price/volume data (columns: date, symbol, close, volume)
        
        Returns:
            Filtered set of symbols
        """
        if bars_df is None or len(bars_df) == 0:
            return symbols
        
        filtered = symbols.copy()
        
        # Filter by price
        if min_price is not None:
            date_bars = bars_df[bars_df['date'] == as_of_date]
            if len(date_bars) > 0:
                price_filter = date_bars[date_bars['close'] >= min_price]['symbol'].values
                filtered = filtered.intersection(set(price_filter))
        
        # Filter by dollar volume
        if min_dollar_volume is not None:
            date_bars = bars_df[bars_df['date'] == as_of_date]
            if len(date_bars) > 0:
                date_bars = date_bars.copy()
                date_bars['dollar_volume'] = date_bars['close'] * date_bars['volume']
                volume_filter = date_bars[date_bars['dollar_volume'] >= min_dollar_volume]['symbol'].values
                filtered = filtered.intersection(set(volume_filter))
        
        return filtered
    
    def validate_trades_against_universe(
        self,
        trades_df: pd.DataFrame,
        index_name: str = "SP500"
    ) -> dict:
        """
        Validate that all trades are for assets that were in the universe at trade time.
        
        Args:
            trades_df: DataFrame with columns: date, asset_id
            index_name: Index to check membership against
        
        Returns:
            Dict with 'valid': bool, 'violations': list of dicts with date, asset_id
        """
        import duckdb
        
        violations = []
        
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            
            for _, row in trades_df.iterrows():
                trade_date = row['date']
                asset_id = row['asset_id']
                
                # Check if asset was in universe on trade date
                result = conn.execute("""
                    SELECT COUNT(*) as cnt
                    FROM universe_membership
                    WHERE date = ? AND asset_id = ? AND index_name = ? AND in_index = TRUE
                """, [trade_date, int(asset_id), index_name]).fetchone()
                
                if result[0] == 0:
                    violations.append({
                        'date': trade_date,
                        'asset_id': asset_id,
                        'issue': 'Asset not in universe on trade date'
                    })
            
            conn.close()
            
            return {
                'valid': len(violations) == 0,
                'violations': violations,
                'total_trades': len(trades_df),
                'invalid_trades': len(violations)
            }
        except Exception as e:
            return {
                'valid': False,
                'violations': [],
                'error': str(e),
                'total_trades': len(trades_df),
                'invalid_trades': -1
            }
    
    def get_universe_coverage(
        self,
        start_date: date,
        end_date: date,
        index_name: str = "SP500"
    ) -> pd.DataFrame:
        """
        Get universe membership counts over a date range.
        
        Returns:
            DataFrame with date and member_count columns
        """
        import duckdb
        
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            result = conn.execute("""
                SELECT date, COUNT(DISTINCT asset_id) as member_count
                FROM universe_membership
                WHERE date >= ? AND date <= ? AND index_name = ? AND in_index = TRUE
                GROUP BY date
                ORDER BY date
            """, [start_date, end_date, index_name]).fetchdf()
            conn.close()
            return result
        except Exception:
            return pd.DataFrame(columns=['date', 'member_count'])

