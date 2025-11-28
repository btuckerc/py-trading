"""Universe definition and calendar utilities."""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from datetime import datetime, date
import pandas_market_calendars as mcal
from loguru import logger


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


def build_membership_from_csv(
    storage,
    csv_path: str = "data/sp500_constituents.csv",
    index_name: str = "SP500",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Build universe_membership table from historical constituents CSV.
    
    This is a reusable function that can be called from scripts or other modules.
    
    Args:
        storage: StorageBackend instance
        csv_path: Path to constituents CSV file
        index_name: Name for the index (e.g., "SP500")
        start_date: Start date for membership (if None, uses earliest in CSV)
        end_date: End date for membership (if None, uses today)
        overwrite: If True, delete existing membership for this index first
    
    Returns:
        Dict with status, counts, and any errors
    """
    from data.normalize import DataNormalizer
    
    result = {
        'success': False,
        'records_created': 0,
        'unique_assets': 0,
        'date_range': None,
        'errors': []
    }
    
    # Initialize universe manager
    universe_manager = UniverseManager(db_path=storage.db_path if hasattr(storage, 'db_path') else str(storage.conn))
    
    # Load constituents CSV
    logger.info(f"Loading constituents from {csv_path}")
    constituents_df = universe_manager.load_sp500_constituents(csv_path)
    
    if len(constituents_df) == 0:
        result['errors'].append(f"No constituents data found in {csv_path}")
        logger.error(result['errors'][-1])
        return result
    
    logger.info(f"Loaded {len(constituents_df)} constituent changes")
    
    # Determine date range
    if start_date is None:
        start_date = constituents_df['date'].min()
    if end_date is None:
        end_date = max(constituents_df['date'].max(), date.today())
    
    result['date_range'] = (str(start_date), str(end_date))
    logger.info(f"Building universe membership from {start_date} to {end_date}")
    
    # Build membership DataFrame (with symbols)
    membership_df = universe_manager.build_universe_membership(
        start_date=start_date,
        end_date=end_date,
        index_name=index_name
    )
    
    if len(membership_df) == 0:
        result['errors'].append("No membership records generated. Check CSV format and date range.")
        logger.warning(result['errors'][-1])
        return result
    
    logger.info(f"Generated {len(membership_df)} membership records")
    
    # Map symbols to asset_ids
    if 'symbol' in membership_df.columns:
        logger.info("Mapping symbols to asset_ids...")
        
        # Get existing symbol -> asset_id mapping
        try:
            assets_df = storage.query("SELECT asset_id, symbol FROM assets")
            symbol_to_asset_id = dict(zip(assets_df['symbol'], assets_df['asset_id'])) if len(assets_df) > 0 else {}
        except Exception:
            symbol_to_asset_id = {}
        
        # Get unique symbols from membership
        unique_symbols = set(membership_df['symbol'].unique())
        
        # Create asset_ids for symbols that don't exist yet
        missing_symbols = unique_symbols - set(symbol_to_asset_id.keys())
        if len(missing_symbols) > 0:
            logger.info(f"Creating asset_ids for {len(missing_symbols)} new symbols...")
            normalizer = DataNormalizer(storage)
            for symbol in missing_symbols:
                normalizer._get_or_create_asset_id(symbol)
            
            # Reload mapping
            assets_df = storage.query("SELECT asset_id, symbol FROM assets")
            symbol_to_asset_id = dict(zip(assets_df['symbol'], assets_df['asset_id']))
        
        # Replace symbol with asset_id
        membership_df['asset_id'] = membership_df['symbol'].map(symbol_to_asset_id)
        membership_df = membership_df.drop(columns=['symbol'])
    
    # Remove rows with missing asset_ids
    before_count = len(membership_df)
    membership_df = membership_df.dropna(subset=['asset_id'])
    if len(membership_df) < before_count:
        dropped = before_count - len(membership_df)
        logger.warning(f"Removed {dropped} rows with missing asset_ids")
        result['errors'].append(f"Dropped {dropped} rows with missing asset_ids")
    
    # Ensure asset_id is integer
    membership_df['asset_id'] = membership_df['asset_id'].astype(int)
    
    # Select final columns
    required_columns = ['date', 'asset_id', 'index_name', 'in_index']
    for col in required_columns:
        if col not in membership_df.columns:
            result['errors'].append(f"Missing required column: {col}")
            logger.error(result['errors'][-1])
            return result
    
    membership_df = membership_df[required_columns]
    
    logger.info(f"Final membership table: {len(membership_df)} records")
    logger.info(f"  Date range: {membership_df['date'].min()} to {membership_df['date'].max()}")
    logger.info(f"  Unique assets: {membership_df['asset_id'].nunique()}")
    
    # Save to database
    if overwrite:
        logger.info(f"Overwriting existing universe_membership for {index_name}...")
        storage.conn.execute("DELETE FROM universe_membership WHERE index_name = ?", [index_name])
    
    logger.info("Inserting into universe_membership table...")
    storage.insert_dataframe('universe_membership', membership_df, if_exists='append')
    
    # Also save to Parquet
    logger.info("Saving to Parquet...")
    storage.save_parquet(membership_df, 'universe_membership')
    
    result['success'] = True
    result['records_created'] = len(membership_df)
    result['unique_assets'] = membership_df['asset_id'].nunique()
    
    logger.info("Universe membership table built successfully!")
    
    return result


def refresh_universe_membership(
    storage,
    config: Optional[Dict[str, Any]] = None,
    mode: str = "rebuild",
    index_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    fetch_missing_data: bool = True,
    vendor: str = "yahoo"
) -> Dict[str, Any]:
    """
    Refresh universe membership from config and optionally fetch missing price data.
    
    This is the main entry point for universe maintenance.
    
    Args:
        storage: StorageBackend instance
        config: Config dict (if None, will use get_config())
        mode: "rebuild" (truncate + full rebuild) or "incremental" (append new)
        index_name: Override index name from config
        csv_path: Override CSV path from config
        fetch_missing_data: If True, fetch price data for symbols without any
        vendor: Data vendor to use for fetching (default: yahoo)
    
    Returns:
        Dict with status and details
    """
    if config is None:
        from configs.loader import get_config
        config = get_config()
    
    # Get values from config with fallbacks
    if hasattr(config, 'universe'):
        universe_config = config.universe
        default_index = getattr(universe_config, 'index_name', 'SP500')
        default_csv = getattr(universe_config, 'constituents_csv_path', 'data/sp500_constituents.csv')
    elif isinstance(config, dict):
        universe_config = config.get('universe', {})
        default_index = universe_config.get('index_name', 'SP500')
        default_csv = universe_config.get('constituents_csv_path', 'data/sp500_constituents.csv')
    else:
        default_index = 'SP500'
        default_csv = 'data/sp500_constituents.csv'
    
    index_name = index_name or default_index
    csv_path = csv_path or default_csv
    
    overwrite = (mode == "rebuild")
    
    # Build membership
    result = build_membership_from_csv(
        storage=storage,
        csv_path=csv_path,
        index_name=index_name,
        overwrite=overwrite
    )
    
    # Optionally fetch missing price data
    if fetch_missing_data and result['success']:
        fetch_result = ensure_universe_data_coverage(
            storage=storage,
            config=config,
            index_name=index_name,
            vendor=vendor
        )
        result['data_fetch'] = fetch_result
    
    return result


def ensure_universe_data_coverage(
    storage,
    config: Optional[Dict[str, Any]] = None,
    index_name: str = "SP500",
    vendor: str = "yahoo",
    start_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Ensure all universe members have price data.
    
    Checks for symbols in the universe that don't have any price data
    and fetches historical data for them.
    
    Args:
        storage: StorageBackend instance
        config: Config dict
        index_name: Universe index name
        vendor: Data vendor to use
        start_date: Start date for data fetch (if None, uses config min_history_start_date)
    
    Returns:
        Dict with fetch results
    """
    from data.maintenance import ensure_data_coverage
    
    result = {
        'success': False,
        'symbols_checked': 0,
        'symbols_missing_data': 0,
        'symbols_fetched': 0,
        'errors': []
    }
    
    # Get universe symbols
    try:
        universe_symbols = storage.query(f"""
            SELECT DISTINCT a.symbol
            FROM universe_membership um
            JOIN assets a ON um.asset_id = a.asset_id
            WHERE um.index_name = '{index_name}'
            ORDER BY a.symbol
        """)['symbol'].tolist()
    except Exception as e:
        result['errors'].append(f"Failed to get universe symbols: {e}")
        return result
    
    result['symbols_checked'] = len(universe_symbols)
    
    if not universe_symbols:
        result['success'] = True
        return result
    
    # Find symbols without price data
    try:
        symbols_with_data = storage.query("""
            SELECT DISTINCT a.symbol
            FROM assets a
            JOIN bars_daily bd ON a.asset_id = bd.asset_id
        """)['symbol'].tolist()
        symbols_with_data_set = set(symbols_with_data)
    except Exception:
        symbols_with_data_set = set()
    
    missing_symbols = [s for s in universe_symbols if s not in symbols_with_data_set]
    result['symbols_missing_data'] = len(missing_symbols)
    
    if not missing_symbols:
        logger.info(f"All {len(universe_symbols)} universe symbols have price data")
        result['success'] = True
        return result
    
    logger.info(f"Found {len(missing_symbols)} universe symbols without price data")
    logger.info(f"Fetching data for: {missing_symbols[:10]}{'...' if len(missing_symbols) > 10 else ''}")
    
    # Determine start date
    if start_date is None:
        if config is not None:
            if hasattr(config, 'data'):
                start_date = getattr(config.data, 'min_history_start_date', date(2020, 1, 1))
            elif isinstance(config, dict) and 'data' in config:
                start_date = config['data'].get('min_history_start_date', date(2020, 1, 1))
            else:
                start_date = date(2020, 1, 1)
        else:
            start_date = date(2020, 1, 1)
    
    # Convert string date if needed
    if isinstance(start_date, str):
        from datetime import datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    
    # Fetch data using maintenance module
    try:
        fetch_result = ensure_data_coverage(
            storage=storage,
            config=config.__dict__ if hasattr(config, '__dict__') else config,
            mode="custom",
            target_start=start_date,
            target_end=date.today(),
            symbols=missing_symbols,
            vendor=vendor,
            auto_fetch=True
        )
        
        result['fetch_result'] = fetch_result
        result['success'] = fetch_result.get('status') in ['ok', 'partial']
        
        # Count successfully fetched symbols
        if fetch_result.get('coverage_after'):
            after_count = fetch_result['coverage_after'].get('num_assets', 0)
            before_count = fetch_result.get('coverage_before', {}).get('num_assets', 0)
            result['symbols_fetched'] = after_count - before_count
        
    except Exception as e:
        result['errors'].append(f"Failed to fetch data: {e}")
        logger.error(f"Failed to fetch data for missing symbols: {e}")
    
    return result

