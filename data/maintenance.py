"""Data maintenance module for automated coverage checks and backfills.

This module provides:
- Config-driven data coverage checking (min history date to target date)
- Automated backfill and top-up via configured vendor(s)
- Integration hooks for backtest and live workflows
"""

import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Optional, Set, Tuple, Dict, Any, Union
from pathlib import Path
from loguru import logger

from data.storage import StorageBackend
from data.normalize import DataNormalizer


class DataCoverageChecker:
    """
    Checks and reports on data coverage in bars_daily.
    
    Compares actual coverage against configured requirements and
    identifies gaps that need to be backfilled.
    """
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
    
    def get_current_coverage(self) -> Dict[str, Any]:
        """
        Get current data coverage summary from bars_daily.
        
        Returns:
            Dict with min_date, max_date, total_bars, num_assets
        """
        try:
            result = self.storage.query("""
                SELECT 
                    MIN(date) as min_date, 
                    MAX(date) as max_date, 
                    COUNT(*) as total_bars,
                    COUNT(DISTINCT asset_id) as num_assets
                FROM bars_daily
            """)
            
            if len(result) == 0 or result['max_date'].iloc[0] is None:
                return {
                    'min_date': None,
                    'max_date': None,
                    'total_bars': 0,
                    'num_assets': 0,
                    'has_data': False
                }
            
            min_date = result['min_date'].iloc[0]
            max_date = result['max_date'].iloc[0]
            
            # Convert to date objects if needed
            if hasattr(min_date, 'date'):
                min_date = min_date.date()
            if hasattr(max_date, 'date'):
                max_date = max_date.date()
            
            return {
                'min_date': min_date,
                'max_date': max_date,
                'total_bars': int(result['total_bars'].iloc[0]),
                'num_assets': int(result['num_assets'].iloc[0]),
                'has_data': True
            }
        except Exception as e:
            logger.warning(f"Error getting coverage: {e}")
            return {
                'min_date': None,
                'max_date': None,
                'total_bars': 0,
                'num_assets': 0,
                'has_data': False
            }
    
    def get_per_asset_coverage(self, universe: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Get per-asset coverage details.
        
        Args:
            universe: Optional set of asset_ids to filter
        
        Returns:
            DataFrame with asset_id, symbol, min_date, max_date, bar_count
        """
        query = """
            SELECT 
                b.asset_id,
                a.symbol,
                MIN(b.date) as min_date,
                MAX(b.date) as max_date,
                COUNT(*) as bar_count
            FROM bars_daily b
            LEFT JOIN assets a ON b.asset_id = a.asset_id
        """
        
        if universe is not None and len(universe) > 0:
            asset_list = ",".join(map(str, universe))
            query += f" WHERE b.asset_id IN ({asset_list})"
        
        query += " GROUP BY b.asset_id, a.symbol ORDER BY a.symbol"
        
        return self.storage.query(query)
    
    def identify_gaps(
        self,
        target_start: date,
        target_end: date,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify data gaps that need to be filled.
        
        Args:
            target_start: Desired start date for coverage
            target_end: Desired end date for coverage
            symbols: Optional list of symbols to check (if None, checks all)
        
        Returns:
            Dict with:
                - needs_backfill: bool
                - backfill_ranges: List of (start, end) date tuples
                - missing_symbols: List of symbols with no data
                - partial_symbols: Dict of symbol -> missing date ranges
        """
        coverage = self.get_current_coverage()
        
        result = {
            'needs_backfill': False,
            'backfill_ranges': [],
            'missing_symbols': [],
            'partial_symbols': {},
            'coverage': coverage
        }
        
        if not coverage['has_data']:
            # No data at all - need full backfill
            result['needs_backfill'] = True
            result['backfill_ranges'] = [(target_start, target_end)]
            if symbols:
                result['missing_symbols'] = symbols
            return result
        
        # Ensure we have valid date objects for comparison
        min_date = coverage['min_date']
        max_date = coverage['max_date']
        
        # Handle pandas NaT, None, or invalid dates
        try:
            if min_date is None or pd.isna(min_date) or max_date is None or pd.isna(max_date):
                result['needs_backfill'] = True
                result['backfill_ranges'] = [(target_start, target_end)]
                if symbols:
                    result['missing_symbols'] = symbols
                return result
        except (TypeError, ValueError):
            # If comparison fails, treat as no data
            result['needs_backfill'] = True
            result['backfill_ranges'] = [(target_start, target_end)]
            if symbols:
                result['missing_symbols'] = symbols
            return result
        
        # Check for gaps at the start
        if target_start < min_date:
            result['needs_backfill'] = True
            result['backfill_ranges'].append((target_start, min_date - timedelta(days=1)))
        
        # Check for gaps at the end
        if target_end > max_date:
            result['needs_backfill'] = True
            result['backfill_ranges'].append((max_date + timedelta(days=1), target_end))
        
        # Check for missing symbols (symbols with NO price data, not just missing from assets)
        if symbols:
            # Get symbols that have actual price data in bars_daily
            symbols_with_data = self.storage.query("""
                SELECT DISTINCT a.symbol 
                FROM assets a
                JOIN bars_daily bd ON a.asset_id = bd.asset_id
            """)
            symbols_with_data_set = set(symbols_with_data['symbol'].values) if len(symbols_with_data) > 0 else set()
            
            # Find symbols that don't have any price data
            missing = [s for s in symbols if s not in symbols_with_data_set]
            if missing:
                result['needs_backfill'] = True
                result['missing_symbols'] = missing
        
        return result


class DataBackfiller:
    """
    Handles fetching and storing missing data from configured vendors.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        vendor: str = "yahoo",
        vendor_client=None
    ):
        self.storage = storage
        self.vendor = vendor
        self._vendor_client = vendor_client
    
    @property
    def vendor_client(self):
        """Lazy-load vendor client."""
        if self._vendor_client is None:
            self._vendor_client = self._create_vendor_client(self.vendor)
        return self._vendor_client
    
    def _create_vendor_client(self, vendor: str):
        """Create vendor client based on vendor name."""
        if vendor == "yahoo":
            from data.vendors.yahoo import YahooClient
            return YahooClient()
        elif vendor == "tiingo":
            from data.vendors.tiingo import TiingoClient
            return TiingoClient()
        else:
            raise ValueError(f"Unknown vendor: {vendor}")
    
    def fetch_and_store(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Fetch data for symbols and date range, then store in DuckDB and Parquet.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict with fetch results (success, bars_fetched, errors)
        """
        result = {
            'success': False,
            'bars_fetched': 0,
            'symbols_fetched': 0,
            'errors': []
        }
        
        if not symbols:
            result['errors'].append("No symbols provided")
            return result
        
        logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date} via {self.vendor}")
        
        try:
            # Fetch bars from vendor
            bars_df = self.vendor_client.fetch_daily_bars(symbols, start_date, end_date)
            
            if len(bars_df) == 0:
                logger.warning("No bars fetched from vendor")
                result['errors'].append("No bars returned from vendor")
                return result
            
            logger.info(f"Fetched {len(bars_df)} bars from {self.vendor}")
            
            # Normalize
            normalizer = DataNormalizer(self.storage, vendor_client=self.vendor_client)
            normalized_bars = normalizer.normalize_bars(bars_df, vendor=self.vendor)
            
            if len(normalized_bars) == 0:
                result['errors'].append("Normalization produced no records")
                return result
            
            logger.info(f"Normalized to {len(normalized_bars)} records")
            
            # Save to Parquet
            self.storage.save_parquet(normalized_bars, "bars_daily")
            
            # Insert into DuckDB (append mode, will skip duplicates due to PK)
            self.storage.insert_dataframe("bars_daily", normalized_bars, if_exists="append")
            
            result['success'] = True
            result['bars_fetched'] = len(normalized_bars)
            result['symbols_fetched'] = len(bars_df['symbol'].unique())
            
            logger.info(f"Stored {result['bars_fetched']} bars for {result['symbols_fetched']} symbols")
            
        except Exception as e:
            logger.error(f"Error in fetch_and_store: {e}")
            result['errors'].append(str(e))
        
        return result


class DataMaintenanceManager:
    """
    High-level manager for data maintenance operations.
    
    Coordinates coverage checking and backfilling based on config.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        config: Optional[Dict[str, Any]] = None
    ):
        self.storage = storage
        self.config = config or {}
        self.checker = DataCoverageChecker(storage)
        
        # Extract config values with defaults
        # Handle both dict and Pydantic model configs
        data_config = self._get_config_section('data', {})
        self.min_history_start_date = self._parse_date(
            self._get_from_config(data_config, 'min_history_start_date', '2020-01-01')
        )
        self.max_history_lag_days = self._get_from_config(data_config, 'max_history_lag_days', 1)
        vendors_config = self._get_config_section('vendors', {})
        self.default_vendor = self._get_from_config(vendors_config, 'primary', 'yahoo')
    
    def _get_config_section(self, key: str, default: Any) -> Any:
        """Get a config section, handling both dict and Pydantic model."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        elif hasattr(self.config, key):
            return getattr(self.config, key)
        return default
    
    def _get_from_config(self, config_section: Any, key: str, default: Any) -> Any:
        """Get a value from a config section, handling both dict and Pydantic model."""
        if isinstance(config_section, dict):
            return config_section.get(key, default)
        elif hasattr(config_section, key):
            return getattr(config_section, key)
        return default
    
    def _parse_date(self, date_val) -> date:
        """Parse date from string or return as-is if already date."""
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            return datetime.strptime(date_val, "%Y-%m-%d").date()
        return date_val
    
    def get_symbols_from_config(self, include_universe: bool = True) -> List[str]:
        """
        Get list of symbols from config, universe membership, CSV, or database.
        
        Priority:
        1. Config-specified symbol list (if any)
        2. Universe membership symbols (if include_universe=True and universe exists in DB)
        3. Universe constituents CSV file (bootstrap case - empty DB)
        4. All symbols with price data in bars_daily
        
        Args:
            include_universe: If True, include symbols from universe_membership table or CSV
        """
        # Check for config-specified symbols
        data_config = self._get_config_section('data', {})
        symbols = self._get_from_config(data_config, 'symbols', [])
        if symbols:
            return symbols
        
        # Try universe membership (survivorship-bias-free S&P 500 or other index)
        if include_universe:
            try:
                universe_config = self._get_config_section('universe', {})
                index_name = self._get_from_config(universe_config, 'index_name', 'SP500')
                
                universe_df = self.storage.query(f"""
                    SELECT DISTINCT a.symbol
                    FROM universe_membership um
                    JOIN assets a ON um.asset_id = a.asset_id
                    WHERE um.index_name = '{index_name}'
                    ORDER BY a.symbol
                """)
                if len(universe_df) > 0:
                    return universe_df['symbol'].tolist()
            except Exception:
                pass
            
            # Bootstrap case: If universe_membership is empty, read directly from CSV
            try:
                universe_config = self._get_config_section('universe', {})
                csv_path = self._get_from_config(universe_config, 'constituents_csv_path', 'data/sp500_constituents.csv')
                
                if Path(csv_path).exists():
                    import pandas as pd
                    constituents_df = pd.read_csv(csv_path)
                    if 'symbol' in constituents_df.columns:
                        symbols_from_csv = constituents_df['symbol'].unique().tolist()
                        if symbols_from_csv:
                            logger.info(f"Bootstrapping from {csv_path}: {len(symbols_from_csv)} symbols")
                            return symbols_from_csv
            except Exception as e:
                logger.debug(f"Could not read constituents CSV: {e}")
        
        # Fall back to symbols that have price data
        try:
            symbols_df = self.storage.query("""
                SELECT DISTINCT a.symbol 
                FROM assets a
                JOIN bars_daily bd ON a.asset_id = bd.asset_id
                ORDER BY a.symbol
            """)
            if len(symbols_df) > 0:
                return symbols_df['symbol'].tolist()
        except Exception:
            pass
        
        return []
    
    def ensure_coverage(
        self,
        mode: str = "full-history",
        target_start: Optional[date] = None,
        target_end: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        vendor: Optional[str] = None,
        auto_fetch: bool = True,
        bootstrap_universe: bool = True
    ) -> Dict[str, Any]:
        """
        Ensure data coverage for the specified mode and date range.
        
        Args:
            mode: One of:
                - "full-history": Ensure coverage from min_history_start_date to today
                - "daily-top-up": Ensure coverage for yesterday and today only
                - "custom": Use provided target_start and target_end
            target_start: Custom start date (required if mode="custom")
            target_end: Custom end date (optional, defaults to today)
            symbols: List of symbols (if None, uses config or all in DB)
            vendor: Data vendor to use (if None, uses config default)
            auto_fetch: If True, automatically fetch missing data
            bootstrap_universe: If True, build universe_membership after fetching data for empty DB
        
        Returns:
            Dict with status, gaps identified, and fetch results
        """
        result = {
            'mode': mode,
            'status': 'ok',
            'gaps_identified': False,
            'fetch_attempted': False,
            'fetch_result': None,
            'coverage_before': None,
            'coverage_after': None,
            'bootstrapped': False
        }
        
        # Determine target dates based on mode
        today = date.today()
        
        if mode == "full-history":
            target_start = self.min_history_start_date
            # Use provided target_end or default to today minus lag
            target_end = target_end or (today - timedelta(days=self.max_history_lag_days))
        elif mode == "daily-top-up":
            # For daily top-up, use provided target_end or default to today
            effective_end = target_end or today
            target_start = effective_end - timedelta(days=3)  # A few days buffer
            target_end = effective_end
        elif mode == "custom":
            if target_start is None:
                raise ValueError("target_start required for mode='custom'")
            target_end = target_end or today
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        result['target_start'] = target_start
        result['target_end'] = target_end
        
        # Get symbols
        if symbols is None:
            symbols = self.get_symbols_from_config()
        
        if not symbols:
            result['status'] = 'warning'
            result['message'] = "No symbols configured or found in database"
            logger.warning(result['message'])
            return result
        
        result['symbols_count'] = len(symbols)
        
        # Check current coverage
        result['coverage_before'] = self.checker.get_current_coverage()
        
        # Identify gaps
        gaps = self.checker.identify_gaps(target_start, target_end, symbols)
        result['gaps'] = gaps
        result['gaps_identified'] = gaps['needs_backfill']
        
        if not gaps['needs_backfill']:
            logger.info(f"Data coverage is complete for {target_start} to {target_end}")
            result['status'] = 'ok'
            result['message'] = "Coverage complete, no backfill needed"
            return result
        
        logger.info(f"Data gaps identified: {len(gaps['backfill_ranges'])} ranges, {len(gaps['missing_symbols'])} missing symbols")
        
        if not auto_fetch:
            result['status'] = 'gaps_found'
            result['message'] = "Gaps found but auto_fetch=False"
            return result
        
        # Perform backfill
        vendor = vendor or self.default_vendor
        backfiller = DataBackfiller(self.storage, vendor=vendor)
        
        result['fetch_attempted'] = True
        all_fetch_results = []
        
        # First, fetch full history for missing symbols (symbols with NO data at all)
        if gaps['missing_symbols']:
            # For missing symbols, always fetch from min_history_start_date regardless of mode
            # This ensures proper bootstrapping when database is empty
            fetch_start = self.min_history_start_date
            fetch_end = target_end
            
            logger.info(f"Fetching full history for {len(gaps['missing_symbols'])} symbols with no data ({fetch_start} to {fetch_end})...")
            fetch_result = backfiller.fetch_and_store(
                gaps['missing_symbols'], 
                fetch_start,
                fetch_end
            )
            all_fetch_results.append({
                'range': (str(target_start), str(target_end)),
                'type': 'missing_symbols',
                'symbols_count': len(gaps['missing_symbols']),
                'result': fetch_result
            })
        
        # Then fetch for any date gap ranges (for symbols that have partial data)
        for gap_start, gap_end in gaps['backfill_ranges']:
            # Only fetch for symbols that already have some data (not the missing ones)
            existing_symbols = [s for s in symbols if s not in gaps['missing_symbols']]
            
            if existing_symbols:
                fetch_result = backfiller.fetch_and_store(existing_symbols, gap_start, gap_end)
                all_fetch_results.append({
                    'range': (str(gap_start), str(gap_end)),
                    'type': 'gap_fill',
                    'symbols_count': len(existing_symbols),
                    'result': fetch_result
                })
        
        result['fetch_results'] = all_fetch_results
        
        # Check coverage after
        result['coverage_after'] = self.checker.get_current_coverage()
        
        # Determine overall status
        total_bars = sum(r['result']['bars_fetched'] for r in all_fetch_results)
        total_errors = sum(len(r['result']['errors']) for r in all_fetch_results)
        
        if total_bars > 0 and total_errors == 0:
            result['status'] = 'ok'
            result['message'] = f"Successfully fetched {total_bars} bars"
        elif total_bars > 0:
            result['status'] = 'partial'
            result['message'] = f"Fetched {total_bars} bars with {total_errors} errors"
        else:
            result['status'] = 'error'
            result['message'] = f"Failed to fetch data: {total_errors} errors"
        
        # Bootstrap universe_membership if this was a fresh database
        if bootstrap_universe and total_bars > 0:
            try:
                # Check if universe_membership is empty
                um_count = self.storage.query("SELECT COUNT(*) as cnt FROM universe_membership")
                if um_count['cnt'].iloc[0] == 0:
                    logger.info("Bootstrapping universe_membership from CSV...")
                    self._bootstrap_universe_membership()
                    result['bootstrapped'] = True
                    result['message'] += " (universe bootstrapped)"
            except Exception as e:
                logger.warning(f"Could not bootstrap universe_membership: {e}")
        
        logger.info(f"Data maintenance complete: {result['message']}")
        return result
    
    def _bootstrap_universe_membership(self):
        """
        Build universe_membership table from the constituents CSV.
        Called automatically when database is bootstrapped from empty.
        """
        try:
            from data.universe import build_membership_from_csv
            
            universe_config = self._get_config_section('universe', {})
            csv_path = self._get_from_config(universe_config, 'constituents_csv_path', 'data/sp500_constituents.csv')
            index_name = self._get_from_config(universe_config, 'index_name', 'SP500')
            
            if not Path(csv_path).exists():
                logger.warning(f"Constituents CSV not found: {csv_path}")
                return
            
            result = build_membership_from_csv(
                storage=self.storage,
                csv_path=csv_path,
                index_name=index_name,
                overwrite=True
            )
            
            logger.info(f"Universe membership bootstrapped: {result.get('rows_inserted', 0)} rows")
            
        except Exception as e:
            logger.warning(f"Failed to bootstrap universe_membership: {e}")
    
    def get_coverage_report(self, universe: Optional[Set[int]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive coverage report.
        
        Args:
            universe: Optional set of asset_ids to focus on
        
        Returns:
            Dict with coverage summary and per-asset details
        """
        summary = self.checker.get_current_coverage()
        per_asset = self.checker.get_per_asset_coverage(universe)
        
        report = {
            'summary': summary,
            'per_asset': per_asset.to_dict('records') if len(per_asset) > 0 else [],
            'config': {
                'min_history_start_date': str(self.min_history_start_date),
                'max_history_lag_days': self.max_history_lag_days,
                'default_vendor': self.default_vendor
            }
        }
        
        # Calculate coverage metrics
        if summary['has_data'] and len(per_asset) > 0:
            report['metrics'] = {
                'assets_with_data': len(per_asset),
                'avg_bars_per_asset': per_asset['bar_count'].mean(),
                'min_bars_per_asset': per_asset['bar_count'].min(),
                'max_bars_per_asset': per_asset['bar_count'].max(),
            }
        
        return report


def ensure_data_coverage(
    storage: StorageBackend,
    config: Optional[Dict[str, Any]] = None,
    mode: str = "full-history",
    target_start: Optional[date] = None,
    target_end: Optional[date] = None,
    symbols: Optional[List[str]] = None,
    vendor: Optional[str] = None,
    auto_fetch: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to ensure data coverage.
    
    This is the main entry point for other modules to call.
    
    Args:
        storage: StorageBackend instance
        config: Config dict (if None, will use get_config())
        mode: "full-history", "daily-top-up", or "custom"
        target_start: Custom start date (for mode="custom")
        target_end: Custom end date (optional)
        symbols: List of symbols (if None, uses config/DB)
        vendor: Data vendor (if None, uses config default)
        auto_fetch: If True, automatically fetch missing data
    
    Returns:
        Dict with status and details
    """
    if config is None:
        from configs.loader import get_config
        config = get_config().__dict__
    
    manager = DataMaintenanceManager(storage, config)
    return manager.ensure_coverage(
        mode=mode,
        target_start=target_start,
        target_end=target_end,
        symbols=symbols,
        vendor=vendor,
        auto_fetch=auto_fetch
    )

