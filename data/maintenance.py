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
            
            total_bars = int(result['total_bars'].iloc[0]) if result['total_bars'].iloc[0] is not None else 0
            num_assets = int(result['num_assets'].iloc[0]) if result['num_assets'].iloc[0] is not None else 0
            
            # Check if we actually have data (not just an empty table)
            if total_bars == 0 or result['max_date'].iloc[0] is None:
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
                'total_bars': total_bars,
                'num_assets': num_assets,
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
        
        # Check for sparse data - if we have very few bars relative to the date range,
        # we should treat this as needing a full refetch rather than just filling edges.
        # This catches cases where a partial/failed fetch left scattered data points.
        if coverage['num_assets'] > 0:
            avg_bars_per_asset = coverage['total_bars'] / coverage['num_assets']
            expected_trading_days = (max_date - min_date).days * 252 / 365  # Rough estimate
            # If we're missing more than 10% of expected data, treat as sparse
            # This is aggressive because partial data is worse than no data for training
            if expected_trading_days > 100 and avg_bars_per_asset < expected_trading_days * 0.9:
                # Data is sparse (less than 90% of expected days)
                # Treat as needing full refetch from target_start to target_end
                coverage_pct = (avg_bars_per_asset / expected_trading_days) * 100
                logger.warning(
                    f"Sparse data detected: {avg_bars_per_asset:.0f} bars/asset "
                    f"({coverage_pct:.1f}% of ~{expected_trading_days:.0f} expected trading days). "
                    f"Triggering full refetch."
                )
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
        from data.universe import TradingCalendar
        
        self.storage = storage
        self.config = config or {}
        self.checker = DataCoverageChecker(storage)
        self.calendar = TradingCalendar()  # For trading day checks
        
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
    
    def get_symbols_from_config(self, include_universe: bool = True, include_benchmarks: bool = True) -> List[str]:
        """
        Get list of symbols from config, universe membership, CSV, or database.
        
        Priority:
        1. Config-specified symbol list (if any)
        2. Universe membership symbols (if include_universe=True and universe exists in DB)
        3. Universe constituents CSV file (bootstrap case - empty DB)
        4. All symbols with price data in bars_daily
        
        Additionally, benchmark symbols (SPY, DIA, QQQ) are always included when
        include_benchmarks=True to ensure they're available for comparison.
        
        Args:
            include_universe: If True, include symbols from universe_membership table or CSV
            include_benchmarks: If True, include benchmark symbols from config
        """
        symbols = []
        
        # Check for config-specified symbols
        data_config = self._get_config_section('data', {})
        config_symbols = self._get_from_config(data_config, 'symbols', [])
        if config_symbols:
            symbols = list(config_symbols)
        else:
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
                        symbols = universe_df['symbol'].tolist()
                except Exception:
                    pass
                
                # Bootstrap case: If universe_membership is empty, read directly from CSV
                if not symbols:
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
                                    symbols = symbols_from_csv
                    except Exception as e:
                        logger.debug(f"Could not read constituents CSV: {e}")
            
            # Fall back to symbols that have price data
            if not symbols:
                try:
                    symbols_df = self.storage.query("""
                        SELECT DISTINCT a.symbol 
                        FROM assets a
                        JOIN bars_daily bd ON a.asset_id = bd.asset_id
                        ORDER BY a.symbol
                    """)
                    if len(symbols_df) > 0:
                        symbols = symbols_df['symbol'].tolist()
                except Exception:
                    pass
        
        # Always include benchmark symbols for comparison
        if include_benchmarks:
            benchmark_symbols = self._get_benchmark_symbols()
            for sym in benchmark_symbols:
                if sym not in symbols:
                    symbols.append(sym)
        
        return symbols
    
    def _get_benchmark_symbols(self) -> List[str]:
        """Get benchmark ticker symbols from config."""
        benchmark_config = self._get_config_section('benchmarks', {})
        definitions = self._get_from_config(benchmark_config, 'definitions', {})
        default_benchmarks = self._get_from_config(benchmark_config, 'default', ['sp500'])
        
        benchmark_symbols = []
        for bench_name in default_benchmarks:
            if bench_name in definitions:
                ticker = definitions[bench_name].get('ticker')
                if ticker:
                    benchmark_symbols.append(ticker)
        
        # Fallback to SPY if no benchmarks configured
        if not benchmark_symbols:
            benchmark_symbols = ['SPY']
        
        return benchmark_symbols
    
    def ensure_coverage(
        self,
        mode: str = "full-history",
        target_start: Optional[date] = None,
        target_end: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        vendor: Optional[str] = None,
        auto_fetch: bool = True,
        bootstrap_universe: bool = True,
        force_full_refetch: bool = False
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
            force_full_refetch: If True, treat all symbols as missing and fetch full history
                               (useful when existing data is insufficient)
        
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
            
            # On weekends/holidays, don't try to fetch the very latest data
            # as vendors often have delays. Use the last completed trading day.
            if not self.calendar.is_trading_day(effective_end):
                # Find the last trading day
                effective_end = self.calendar.previous_trading_day(effective_end)
                logger.info(f"Non-trading day detected, targeting last trading day: {effective_end}")
            
            target_start = effective_end - timedelta(days=5)  # A few days buffer for holidays
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
        
        # If force_full_refetch is True, treat ALL symbols as missing
        # This is useful when the database has some data but it's insufficient
        if force_full_refetch:
            logger.info(f"Force full refetch enabled - treating all {len(symbols)} symbols as missing")
            gaps = {
                'needs_backfill': True,
                'backfill_ranges': [],  # No gap ranges, we'll fetch full history for all
                'missing_symbols': symbols,  # Treat ALL symbols as missing
                'partial_symbols': {},
                'coverage': result['coverage_before']
            }
        else:
            # Identify gaps normally
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
        elif mode == 'daily-top-up' and len(gaps.get('missing_symbols', [])) == 0:
            # For daily top-up, if we have no missing symbols (just date gaps),
            # and we couldn't fetch data, it's likely because:
            # 1. Today is a holiday or weekend
            # 2. Market hasn't closed yet
            # 3. Data isn't available yet from the vendor
            # In these cases, check if we have reasonably recent data
            coverage_after = self.checker.get_current_coverage()
            if coverage_after['has_data']:
                from data.universe import TradingCalendar
                calendar = TradingCalendar()
                today = date.today()
                
                # Find the last expected trading day (could be today if market closed, or previous day)
                try:
                    # If today is not a trading day, get previous
                    if not calendar.is_trading_day(today):
                        last_expected = calendar.previous_trading_day(today)
                    else:
                        # If today is a trading day, data might not be available yet
                        # Accept data from yesterday or today
                        last_expected = calendar.previous_trading_day(today)
                    
                    days_stale = (last_expected - coverage_after['max_date']).days if coverage_after['max_date'] else 999
                    
                    if days_stale <= 1:
                        # Data is fresh enough (within 1 trading day)
                        result['status'] = 'ok'
                        result['message'] = f"Data is current (last: {coverage_after['max_date']}, expected: {last_expected})"
                        logger.info(result['message'])
                    else:
                        result['status'] = 'warning'
                        result['message'] = f"Data is {days_stale} days stale but continuing (last: {coverage_after['max_date']})"
                        logger.warning(result['message'])
                except Exception as e:
                    logger.debug(f"Could not check data freshness: {e}")
                    result['status'] = 'warning'
                    result['message'] = f"Could not fetch new data but existing data available"
            else:
                result['status'] = 'error'
                result['message'] = f"Failed to fetch data: {total_errors} errors"
        else:
            result['status'] = 'error'
            result['message'] = f"Failed to fetch data: {total_errors} errors"
        
        # Bootstrap universe_membership and regimes if this was a fresh database
        if bootstrap_universe and total_bars > 0:
            # Bootstrap universe membership
            try:
                um_count = self.storage.query("SELECT COUNT(*) as cnt FROM universe_membership")
                if um_count['cnt'].iloc[0] == 0:
                    logger.info("Bootstrapping universe_membership from CSV...")
                    self._bootstrap_universe_membership()
                    result['bootstrapped'] = True
                    result['message'] += " (universe bootstrapped)"
            except Exception as e:
                logger.warning(f"Could not bootstrap universe_membership: {e}")
            
            # Bootstrap regimes (check separately - may need to create even if universe exists)
            try:
                regimes_count = self.storage.query("SELECT COUNT(*) as cnt FROM regimes")
                regimes_exist = regimes_count['cnt'].iloc[0] > 0
            except Exception:
                regimes_exist = False
            
            if not regimes_exist:
                logger.info("Bootstrapping regimes...")
                self._bootstrap_regimes()
                result['message'] += " (regimes bootstrapped)"
        
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
    
    def _bootstrap_regimes(self):
        """
        Build regimes table by fitting a regime model on historical data.
        Called automatically when database is bootstrapped from empty.
        """
        try:
            from labels.regimes import RegimeLabelGenerator
            
            coverage = self.checker.get_current_coverage()
            if not coverage['has_data']:
                logger.warning("Cannot bootstrap regimes: no price data available")
                return
            
            # Use the full date range from the data
            start_date = coverage['min_date']
            end_date = coverage['max_date']
            
            logger.info(f"Fitting regime model from {start_date} to {end_date}...")
            
            regime_gen = RegimeLabelGenerator(self.storage, n_regimes=4)
            regimes_df = regime_gen.fit_regimes(start_date, end_date, method="kmeans")
            
            if len(regimes_df) > 0:
                regime_gen.save_regimes(regimes_df)
                regime_gen.save_model("artifacts/models/regime_model.pkl")
                logger.info(f"Regimes bootstrapped: {len(regimes_df)} dates labeled")
            else:
                logger.warning("No regime labels generated")
                
        except Exception as e:
            logger.warning(f"Failed to bootstrap regimes: {e}")
    
    def validate_data_sufficiency(self, min_days: int = 252) -> Dict[str, Any]:
        """
        Validate that we have sufficient data for trading.
        
        Args:
            min_days: Minimum number of trading days required (default: 252, ~1 year)
        
        Returns:
            Dict with validation results
        """
        coverage = self.checker.get_current_coverage()
        
        result = {
            'is_sufficient': False,
            'has_data': coverage['has_data'],
            'total_bars': coverage['total_bars'],
            'num_assets': coverage['num_assets'],
            'min_date': coverage['min_date'],
            'max_date': coverage['max_date'],
            'issues': []
        }
        
        if not coverage['has_data']:
            result['issues'].append("No data in database - bootstrap required")
            return result
        
        # Check number of assets
        if coverage['num_assets'] < 10:
            result['issues'].append(f"Only {coverage['num_assets']} assets - need at least 10")
        
        # Estimate trading days based on total bars and assets
        if coverage['num_assets'] > 0:
            avg_bars_per_asset = coverage['total_bars'] / coverage['num_assets']
            if avg_bars_per_asset < min_days:
                result['issues'].append(
                    f"Average {avg_bars_per_asset:.0f} bars per asset - need at least {min_days}"
                )
        
        # Check date range
        if coverage['min_date'] and coverage['max_date']:
            date_range_days = (coverage['max_date'] - coverage['min_date']).days
            if date_range_days < min_days:
                result['issues'].append(
                    f"Date range is only {date_range_days} days - need at least {min_days}"
                )
        
        # Check data freshness using proper trading calendar
        if coverage['max_date']:
            from data.universe import TradingCalendar
            calendar = TradingCalendar()
            today = date.today()
            
            # Get last trading day
            try:
                if not calendar.is_trading_day(today):
                    last_trading_day = calendar.previous_trading_day(today)
                else:
                    last_trading_day = today
                
                days_stale = (last_trading_day - coverage['max_date']).days
                
                if days_stale > 5:
                    result['issues'].append(
                        f"Data is {days_stale} calendar days stale (last: {coverage['max_date']}, expected: {last_trading_day})"
                    )
            except Exception as e:
                logger.debug(f"Could not check data freshness: {e}")
        
        result['is_sufficient'] = len(result['issues']) == 0
        return result
    
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
    auto_fetch: bool = True,
    force_full_refetch: bool = False
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
        force_full_refetch: If True, treat all symbols as missing and fetch full history
    
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
        auto_fetch=auto_fetch,
        force_full_refetch=force_full_refetch
    )


def prepare_for_trading(
    storage: StorageBackend,
    config: Optional[Dict[str, Any]] = None,
    requested_date: Optional[date] = None,
    lookback_days: int = 252,
    train_days: int = 750,
    auto_fetch: bool = True,
    force_mode: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive preparation for live/paper trading.
    
    This function handles all data validation and preparation needed before
    running the trading loop. It:
    1. Determines the correct trading date (handles holidays/weekends)
    2. Validates sufficient historical data exists for features and training
    3. Attempts to fetch missing data if auto_fetch is enabled
    4. Returns the effective trading date and data readiness status
    
    AUTOMATIC BOOTSTRAP: If the database is empty or severely lacking data
    (< 100 bars per asset when we expect ~1500), the system will automatically
    fetch full historical data. No manual bootstrap command needed.
    
    This is the main entry point for the live loop to call.
    
    Args:
        storage: StorageBackend instance
        config: Config dict (if None, will use get_config())
        requested_date: Date to trade on (if None, uses today or last trading day)
        lookback_days: Days of history needed for feature generation (default: 252)
        train_days: Days of history needed for model training (default: 750)
        auto_fetch: If True, automatically fetch missing data
        force_mode: If True, skip validation and just use whatever data is available
    
    Returns:
        Dict with:
            - ready: bool - whether trading can proceed
            - trading_date: date - the effective date to use for trading
            - data_date: date - the last date with available data
            - issues: list - any issues that prevent trading
            - warnings: list - non-fatal warnings
            - coverage: dict - data coverage summary
            - fetch_result: dict - result of any data fetch attempt
    """
    from data.universe import TradingCalendar
    
    if config is None:
        from configs.loader import get_config
        config = get_config().__dict__
    
    calendar = TradingCalendar()
    today = date.today()
    
    result = {
        'ready': False,
        'trading_date': None,
        'data_date': None,
        'issues': [],
        'warnings': [],
        'coverage': None,
        'fetch_result': None,
        'bootstrapped': False
    }
    
    # Step 1: Determine the target trading date
    if requested_date:
        target_date = requested_date
    else:
        # Use today if it's a trading day, otherwise previous trading day
        if calendar.is_trading_day(today):
            target_date = today
        else:
            target_date = calendar.previous_trading_day(today)
    
    result['trading_date'] = target_date
    logger.info(f"Target trading date: {target_date}")
    
    # Step 2: Check current data coverage
    manager = DataMaintenanceManager(storage, config)
    coverage = manager.checker.get_current_coverage()
    result['coverage'] = coverage
    
    # Step 3: Automatic bootstrap if needed
    # Check if we need to bootstrap (empty DB or severely lacking data)
    needs_bootstrap = False
    bootstrap_reason = None
    
    if not coverage['has_data']:
        needs_bootstrap = True
        bootstrap_reason = "Database is empty"
    elif coverage['num_assets'] > 0:
        avg_bars = coverage['total_bars'] / coverage['num_assets']
        # If we have less than 100 bars per asset, we need a full bootstrap
        # (We expect ~1500 bars for 6 years of daily data)
        if avg_bars < 100:
            needs_bootstrap = True
            bootstrap_reason = f"Insufficient data: only {avg_bars:.0f} bars per asset (need ~1500 for 6 years)"
    
    if needs_bootstrap and auto_fetch:
        logger.warning(f"AUTO-BOOTSTRAP TRIGGERED: {bootstrap_reason}")
        logger.info("Fetching full historical data (this may take a few minutes)...")
        
        fetch_result = manager.ensure_coverage(
            mode="full-history",
            target_end=target_date,
            auto_fetch=True,
            bootstrap_universe=True
        )
        result['fetch_result'] = fetch_result
        result['bootstrapped'] = True
        
        # Re-check coverage
        coverage = manager.checker.get_current_coverage()
        result['coverage'] = coverage
        
        if coverage['has_data']:
            logger.info(f"Bootstrap successful: {coverage['total_bars']:,} bars for {coverage['num_assets']} assets")
        else:
            result['issues'].append(f"Bootstrap failed: {fetch_result.get('message')}")
            return result
    elif needs_bootstrap and not auto_fetch:
        result['issues'].append(f"{bootstrap_reason}. Run with auto_fetch=True to bootstrap.")
        return result
    
    # For force_mode, skip all validation and just use available data
    if force_mode:
        if coverage['has_data']:
            # Just use whatever data we have
            data_date = coverage['max_date']
            result['data_date'] = data_date
            result['trading_date'] = data_date  # Use the last date with data
            result['ready'] = True
            
            # Add informational warnings
            if coverage['num_assets'] > 0:
                avg_bars = coverage['total_bars'] / coverage['num_assets']
                if avg_bars < lookback_days:
                    result['warnings'].append(
                        f"Limited data: {avg_bars:.0f} bars per asset (recommend {lookback_days}+)"
                    )
            
            # Still ensure regimes exist even in force mode
            try:
                regimes_count = manager.storage.query("SELECT COUNT(*) as cnt FROM regimes")
                regimes_exist = regimes_count['cnt'].iloc[0] > 0
            except Exception:
                regimes_exist = False
            
            if not regimes_exist:
                logger.info("FORCE MODE: Regimes table missing - bootstrapping...")
                try:
                    manager._bootstrap_regimes()
                    result['regimes_bootstrapped'] = True
                except Exception as e:
                    logger.warning(f"Could not bootstrap regimes: {e}")
            
            logger.info(f"FORCE MODE: Using available data as-of {data_date}")
            return result
        else:
            result['issues'].append("No data available even for force mode")
            return result
    
    # Normal validation flow continues below...
    if not coverage['has_data']:
        # No data at all - need full bootstrap
        result['issues'].append("No data in database - full bootstrap required")
        
        if auto_fetch:
            logger.info("Attempting full data bootstrap...")
            fetch_result = manager.ensure_coverage(
                mode="full-history",
                target_end=target_date,
                auto_fetch=True
            )
            result['fetch_result'] = fetch_result
            
            # Re-check coverage
            coverage = manager.checker.get_current_coverage()
            result['coverage'] = coverage
            
            if coverage['has_data']:
                result['issues'] = []  # Clear the issue
                logger.info("Bootstrap successful")
            else:
                result['issues'].append(f"Bootstrap failed: {fetch_result.get('message')}")
                return result
        else:
            return result
    
    # Step 3: Determine the effective data date (last date with data)
    data_date = coverage['max_date']
    result['data_date'] = data_date
    logger.info(f"Last data date: {data_date}")
    
    # Step 4: Check if data is fresh enough
    # Get the last trading day before or on target_date
    if target_date > today:
        result['issues'].append(f"Cannot trade future date {target_date}")
        return result
    
    # Find the expected data date (the most recent trading day for which data SHOULD be available)
    # Key insight: We can only expect data for COMPLETED trading days
    # - If today is a trading day, we might not have today's data yet (market still open or vendor delay)
    # - We should expect data from the PREVIOUS completed trading day
    
    if target_date == today:
        # For today, we expect data from the previous trading day (yesterday's close)
        # This handles: market still open, vendor delays, early close days, etc.
        yesterday = today - timedelta(days=1)
        expected_data_date = calendar.previous_trading_day(yesterday)
        if not calendar.is_trading_day(yesterday):
            # yesterday wasn't a trading day, so previous_trading_day gives us the right date
            pass
        else:
            # yesterday was a trading day, so that's what we expect
            expected_data_date = yesterday
        
        logger.info(f"Trading on {target_date}, expecting data from previous trading day: {expected_data_date}")
    elif calendar.is_trading_day(target_date):
        # For a past trading day, we expect data from that day
        expected_data_date = target_date
    else:
        # For a non-trading day, expect data from the previous trading day
        expected_data_date = calendar.previous_trading_day(target_date)
    
    # Calculate staleness in TRADING days (not calendar days)
    def count_trading_days_between(start_date: date, end_date: date) -> int:
        """Count trading days between two dates (exclusive of start, inclusive of end)."""
        if start_date >= end_date:
            return 0
        try:
            trading_days = calendar.get_trading_days(start_date, end_date)
            # Exclude start_date if it's in the list
            return len([d for d in trading_days if d.date() > start_date])
        except Exception:
            # Fallback to calendar days
            return (end_date - start_date).days
    
    if data_date < expected_data_date:
        trading_days_stale = count_trading_days_between(data_date, expected_data_date)
        calendar_days_stale = (expected_data_date - data_date).days
        
        if auto_fetch and trading_days_stale > 0:
            logger.info(f"Data is {trading_days_stale} trading days stale ({calendar_days_stale} calendar days), attempting top-up...")
            fetch_result = manager.ensure_coverage(
                mode="daily-top-up",
                target_end=target_date,
                auto_fetch=True
            )
            result['fetch_result'] = fetch_result
            
            # Re-check coverage
            coverage = manager.checker.get_current_coverage()
            result['coverage'] = coverage
            data_date = coverage['max_date']
            result['data_date'] = data_date
        
        # Re-calculate staleness in trading days
        trading_days_stale = count_trading_days_between(data_date, expected_data_date) if data_date else 999
        
        if trading_days_stale > 3:
            # More than 3 trading days stale is a problem
            result['issues'].append(
                f"Data is {trading_days_stale} trading days stale (last: {data_date}, expected: {expected_data_date})"
            )
        elif trading_days_stale > 0:
            # 1-3 trading days stale is a warning (could be vendor delay, market just closed, etc.)
            result['warnings'].append(
                f"Data is {trading_days_stale} trading days behind expected (last: {data_date}, expected: {expected_data_date}). "
                f"This may be normal if market just closed or vendor has delay."
            )
            # Adjust trading date to use available data
            result['trading_date'] = data_date
            logger.info(f"Adjusted trading date to last available data: {data_date}")
        else:
            # trading_days_stale == 0 means data is current for the last completed trading day
            # This can happen if expected_data_date is today but market hasn't closed yet
            # or if there were holidays between data_date and expected_data_date
            logger.info(f"Data is current (last: {data_date}, no trading days missed)")
            result['trading_date'] = data_date
    
    # Step 5: Validate sufficient history for features and training
    min_history_start = manager.min_history_start_date
    
    # Calculate required dates
    required_feature_start = data_date - timedelta(days=int(lookback_days * 1.5))  # Buffer for weekends/holidays
    required_train_start = data_date - timedelta(days=int(train_days * 1.5))
    required_start = min(required_feature_start, required_train_start, min_history_start)
    
    # Allow a small tolerance (7 days) for the start date since:
    # 1. min_history_start_date might be a holiday (e.g., 2020-01-01 is New Year's Day)
    # 2. The first few days of data might have issues
    # The important thing is having enough trading days of history
    start_tolerance_days = 7
    if coverage['min_date'] > required_start + timedelta(days=start_tolerance_days):
        result['issues'].append(
            f"Insufficient history: data starts {coverage['min_date']}, "
            f"but need data from around {required_start} for {train_days}-day training + {lookback_days}-day lookback"
        )
    
    # Step 6: Validate sufficient assets
    if coverage['num_assets'] < 10:
        result['issues'].append(f"Only {coverage['num_assets']} assets in database - need at least 10")
    
    # Step 7: Estimate if we have enough bars per asset
    if coverage['num_assets'] > 0:
        avg_bars = coverage['total_bars'] / coverage['num_assets']
        min_required_bars = max(lookback_days, train_days // 5)  # Need at least lookback days per asset
        
        if avg_bars < min_required_bars:
            if avg_bars < 50:
                # Very few bars - likely needs full bootstrap
                result['issues'].append(
                    f"INSUFFICIENT DATA: Only {avg_bars:.0f} bars per asset (need {min_required_bars}+). "
                    f"Run: python scripts/ensure_data_coverage.py --mode full-history --auto-fetch"
                )
            else:
                result['warnings'].append(
                    f"Average {avg_bars:.0f} bars per asset may be insufficient "
                    f"(recommend at least {min_required_bars})"
                )
    
    # Step 8: Ensure regimes table exists (needed for regime features)
    # This is checked separately because regimes might be missing even if data is complete
    try:
        regimes_count = manager.storage.query("SELECT COUNT(*) as cnt FROM regimes")
        regimes_exist = regimes_count['cnt'].iloc[0] > 0
    except Exception:
        regimes_exist = False
    
    if not regimes_exist and coverage['has_data']:
        logger.info("Regimes table missing - bootstrapping regime model...")
        try:
            manager._bootstrap_regimes()
            result['regimes_bootstrapped'] = True
        except Exception as e:
            # Regimes are nice-to-have, not critical - just warn
            logger.warning(f"Could not bootstrap regimes (non-fatal): {e}")
    
    # Determine final readiness
    result['ready'] = len(result['issues']) == 0
    
    if result['ready']:
        logger.info(f"Data ready for trading on {result['trading_date']}")
        if result['warnings']:
            for warning in result['warnings']:
                logger.warning(warning)
    else:
        logger.error(f"Data not ready for trading: {result['issues']}")
    
    return result

