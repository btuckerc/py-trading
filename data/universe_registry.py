"""Universe registry for configurable, named universe definitions.

This module provides a centralized way to get universe asset_ids based on
named universe definitions in the config. It combines:
1. Index membership from universe_membership table
2. Liquidity filters (price, volume)
3. Optional blacklists/whitelists

Usage:
    from data.universe_registry import UniverseRegistry
    
    registry = UniverseRegistry(storage, config)
    
    # Get universe for backtest
    universe = registry.get_universe("sp500_research", as_of_date)
    
    # Get default universe for a mode
    universe = registry.get_default_universe("backtest", as_of_date)
"""

import pandas as pd
from datetime import date
from typing import Dict, Any, Optional, Set, List
from loguru import logger


class UniverseRegistry:
    """
    Registry for named universe definitions.
    
    Provides a single point of access for getting universe asset_ids
    based on config-defined universe specifications.
    """
    
    def __init__(self, storage, config=None):
        """
        Initialize the universe registry.
        
        Args:
            storage: StorageBackend instance
            config: Config object or dict (if None, will load from get_config())
        """
        self.storage = storage
        
        if config is None:
            from configs.loader import get_config
            config = get_config()
        
        self.config = config
        self._load_universe_definitions()
    
    def _load_universe_definitions(self):
        """Load universe definitions from config."""
        # Handle both Pydantic model and dict configs
        if hasattr(self.config, 'universes'):
            self.universes = self.config.universes or {}
        elif isinstance(self.config, dict):
            self.universes = self.config.get('universes', {})
        else:
            self.universes = {}
        
        # Load defaults
        if hasattr(self.config, 'universe_defaults'):
            self.defaults = self.config.universe_defaults or {}
        elif isinstance(self.config, dict):
            self.defaults = self.config.get('universe_defaults', {})
        else:
            self.defaults = {}
        
        # Fall back to legacy 'universe' config if no named universes
        if not self.universes:
            if hasattr(self.config, 'universe'):
                legacy = self.config.universe
                self.universes = {
                    'default': {
                        'index_name': getattr(legacy, 'index_name', 'SP500'),
                        'min_price_usd': getattr(legacy, 'min_price_usd', 3.0),
                        'min_dollar_volume_percentile': getattr(legacy, 'min_dollar_volume_percentile', 10),
                        'use_survivorship_bias_free': getattr(legacy, 'use_survivorship_bias_free', True),
                    }
                }
            elif isinstance(self.config, dict) and 'universe' in self.config:
                legacy = self.config['universe']
                self.universes = {
                    'default': {
                        'index_name': legacy.get('index_name', 'SP500'),
                        'min_price_usd': legacy.get('min_price_usd', 3.0),
                        'min_dollar_volume_percentile': legacy.get('min_dollar_volume_percentile', 10),
                        'use_survivorship_bias_free': legacy.get('use_survivorship_bias_free', True),
                    }
                }
    
    def list_universes(self) -> List[str]:
        """Get list of available universe names."""
        return list(self.universes.keys())
    
    def get_universe_config(self, universe_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a named universe.
        
        Args:
            universe_name: Name of the universe
        
        Returns:
            Dict with universe configuration
        
        Raises:
            ValueError if universe_name not found
        """
        if universe_name not in self.universes:
            available = ', '.join(self.list_universes())
            raise ValueError(f"Unknown universe: {universe_name}. Available: {available}")
        
        universe_config = self.universes[universe_name]
        
        # Handle both dict and Pydantic model
        if isinstance(universe_config, dict):
            return universe_config
        else:
            # Convert Pydantic model to dict
            return dict(universe_config) if hasattr(universe_config, '__dict__') else {}
    
    def get_default_universe_name(self, mode: str = "backtest") -> str:
        """
        Get the default universe name for a mode.
        
        Args:
            mode: "backtest" or "live"
        
        Returns:
            Universe name
        """
        if isinstance(self.defaults, dict):
            return self.defaults.get(mode, 'default')
        elif hasattr(self.defaults, mode):
            return getattr(self.defaults, mode)
        return 'default'
    
    def get_universe(
        self,
        universe_name: str,
        as_of_date: date,
        apply_filters: bool = True
    ) -> Set[int]:
        """
        Get asset_ids for a named universe as of a specific date.
        
        Args:
            universe_name: Name of the universe (e.g., "sp500_research")
            as_of_date: Date to get universe for
            apply_filters: If True, apply price/volume filters
        
        Returns:
            Set of asset_ids
        """
        config = self.get_universe_config(universe_name)
        
        index_name = config.get('index_name')
        use_survivorship_free = config.get('use_survivorship_bias_free', True)
        min_price = config.get('min_price_usd')
        min_volume_pct = config.get('min_dollar_volume_percentile')
        
        # New filters for multi-asset support
        asset_types = config.get('asset_types')  # e.g., ['equity', 'etf']
        countries = config.get('countries')       # e.g., ['US']
        
        # Start with index membership or all assets
        if use_survivorship_free and index_name:
            universe = self._get_index_members(index_name, as_of_date)
        else:
            universe = self._get_all_assets()
        
        if len(universe) == 0:
            logger.warning(f"Universe {universe_name} has no members for {as_of_date}")
            return universe
        
        # Apply asset_type and country filters
        if asset_types:
            universe = self._filter_by_asset_type(universe, asset_types)
        
        if countries:
            universe = self._filter_by_country(universe, countries)
        
        # Apply price/volume filters if requested
        if apply_filters:
            if min_price is not None:
                universe = self._filter_by_price(universe, as_of_date, min_price)
            
            if min_volume_pct is not None:
                universe = self._filter_by_volume_percentile(universe, as_of_date, min_volume_pct)
        
        logger.debug(f"Universe {universe_name} at {as_of_date}: {len(universe)} assets")
        return universe
    
    def get_default_universe(
        self,
        mode: str,
        as_of_date: date,
        apply_filters: bool = True
    ) -> Set[int]:
        """
        Get asset_ids for the default universe of a mode.
        
        Args:
            mode: "backtest" or "live"
            as_of_date: Date to get universe for
            apply_filters: If True, apply price/volume filters
        
        Returns:
            Set of asset_ids
        """
        universe_name = self.get_default_universe_name(mode)
        return self.get_universe(universe_name, as_of_date, apply_filters)
    
    def _get_index_members(self, index_name: str, as_of_date: date) -> Set[int]:
        """Get asset_ids from universe_membership table."""
        try:
            result = self.storage.conn.execute("""
                SELECT DISTINCT asset_id
                FROM universe_membership
                WHERE date = ? AND index_name = ? AND in_index = TRUE
            """, [as_of_date, index_name]).df()
            
            if len(result) > 0:
                return set(result['asset_id'].values)
            
            # If no exact date match, try most recent date before as_of_date
            result = self.storage.conn.execute("""
                SELECT DISTINCT asset_id
                FROM universe_membership
                WHERE date = (
                    SELECT MAX(date) FROM universe_membership 
                    WHERE date <= ? AND index_name = ?
                ) AND index_name = ? AND in_index = TRUE
            """, [as_of_date, index_name, index_name]).df()
            
            if len(result) > 0:
                return set(result['asset_id'].values)
            
            return set()
        except Exception as e:
            logger.warning(f"Error getting index members: {e}")
            return set()
    
    def _get_all_assets(self) -> Set[int]:
        """Get all asset_ids from assets table."""
        try:
            result = self.storage.query("SELECT asset_id FROM assets WHERE is_active = TRUE")
            if len(result) > 0:
                return set(result['asset_id'].values)
            return set()
        except Exception:
            return set()
    
    def _filter_by_price(
        self,
        universe: Set[int],
        as_of_date: date,
        min_price: float
    ) -> Set[int]:
        """Filter universe by minimum price."""
        if not universe:
            return universe
        
        try:
            asset_list = ",".join(map(str, universe))
            result = self.storage.conn.execute(f"""
                SELECT DISTINCT asset_id
                FROM bars_daily
                WHERE date = ? AND asset_id IN ({asset_list}) AND close >= ?
            """, [as_of_date, min_price]).df()
            
            if len(result) > 0:
                return universe.intersection(set(result['asset_id'].values))
            
            # If no data for exact date, try most recent
            result = self.storage.conn.execute(f"""
                SELECT asset_id, close
                FROM (
                    SELECT asset_id, close, 
                           ROW_NUMBER() OVER (PARTITION BY asset_id ORDER BY date DESC) as rn
                    FROM bars_daily
                    WHERE date <= ? AND asset_id IN ({asset_list})
                ) sub
                WHERE rn = 1 AND close >= ?
            """, [as_of_date, min_price]).df()
            
            if len(result) > 0:
                return universe.intersection(set(result['asset_id'].values))
            
            return universe
        except Exception as e:
            logger.warning(f"Error filtering by price: {e}")
            return universe
    
    def _filter_by_volume_percentile(
        self,
        universe: Set[int],
        as_of_date: date,
        min_percentile: float,
        lookback_days: int = 20
    ) -> Set[int]:
        """Filter universe by dollar volume percentile."""
        if not universe:
            return universe
        
        try:
            asset_list = ",".join(map(str, universe))
            
            # Get average dollar volume over lookback period
            result = self.storage.conn.execute(f"""
                SELECT asset_id, AVG(close * volume) as avg_dollar_volume
                FROM bars_daily
                WHERE date <= ? 
                  AND date > DATE(?, '-{lookback_days} days')
                  AND asset_id IN ({asset_list})
                GROUP BY asset_id
            """, [as_of_date, as_of_date]).df()
            
            if len(result) == 0:
                return universe
            
            # Calculate percentile threshold
            threshold = result['avg_dollar_volume'].quantile(min_percentile / 100.0)
            
            # Filter by threshold
            filtered = result[result['avg_dollar_volume'] >= threshold]['asset_id'].values
            return universe.intersection(set(filtered))
        except Exception as e:
            logger.warning(f"Error filtering by volume: {e}")
            return universe
    
    def _filter_by_asset_type(
        self,
        universe: Set[int],
        asset_types: List[str]
    ) -> Set[int]:
        """Filter universe by asset type (equity, etf, future, crypto, etc.)."""
        if not universe or not asset_types:
            return universe
        
        try:
            asset_list = ",".join(map(str, universe))
            type_list = "', '".join(asset_types)
            
            result = self.storage.conn.execute(f"""
                SELECT asset_id
                FROM assets
                WHERE asset_id IN ({asset_list})
                  AND (asset_type IN ('{type_list}') OR asset_type IS NULL)
            """).df()
            
            if len(result) > 0:
                return universe.intersection(set(result['asset_id'].values))
            return universe
        except Exception as e:
            logger.warning(f"Error filtering by asset type: {e}")
            return universe
    
    def _filter_by_country(
        self,
        universe: Set[int],
        countries: List[str]
    ) -> Set[int]:
        """Filter universe by country code (e.g., 'US', 'GB', 'JP')."""
        if not universe or not countries:
            return universe
        
        try:
            asset_list = ",".join(map(str, universe))
            country_list = "', '".join(countries)
            
            result = self.storage.conn.execute(f"""
                SELECT asset_id
                FROM assets
                WHERE asset_id IN ({asset_list})
                  AND (country IN ('{country_list}') OR country IS NULL)
            """).df()
            
            if len(result) > 0:
                return universe.intersection(set(result['asset_id'].values))
            return universe
        except Exception as e:
            logger.warning(f"Error filtering by country: {e}")
            return universe


def get_universe(
    storage,
    universe_name: str,
    as_of_date: date,
    config=None,
    apply_filters: bool = True
) -> Set[int]:
    """
    Convenience function to get universe asset_ids.
    
    Args:
        storage: StorageBackend instance
        universe_name: Name of the universe
        as_of_date: Date to get universe for
        config: Optional config (if None, uses get_config())
        apply_filters: If True, apply price/volume filters
    
    Returns:
        Set of asset_ids
    """
    registry = UniverseRegistry(storage, config)
    return registry.get_universe(universe_name, as_of_date, apply_filters)


def get_default_universe(
    storage,
    mode: str,
    as_of_date: date,
    config=None,
    apply_filters: bool = True
) -> Set[int]:
    """
    Convenience function to get default universe for a mode.
    
    Args:
        storage: StorageBackend instance
        mode: "backtest" or "live"
        as_of_date: Date to get universe for
        config: Optional config (if None, uses get_config())
        apply_filters: If True, apply price/volume filters
    
    Returns:
        Set of asset_ids
    """
    registry = UniverseRegistry(storage, config)
    return registry.get_default_universe(mode, as_of_date, apply_filters)

