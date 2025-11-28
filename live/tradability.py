"""Tradability filter for live trading.

This module provides utilities to filter a research universe down to
symbols that are actually tradable at a broker, with configurable
policies for handling untradable assets.

Usage:
    from live.tradability import TradabilityFilter
    
    filter = TradabilityFilter(broker, config)
    
    # Filter universe
    tradable, untradable = filter.filter_universe(symbols)
    
    # Apply policy to weights
    adjusted_weights = filter.apply_policy(weights_df, symbols)
"""

import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import date
from loguru import logger

from live.broker_base import BrokerAPI


class TradabilityFilter:
    """
    Filters research universe to broker-tradable symbols.
    
    Supports policies for handling untradable assets:
    - "renormalize": Redistribute weights to tradable assets
    - "cash": Park untradable weight in cash
    - "error": Raise error if any assets are untradable
    - "warn": Log warning but proceed (default)
    """
    
    def __init__(
        self,
        broker: BrokerAPI,
        config: Optional[Dict[str, Any]] = None,
        policy: str = "renormalize"
    ):
        """
        Initialize the tradability filter.
        
        Args:
            broker: BrokerAPI instance to check tradability
            config: Optional config dict with tradability settings
            policy: Policy for untradable assets:
                   - "renormalize": Redistribute weights
                   - "cash": Park in cash
                   - "error": Raise error
                   - "warn": Log warning only
        """
        self.broker = broker
        self.config = config or {}
        self.policy = policy
        
        # Cache for tradability checks (to avoid repeated API calls)
        self._tradable_cache: Dict[str, bool] = {}
        self._cache_date: Optional[date] = None
    
    def clear_cache(self):
        """Clear the tradability cache."""
        self._tradable_cache.clear()
        self._cache_date = None
    
    def is_tradable(self, symbol: str, use_cache: bool = True) -> bool:
        """
        Check if a symbol is tradable.
        
        Args:
            symbol: Symbol to check
            use_cache: Whether to use cached results
        
        Returns:
            True if tradable, False otherwise
        """
        today = date.today()
        
        # Clear cache if it's a new day
        if self._cache_date != today:
            self.clear_cache()
            self._cache_date = today
        
        # Check cache
        if use_cache and symbol in self._tradable_cache:
            return self._tradable_cache[symbol]
        
        # Query broker
        result = self.broker.is_tradable(symbol)
        self._tradable_cache[symbol] = result
        
        return result
    
    def filter_universe(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> Tuple[Set[str], Set[str]]:
        """
        Filter symbols to tradable and untradable sets.
        
        Args:
            symbols: List of symbols to filter
            use_cache: Whether to use cached results
        
        Returns:
            Tuple of (tradable_symbols, untradable_symbols)
        """
        tradable = set()
        untradable = set()
        
        for symbol in symbols:
            if self.is_tradable(symbol, use_cache):
                tradable.add(symbol)
            else:
                untradable.add(symbol)
        
        if untradable:
            logger.info(
                f"Tradability filter: {len(tradable)} tradable, "
                f"{len(untradable)} untradable"
            )
            logger.debug(f"Untradable symbols: {sorted(untradable)}")
        
        return tradable, untradable
    
    def apply_policy(
        self,
        weights_df: pd.DataFrame,
        symbol_column: str = "symbol",
        weight_column: str = "weight"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply tradability policy to portfolio weights.
        
        Args:
            weights_df: DataFrame with symbol and weight columns
            symbol_column: Name of symbol column
            weight_column: Name of weight column
        
        Returns:
            Tuple of (adjusted_weights_df, policy_result)
            
            policy_result contains:
                - original_symbols: List of original symbols
                - tradable_symbols: List of tradable symbols
                - untradable_symbols: List of untradable symbols
                - untradable_weight: Total weight of untradable assets
                - policy_applied: The policy that was applied
                - cash_weight: Weight parked in cash (if policy="cash")
        """
        if len(weights_df) == 0:
            return weights_df, {
                'original_symbols': [],
                'tradable_symbols': [],
                'untradable_symbols': [],
                'untradable_weight': 0.0,
                'policy_applied': self.policy,
                'cash_weight': 0.0
            }
        
        # Get symbols from weights
        symbols = weights_df[symbol_column].tolist()
        tradable, untradable = self.filter_universe(symbols)
        
        # Calculate untradable weight
        untradable_mask = weights_df[symbol_column].isin(untradable)
        untradable_weight = weights_df.loc[untradable_mask, weight_column].sum()
        
        result = {
            'original_symbols': symbols,
            'tradable_symbols': list(tradable),
            'untradable_symbols': list(untradable),
            'untradable_weight': float(untradable_weight),
            'policy_applied': self.policy,
            'cash_weight': 0.0
        }
        
        if not untradable:
            # All symbols are tradable, return as-is
            return weights_df, result
        
        # Apply policy
        if self.policy == "error":
            raise ValueError(
                f"Untradable symbols in universe: {sorted(untradable)}. "
                f"Total untradable weight: {untradable_weight:.2%}"
            )
        
        elif self.policy == "warn":
            logger.warning(
                f"Untradable symbols: {sorted(untradable)} "
                f"(weight: {untradable_weight:.2%})"
            )
            # Return original weights with untradable assets removed
            adjusted_df = weights_df[~untradable_mask].copy()
            return adjusted_df, result
        
        elif self.policy == "cash":
            # Remove untradable assets, track weight as cash
            adjusted_df = weights_df[~untradable_mask].copy()
            result['cash_weight'] = float(untradable_weight)
            logger.info(
                f"Parking {untradable_weight:.2%} in cash due to "
                f"{len(untradable)} untradable symbols"
            )
            return adjusted_df, result
        
        elif self.policy == "renormalize":
            # Remove untradable assets and renormalize remaining weights
            adjusted_df = weights_df[~untradable_mask].copy()
            
            if len(adjusted_df) > 0:
                total_tradable_weight = adjusted_df[weight_column].sum()
                if total_tradable_weight > 0:
                    # Renormalize to sum to 1 (or original total)
                    original_total = weights_df[weight_column].sum()
                    scale_factor = original_total / total_tradable_weight
                    adjusted_df[weight_column] = adjusted_df[weight_column] * scale_factor
                    
                    logger.info(
                        f"Renormalized weights after removing {len(untradable)} "
                        f"untradable symbols (redistributed {untradable_weight:.2%})"
                    )
            
            return adjusted_df, result
        
        else:
            logger.warning(f"Unknown policy: {self.policy}, using 'warn'")
            adjusted_df = weights_df[~untradable_mask].copy()
            return adjusted_df, result


def filter_weights_by_tradability(
    weights_df: pd.DataFrame,
    broker: BrokerAPI,
    symbol_column: str = "symbol",
    weight_column: str = "weight",
    policy: str = "renormalize"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to filter weights by tradability.
    
    Args:
        weights_df: DataFrame with symbol and weight columns
        broker: BrokerAPI instance
        symbol_column: Name of symbol column
        weight_column: Name of weight column
        policy: Policy for untradable assets
    
    Returns:
        Tuple of (adjusted_weights_df, policy_result)
    """
    filter = TradabilityFilter(broker, policy=policy)
    return filter.apply_policy(weights_df, symbol_column, weight_column)

