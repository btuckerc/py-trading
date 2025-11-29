"""Base broker interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
from datetime import date, datetime


class BrokerAPI(ABC):
    """Abstract base class for broker interfaces."""
    
    @abstractmethod
    def get_account_value(self) -> float:
        """
        Get total account value (equity).
        
        Returns:
            Total account equity (cash + positions value)
        """
        pass
    
    @abstractmethod
    def get_cash(self) -> float:
        """
        Get cash balance.
        
        Returns:
            Available cash balance
        """
        pass
    
    @abstractmethod
    def get_buying_power(self) -> float:
        """
        Get available buying power.
        
        For simple brokers, this can default to get_cash().
        For margin accounts, this may be higher than cash.
        
        Returns:
            Available buying power
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of dicts with keys: symbol, quantity, avg_price, etc.
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.
        
        Returns:
            Dict with keys: bid, ask, last, etc.
        """
        pass
    
    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        time_in_force: str = "day"
    ) -> str:
        """
        Submit an order.
        
        Args:
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market", "limit", etc.
            time_in_force: "day", "gtc", etc.
        
        Returns:
            Order ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get orders.
        
        Args:
            status: Optional filter by status ("open", "filled", "cancelled")
        
        Returns:
            List of order dicts
        """
        pass
    
    def is_tradable(self, symbol: str) -> bool:
        """
        Check if a symbol is tradable at this broker.
        
        Default implementation returns True for all symbols.
        Override in subclasses for broker-specific checks.
        
        Args:
            symbol: Ticker symbol to check
        
        Returns:
            True if the symbol can be traded, False otherwise
        """
        return True
    
    def get_tradable_symbols(self, symbols: List[str]) -> Set[str]:
        """
        Filter a list of symbols to only those that are tradable.
        
        Args:
            symbols: List of symbols to check
        
        Returns:
            Set of tradable symbols
        """
        return {s for s in symbols if self.is_tradable(s)}
    
    def get_untradable_symbols(self, symbols: List[str]) -> Set[str]:
        """
        Get symbols from a list that are NOT tradable.
        
        Args:
            symbols: List of symbols to check
        
        Returns:
            Set of untradable symbols
        """
        return {s for s in symbols if not self.is_tradable(s)}

