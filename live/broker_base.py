"""Base broker interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import date, datetime


class BrokerAPI(ABC):
    """Abstract base class for broker interfaces."""
    
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

