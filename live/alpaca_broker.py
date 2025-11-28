"""Alpaca broker adapter implementing BrokerAPI.

This module provides integration with Alpaca's trading API for both
paper and live trading. Alpaca offers commission-free trading which
makes it ideal for small accounts.

Setup:
    1. Create an Alpaca account at https://alpaca.markets
    2. Generate API keys (paper or live)
    3. Set environment variables:
       - ALPACA_API_KEY
       - ALPACA_SECRET_KEY
       - ALPACA_BASE_URL (optional, defaults to paper trading)

Usage:
    from live.alpaca_broker import AlpacaBroker
    
    # Paper trading (default)
    broker = AlpacaBroker()
    
    # Live trading
    broker = AlpacaBroker(paper=False)
    
    # Get account info
    print(broker.get_account_value())
    print(broker.get_positions())
    
    # Submit order
    order_id = broker.submit_order("AAPL", "buy", 10)
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from .broker_base import BrokerAPI

# Alpaca SDK import - will fail gracefully if not installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not installed. Run: pip install alpaca-py")


class AlpacaBroker(BrokerAPI):
    """
    Alpaca broker implementation.
    
    Supports both paper and live trading via Alpaca's REST API.
    Commission-free trading makes this ideal for small accounts.
    
    Attributes:
        paper: Whether to use paper trading (default True)
        api_key: Alpaca API key (from env or constructor)
        secret_key: Alpaca secret key (from env or constructor)
    """
    
    # Base URLs for Alpaca API
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca broker.
        
        Args:
            api_key: Alpaca API key. If None, reads from ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. If None, reads from ALPACA_SECRET_KEY env var.
            paper: If True (default), use paper trading. If False, use live trading.
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Install with: pip install alpaca-py"
            )
        
        self.paper = paper
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass to constructor."
            )
        
        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )
        
        # Initialize data client for quotes
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # Cache account info
        self._account = None
        self._last_account_fetch = None
        
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaBroker initialized in {mode} mode")
    
    def _refresh_account(self, force: bool = False):
        """Refresh cached account info."""
        now = datetime.now()
        # Cache for 5 seconds
        if (
            force or 
            self._account is None or 
            self._last_account_fetch is None or
            (now - self._last_account_fetch).seconds > 5
        ):
            self._account = self.trading_client.get_account()
            self._last_account_fetch = now
    
    def get_account_value(self) -> float:
        """Get total account value (equity)."""
        self._refresh_account()
        return float(self._account.equity)
    
    def get_buying_power(self) -> float:
        """Get available buying power."""
        self._refresh_account()
        return float(self._account.buying_power)
    
    def get_cash(self) -> float:
        """Get cash balance."""
        self._refresh_account()
        return float(self._account.cash)
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dicts with keys:
                - symbol: Ticker symbol
                - quantity: Number of shares (positive for long, negative for short)
                - avg_price: Average entry price
                - market_value: Current market value
                - unrealized_pnl: Unrealized profit/loss
                - unrealized_pnl_pct: Unrealized P&L as percentage
        """
        positions = self.trading_client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                'symbol': pos.symbol,
                'quantity': int(pos.qty),
                'avg_price': float(pos.avg_entry_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
                'current_price': float(pos.current_price),
                'side': 'long' if int(pos.qty) > 0 else 'short',
            })
        
        return result
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.
        
        Returns:
            Dict with keys: bid, ask, last, bid_size, ask_size
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'symbol': symbol,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'last': float(quote.ask_price),  # Use ask as last for market orders
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                    'timestamp': quote.timestamp.isoformat() if quote.timestamp else None,
                }
            else:
                logger.warning(f"No quote data for {symbol}")
                return {
                    'symbol': symbol,
                    'bid': 0.0,
                    'ask': 0.0,
                    'last': 0.0,
                    'bid_size': 0,
                    'ask_size': 0,
                    'timestamp': None,
                }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {
                'symbol': symbol,
                'bid': 0.0,
                'ask': 0.0,
                'last': 0.0,
                'error': str(e),
            }
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None
    ) -> str:
        """
        Submit an order.
        
        Args:
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market" or "limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Required for limit orders
        
        Returns:
            Order ID string
        
        Raises:
            ValueError: If invalid parameters
            Exception: If order submission fails
        """
        # Validate inputs
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        
        if side.lower() not in ("buy", "sell"):
            raise ValueError(f"Side must be 'buy' or 'sell', got {side}")
        
        if order_type.lower() not in ("market", "limit"):
            raise ValueError(f"Order type must be 'market' or 'limit', got {order_type}")
        
        if order_type.lower() == "limit" and limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        # Map side
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        # Map time in force
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        
        # Create order request
        if order_type.lower() == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=tif
            )
        else:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price
            )
        
        # Submit order
        try:
            order = self.trading_client.submit_order(order_request)
            logger.info(
                f"Order submitted: {side} {quantity} {symbol} @ {order_type} "
                f"(ID: {order.id})"
            )
            return str(order.id)
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        try:
            cancelled = self.trading_client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get orders.
        
        Args:
            status: Optional filter - "open", "closed", "all"
        
        Returns:
            List of order dicts with keys:
                - id: Order ID
                - symbol: Ticker symbol
                - side: "buy" or "sell"
                - quantity: Order quantity
                - filled_quantity: Filled quantity
                - order_type: Order type
                - status: Order status
                - submitted_at: Submission timestamp
                - filled_at: Fill timestamp (if filled)
                - avg_fill_price: Average fill price (if filled)
        """
        try:
            # Map status filter
            if status == "open":
                orders = self.trading_client.get_orders(status=OrderStatus.OPEN)
            elif status == "closed":
                orders = self.trading_client.get_orders(status=OrderStatus.CLOSED)
            else:
                orders = self.trading_client.get_orders()
            
            result = []
            for order in orders:
                result.append({
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': int(order.qty),
                    'filled_quantity': int(order.filled_qty) if order.filled_qty else 0,
                    'order_type': order.type.value,
                    'status': order.status.value,
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'time_in_force': order.time_in_force.value,
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def close_position(self, symbol: str) -> Optional[str]:
        """
        Close entire position for a symbol.
        
        Args:
            symbol: Ticker symbol
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order = self.trading_client.close_position(symbol)
            logger.info(f"Closed position for {symbol} (Order ID: {order.id})")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None
    
    def close_all_positions(self) -> List[str]:
        """
        Close all positions.
        
        Returns:
            List of order IDs for close orders
        """
        try:
            orders = self.trading_client.close_all_positions()
            order_ids = [str(o.id) for o in orders] if orders else []
            logger.info(f"Closed all positions ({len(order_ids)} orders)")
            return order_ids
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return []
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    def get_market_hours(self) -> Dict:
        """Get today's market hours."""
        try:
            clock = self.trading_client.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if clock.next_open else None,
                'next_close': clock.next_close.isoformat() if clock.next_close else None,
            }
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {'is_open': False, 'error': str(e)}

