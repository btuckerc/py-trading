"""Paper trading broker (simulated)."""

import pandas as pd
from typing import List, Dict, Optional
from datetime import date, datetime
from live.broker_base import BrokerAPI
from data.asof_api import AsOfQueryAPI


class PaperBroker(BrokerAPI):
    """Simulated broker for paper trading."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        api: Optional[AsOfQueryAPI] = None
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, ...}
        self.orders: Dict[str, Dict] = {}  # order_id -> order dict
        self.order_counter = 0
        self.api = api
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        return [
            {
                'symbol': symbol,
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'market_value': pos.get('market_value', 0)
            }
            for symbol, pos in self.positions.items()
            if pos['quantity'] != 0
        ]
    
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote (simulated)."""
        # In paper trading, use latest price from data
        if self.api:
            # Get latest bar for symbol
            # This is simplified - would need to map symbol to asset_id
            return {
                'bid': 100.0,  # Placeholder
                'ask': 100.0,
                'last': 100.0
            }
        else:
            return {
                'bid': 100.0,
                'ask': 100.0,
                'last': 100.0
            }
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        time_in_force: str = "day"
    ) -> str:
        """Submit an order (simulated fill)."""
        order_id = f"PAPER_{self.order_counter}"
        self.order_counter += 1
        
        # Get quote for fill price
        quote = self.get_quote(symbol)
        fill_price = quote['last']
        
        # Simulate immediate fill for market orders
        if order_type == "market":
            if side == "buy":
                cost = quantity * fill_price
                if cost <= self.cash:
                    # Execute
                    if symbol not in self.positions:
                        self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                    
                    # Update position
                    current_qty = self.positions[symbol]['quantity']
                    current_avg = self.positions[symbol]['avg_price']
                    
                    if current_qty == 0:
                        new_avg = fill_price
                    else:
                        total_cost = (current_qty * current_avg) + cost
                        new_qty = current_qty + quantity
                        new_avg = total_cost / new_qty
                    
                    self.positions[symbol]['quantity'] = current_qty + quantity
                    self.positions[symbol]['avg_price'] = new_avg
                    self.cash -= cost
                    
                    order_status = "filled"
                else:
                    order_status = "rejected"  # Insufficient cash
            else:  # sell
                if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                    # Execute
                    proceeds = quantity * fill_price
                    self.positions[symbol]['quantity'] -= quantity
                    self.cash += proceeds
                    
                    if self.positions[symbol]['quantity'] == 0:
                        del self.positions[symbol]
                    
                    order_status = "filled"
                else:
                    order_status = "rejected"  # Insufficient shares
        else:
            order_status = "pending"  # Limit orders not fully implemented
        
        # Record order
        self.orders[order_id] = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'status': order_status,
            'fill_price': fill_price if order_status == "filled" else None,
            'timestamp': datetime.now()
        }
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            if self.orders[order_id]['status'] == "pending":
                self.orders[order_id]['status'] = "cancelled"
                return True
        return False
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders."""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o['status'] == status]
        return orders
    
    def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        total_value = self.cash
        
        for symbol, pos in self.positions.items():
            quote = self.get_quote(symbol)
            market_value = pos['quantity'] * quote['last']
            total_value += market_value
        
        return total_value

