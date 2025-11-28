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
        api: Optional[AsOfQueryAPI] = None,
        as_of_date: Optional[date] = None
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, ...}
        self.orders: Dict[str, Dict] = {}  # order_id -> order dict
        self.order_counter = 0
        self.api = api
        self.as_of_date = as_of_date or date.today()
        
        # Cache for symbol -> asset_id mapping and prices
        self._symbol_to_asset_id: Dict[str, int] = {}
        self._price_cache: Dict[str, float] = {}
        
        # Load mappings if API is provided
        if self.api:
            self._load_mappings()
    
    def _load_mappings(self):
        """Load symbol to asset_id mapping from database."""
        try:
            assets_df = self.api.storage.query("SELECT asset_id, symbol FROM assets")
            self._symbol_to_asset_id = dict(zip(assets_df['symbol'], assets_df['asset_id']))
        except Exception:
            pass
    
    def _load_prices(self, as_of_date: Optional[date] = None):
        """Load latest prices for all assets."""
        query_date = as_of_date or self.as_of_date
        try:
            bars_df = self.api.get_bars_asof(query_date, lookback_days=5)
            if len(bars_df) > 0:
                # Get latest price per asset
                latest_bars = bars_df.groupby('asset_id').last().reset_index()
                
                # Map asset_id to symbol
                asset_to_symbol = {v: k for k, v in self._symbol_to_asset_id.items()}
                for _, row in latest_bars.iterrows():
                    symbol = asset_to_symbol.get(row['asset_id'])
                    if symbol:
                        self._price_cache[symbol] = row['adj_close']
        except Exception:
            pass
    
    def set_as_of_date(self, as_of_date: date):
        """Set the simulation date and refresh prices."""
        self.as_of_date = as_of_date
        self._price_cache = {}
        if self.api:
            self._load_prices(as_of_date)
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        positions_list = []
        for symbol, pos in self.positions.items():
            if pos['quantity'] != 0:
                quote = self.get_quote(symbol)
                market_value = pos['quantity'] * quote['last']
                positions_list.append({
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'avg_price': pos['avg_price'],
                    'market_value': market_value
                })
        return positions_list
    
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote from cached prices or API."""
        # Check cache first
        if symbol in self._price_cache:
            price = self._price_cache[symbol]
            return {
                'bid': price * 0.999,  # Simulate small spread
                'ask': price * 1.001,
                'last': price
            }
        
        # Try to get from API
        if self.api and symbol in self._symbol_to_asset_id:
            asset_id = self._symbol_to_asset_id[symbol]
            try:
                bars_df = self.api.get_bars_asof(
                    self.as_of_date,
                    lookback_days=5,
                    universe={asset_id}
                )
                if len(bars_df) > 0:
                    price = bars_df['adj_close'].iloc[-1]
                    self._price_cache[symbol] = price
                    return {
                        'bid': price * 0.999,
                        'ask': price * 1.001,
                        'last': price
                    }
            except Exception:
                pass
        
        # Fallback
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

