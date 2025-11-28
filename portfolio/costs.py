"""Transaction cost and slippage models."""

import numpy as np
from typing import Dict, Optional


class TransactionCostModel:
    """Models transaction costs and slippage."""
    
    def __init__(
        self,
        commission_bps: float = 1.0,
        slippage_model: str = "simple",
        slippage_bps: float = 2.0
    ):
        """
        Args:
            commission_bps: Commission in basis points per trade
            slippage_model: "simple" or "volatility_based"
            slippage_bps: Slippage in basis points (for simple model)
        """
        self.commission_bps = commission_bps
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
    
    def compute_cost(
        self,
        trade_value: float,
        bar_data: Optional[Dict] = None
    ) -> float:
        """
        Compute total transaction cost (commission + slippage).
        
        Args:
            trade_value: Dollar value of trade
            bar_data: Optional dict with price/volume data for slippage calculation
        
        Returns:
            Total cost in dollars
        """
        # Commission
        commission = trade_value * (self.commission_bps / 10000.0)
        
        # Slippage
        if self.slippage_model == "simple":
            slippage = trade_value * (self.slippage_bps / 10000.0)
        elif self.slippage_model == "volatility_based":
            # Slippage proportional to volatility (if available)
            if bar_data and 'volatility' in bar_data:
                volatility = bar_data['volatility']
                slippage_multiplier = 1.0 + volatility * 10  # Scale volatility
                slippage = trade_value * (self.slippage_bps / 10000.0) * slippage_multiplier
            else:
                slippage = trade_value * (self.slippage_bps / 10000.0)
        else:
            slippage = 0.0
        
        return commission + slippage

