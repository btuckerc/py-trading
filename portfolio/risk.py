"""Risk management and constraints."""

import numpy as np
from typing import Dict, Optional
from datetime import date


class RiskManager:
    """Applies risk constraints to portfolio weights."""
    
    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 0.2,
        stop_loss_pct: float = 0.10
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.stop_loss_pct = stop_loss_pct
    
    def apply_constraints(
        self,
        weights_dict: Dict[int, float],
        as_of_date: Optional[date] = None
    ) -> Dict[int, float]:
        """
        Apply risk constraints to weights.
        
        Args:
            weights_dict: Dictionary mapping asset_id -> weight
            as_of_date: Date for sector/other checks (optional for now)
        
        Returns:
            Constrained weights dictionary
        """
        constrained = weights_dict.copy()
        
        # Per-asset caps
        for asset_id, weight in constrained.items():
            if abs(weight) > self.max_position_pct:
                constrained[asset_id] = np.sign(weight) * self.max_position_pct
        
        # Gross exposure limit
        gross_exposure = sum(abs(w) for w in constrained.values())
        if gross_exposure > self.max_gross_exposure:
            # Scale down proportionally
            scale_factor = self.max_gross_exposure / gross_exposure
            constrained = {k: v * scale_factor for k, v in constrained.items()}
        
        # Net exposure limit
        net_exposure = sum(constrained.values())
        if abs(net_exposure) > self.max_net_exposure:
            # Adjust to meet net exposure constraint
            # (Simplified - would need more sophisticated rebalancing)
            target_net = np.sign(net_exposure) * self.max_net_exposure
            adjustment = target_net - net_exposure
            
            # Distribute adjustment proportionally
            long_positions = {k: v for k, v in constrained.items() if v > 0}
            short_positions = {k: v for k, v in constrained.items() if v < 0}
            
            if adjustment > 0 and len(long_positions) > 0:
                # Increase longs
                for asset_id in long_positions:
                    constrained[asset_id] += adjustment / len(long_positions)
            elif adjustment < 0 and len(short_positions) > 0:
                # Increase shorts
                for asset_id in short_positions:
                    constrained[asset_id] += adjustment / len(short_positions)
        
        # Sector caps (would need sector information - placeholder)
        # This would require merging with asset sector data
        
        return constrained
    
    def check_stop_loss(
        self,
        asset_id: int,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check if stop loss should be triggered."""
        if entry_price <= 0:
            return False
        
        loss_pct = (current_price - entry_price) / entry_price
        return loss_pct <= -self.stop_loss_pct

