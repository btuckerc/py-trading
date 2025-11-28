"""Risk management and constraints."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Set, Tuple, List
from datetime import date
from loguru import logger


# Default regime exposure multipliers
DEFAULT_REGIME_EXPOSURE = {
    'bull_low_vol': 1.0,
    'bull_high_vol': 0.8,
    'bear_low_vol': 0.5,
    'bear_high_vol': 0.25,
    'unknown': 0.5,  # Conservative default
}


class ExposureManager:
    """
    Manages portfolio exposure based on regime and volatility.
    
    Combines three exposure scaling mechanisms:
    1. Regime-based scaling (from regime_policy)
    2. Volatility-based scaling (VIX-style target vol)
    3. Drawdown-based scaling (from DrawdownManager)
    
    Final exposure = regime_scale * vol_scale * drawdown_scale * base_weights
    """
    
    def __init__(
        self,
        regime_policy: Optional[Dict[str, float]] = None,
        volatility_config: Optional[Dict] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.regime_policy = regime_policy or DEFAULT_REGIME_EXPOSURE
        
        # Volatility scaling config
        vol_config = volatility_config or {}
        self.vol_scaling_enabled = vol_config.get('enabled', True)
        self.target_vol = vol_config.get('target_vol', 0.15)
        self.vol_floor = vol_config.get('vol_floor', 0.10)
        self.vol_ceiling = vol_config.get('vol_ceiling', 0.60)
        
        # State tracking
        self.current_regime = 'unknown'
        self.current_regime_scale = 1.0
        self.current_vol_scale = 1.0
        self.current_realized_vol = 0.15
    
    def update_regime(self, regime_descriptor: str) -> float:
        """
        Update current regime and return the regime-based scale factor.
        
        Args:
            regime_descriptor: One of bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
        
        Returns:
            Regime-based exposure scale factor (0.0 to 1.0)
        """
        self.current_regime = regime_descriptor
        self.current_regime_scale = self.regime_policy.get(regime_descriptor, 0.5)
        
        logger.debug(f"Regime updated to {regime_descriptor}, scale={self.current_regime_scale:.2f}")
        return self.current_regime_scale
    
    def update_volatility(self, realized_vol: float) -> float:
        """
        Update volatility-based scaling factor.
        
        Uses inverse volatility scaling: scale = target_vol / realized_vol
        Clamped by vol_floor and vol_ceiling.
        
        Args:
            realized_vol: Current annualized realized volatility
        
        Returns:
            Volatility-based exposure scale factor
        """
        if not self.vol_scaling_enabled:
            self.current_vol_scale = 1.0
            return 1.0
        
        self.current_realized_vol = realized_vol
        
        # Clamp realized vol to floor/ceiling
        clamped_vol = np.clip(realized_vol, self.vol_floor, self.vol_ceiling)
        
        # Inverse vol scaling
        self.current_vol_scale = self.target_vol / clamped_vol
        
        # Cap at 1.0 to prevent leverage
        self.current_vol_scale = min(self.current_vol_scale, 1.0)
        
        logger.debug(
            f"Vol scaling: realized={realized_vol:.2%}, "
            f"clamped={clamped_vol:.2%}, scale={self.current_vol_scale:.2f}"
        )
        return self.current_vol_scale
    
    def get_combined_scale(self, drawdown_scale: float = 1.0) -> float:
        """
        Get the combined exposure scale factor.
        
        Args:
            drawdown_scale: Scale factor from DrawdownManager (0.0 to 1.0)
        
        Returns:
            Combined scale factor = regime_scale * vol_scale * drawdown_scale
        """
        if not self.enabled:
            return 1.0
        
        combined = self.current_regime_scale * self.current_vol_scale * drawdown_scale
        
        logger.debug(
            f"Combined exposure scale: regime={self.current_regime_scale:.2f} * "
            f"vol={self.current_vol_scale:.2f} * dd={drawdown_scale:.2f} = {combined:.2f}"
        )
        return combined
    
    def apply_exposure_scale(
        self,
        weights_dict: Dict[int, float],
        drawdown_scale: float = 1.0
    ) -> Dict[int, float]:
        """
        Apply combined exposure scaling to portfolio weights.
        
        Args:
            weights_dict: Target weights before scaling
            drawdown_scale: Scale factor from DrawdownManager
        
        Returns:
            Scaled weights dictionary
        """
        scale = self.get_combined_scale(drawdown_scale)
        
        if scale >= 1.0:
            return weights_dict
        
        return {k: v * scale for k, v in weights_dict.items()}
    
    def get_state(self) -> Dict:
        """Get current state for logging/debugging."""
        return {
            'regime': self.current_regime,
            'regime_scale': self.current_regime_scale,
            'vol_scale': self.current_vol_scale,
            'realized_vol': self.current_realized_vol,
            'combined_scale': self.get_combined_scale(),
        }


class SectorTiltManager:
    """
    Manages sector tilts based on market regime.
    
    Adjusts sector weights based on regime-specific policies:
    - In bear/high-vol regimes: overweight defensive sectors (Staples, Health Care, Utilities)
    - In bull/low-vol regimes: allow cyclical/growth sectors (Tech, Consumer Discretionary)
    """
    
    # Default defensive sectors
    DEFENSIVE_SECTORS = {'Consumer Staples', 'Health Care', 'Utilities', 'Real Estate'}
    CYCLICAL_SECTORS = {'Technology', 'Consumer Discretionary', 'Financials', 'Industrials', 'Materials'}
    
    def __init__(
        self,
        sector_policy: Optional[Dict[str, Dict[str, float]]] = None,
        enabled: bool = True
    ):
        """
        Initialize sector tilt manager.
        
        Args:
            sector_policy: Dict mapping regime_descriptor -> {sector: tilt_adjustment}
                          Positive values = overweight, negative = underweight
            enabled: Whether sector tilting is active
        """
        self.enabled = enabled
        self.sector_policy = sector_policy or self._default_sector_policy()
        self.current_regime = 'unknown'
    
    def _default_sector_policy(self) -> Dict[str, Dict[str, float]]:
        """Return default sector tilts per regime."""
        return {
            'bull_low_vol': {
                'Technology': 0.05,
                'Consumer Discretionary': 0.05,
                'Financials': 0.03,
                'Utilities': -0.05,
                'Consumer Staples': -0.05,
            },
            'bull_high_vol': {
                'Health Care': 0.03,
                'Technology': 0.02,
                'Consumer Discretionary': -0.02,
            },
            'bear_low_vol': {
                'Consumer Staples': 0.10,
                'Health Care': 0.08,
                'Utilities': 0.08,
                'Technology': -0.10,
                'Consumer Discretionary': -0.10,
                'Financials': -0.05,
            },
            'bear_high_vol': {
                'Consumer Staples': 0.15,
                'Health Care': 0.10,
                'Utilities': 0.10,
                'Technology': -0.15,
                'Consumer Discretionary': -0.15,
                'Financials': -0.10,
                'Industrials': -0.05,
            },
        }
    
    def update_regime(self, regime_descriptor: str):
        """Update current regime."""
        self.current_regime = regime_descriptor
    
    def get_sector_tilts(self, regime_descriptor: Optional[str] = None) -> Dict[str, float]:
        """
        Get sector tilt adjustments for the current or specified regime.
        
        Returns:
            Dict mapping sector name -> tilt adjustment
        """
        if not self.enabled:
            return {}
        
        regime = regime_descriptor or self.current_regime
        return self.sector_policy.get(regime, {})
    
    def apply_sector_tilts(
        self,
        weights_dict: Dict[int, float],
        sector_mapping: Dict[int, str],
        regime_descriptor: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Apply sector tilts to portfolio weights.
        
        Args:
            weights_dict: Dict of asset_id -> weight
            sector_mapping: Dict of asset_id -> sector name
            regime_descriptor: Optional regime to use (defaults to current)
        
        Returns:
            Adjusted weights dict
        """
        if not self.enabled or len(weights_dict) == 0:
            return weights_dict
        
        tilts = self.get_sector_tilts(regime_descriptor)
        if not tilts:
            return weights_dict
        
        result = weights_dict.copy()
        
        # Group assets by sector
        sector_assets: Dict[str, List[int]] = {}
        for asset_id in result.keys():
            sector = sector_mapping.get(asset_id, 'Unknown')
            if sector not in sector_assets:
                sector_assets[sector] = []
            sector_assets[sector].append(asset_id)
        
        # Apply tilts by adjusting weights within each sector
        for sector, tilt in tilts.items():
            if sector not in sector_assets:
                continue
            
            assets_in_sector = sector_assets[sector]
            if len(assets_in_sector) == 0:
                continue
            
            # Calculate current sector weight
            current_sector_weight = sum(result.get(aid, 0) for aid in assets_in_sector)
            
            # Target sector weight = current + tilt (bounded by 0)
            target_sector_weight = max(0, current_sector_weight + tilt)
            
            if current_sector_weight > 0:
                # Scale all assets in sector proportionally
                scale = target_sector_weight / current_sector_weight
                for asset_id in assets_in_sector:
                    if asset_id in result:
                        result[asset_id] *= scale
        
        # Renormalize to sum to original total weight
        original_total = sum(weights_dict.values())
        new_total = sum(result.values())
        
        if new_total > 0 and original_total > 0:
            scale = original_total / new_total
            result = {k: v * scale for k, v in result.items()}
        
        return result
    
    def score_sector_defensiveness(self, sector: str) -> float:
        """
        Score a sector's defensiveness (0 = cyclical, 1 = defensive).
        
        Useful for filtering or weighting decisions.
        """
        if sector in self.DEFENSIVE_SECTORS:
            return 1.0
        elif sector in self.CYCLICAL_SECTORS:
            return 0.0
        else:
            return 0.5  # Neutral


class RiskManager:
    """
    Applies risk constraints to portfolio weights.
    
    Supports:
    - Per-position caps
    - Sector caps and tilts (regime-aware)
    - Gross/net exposure limits
    - Stop-loss checks
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.20,  # Max 20% per position (was 5%)
        min_position_pct: float = 0.02,  # Min 2% to avoid tiny positions
        max_sector_pct: float = 0.40,    # Max 40% per sector
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 1.0,   # 1.0 for long-only
        stop_loss_pct: float = 0.10,
        sector_mapping: Optional[Dict[int, str]] = None,
        sector_tilt_manager: Optional[SectorTiltManager] = None
    ):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.stop_loss_pct = stop_loss_pct
        self.sector_mapping = sector_mapping or {}
        self.sector_tilt_manager = sector_tilt_manager
    
    def set_sector_mapping(self, mapping: Dict[int, str]):
        """Set the asset_id to sector mapping."""
        self.sector_mapping = mapping
    
    def apply_constraints(
        self,
        weights_dict: Dict[int, float],
        as_of_date: Optional[date] = None,
        current_regime: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Apply risk constraints to weights.
        
        Args:
            weights_dict: Dictionary mapping asset_id -> weight
            as_of_date: Date for sector/other checks (optional for now)
            current_regime: Current regime descriptor for sector tilts
        
        Returns:
            Constrained weights dictionary
        """
        if len(weights_dict) == 0:
            return weights_dict
        
        constrained = weights_dict.copy()
        
        # Step 1: Per-asset caps (max position size)
        capped = False
        for asset_id, weight in constrained.items():
            if abs(weight) > self.max_position_pct:
                constrained[asset_id] = np.sign(weight) * self.max_position_pct
                capped = True
        
        # Step 2: Redistribute excess weight from capped positions
        if capped:
            constrained = self._redistribute_weights(constrained)
        
        # Step 3: Apply sector tilts if sector tilt manager is available
        if self.sector_tilt_manager is not None and len(self.sector_mapping) > 0:
            constrained = self.sector_tilt_manager.apply_sector_tilts(
                constrained,
                self.sector_mapping,
                current_regime
            )
        
        # Step 4: Apply sector caps if sector mapping is available
        if len(self.sector_mapping) > 0:
            constrained = self._apply_sector_caps(constrained)
        
        # Step 5: Gross exposure limit
        gross_exposure = sum(abs(w) for w in constrained.values())
        if gross_exposure > self.max_gross_exposure:
            scale_factor = self.max_gross_exposure / gross_exposure
            constrained = {k: v * scale_factor for k, v in constrained.items()}
        
        # Step 6: Net exposure limit (for long/short strategies)
        net_exposure = sum(constrained.values())
        if abs(net_exposure) > self.max_net_exposure:
            target_net = np.sign(net_exposure) * self.max_net_exposure
            adjustment = target_net - net_exposure
            
            long_positions = {k: v for k, v in constrained.items() if v > 0}
            short_positions = {k: v for k, v in constrained.items() if v < 0}
            
            if adjustment > 0 and len(long_positions) > 0:
                for asset_id in long_positions:
                    constrained[asset_id] += adjustment / len(long_positions)
            elif adjustment < 0 and len(short_positions) > 0:
                for asset_id in short_positions:
                    constrained[asset_id] += adjustment / len(short_positions)
        
        # Step 7: Remove positions below minimum threshold
        constrained = {k: v for k, v in constrained.items() 
                      if abs(v) >= self.min_position_pct}
        
        return constrained
    
    def _redistribute_weights(self, weights_dict: Dict[int, float]) -> Dict[int, float]:
        """Redistribute excess weight from capped positions to uncapped ones."""
        result = weights_dict.copy()
        
        # Calculate excess from capped positions
        excess = 0.0
        uncapped_assets = []
        
        for asset_id, weight in result.items():
            if abs(weight) >= self.max_position_pct:
                excess += abs(weight) - self.max_position_pct
            else:
                uncapped_assets.append(asset_id)
        
        # Redistribute excess to uncapped positions proportionally
        if excess > 0 and len(uncapped_assets) > 0:
            uncapped_total = sum(abs(result[a]) for a in uncapped_assets)
            if uncapped_total > 0:
                for asset_id in uncapped_assets:
                    current = result[asset_id]
                    share = abs(current) / uncapped_total
                    additional = excess * share
                    new_weight = current + np.sign(current) * additional
                    # Cap again if redistribution exceeds max
                    result[asset_id] = min(abs(new_weight), self.max_position_pct) * np.sign(new_weight)
        
        return result
    
    def _apply_sector_caps(self, weights_dict: Dict[int, float]) -> Dict[int, float]:
        """Apply sector concentration limits."""
        result = weights_dict.copy()
        
        # Group by sector
        sector_weights: Dict[str, float] = {}
        sector_assets: Dict[str, list] = {}
        
        for asset_id, weight in result.items():
            sector = self.sector_mapping.get(asset_id, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)
            if sector not in sector_assets:
                sector_assets[sector] = []
            sector_assets[sector].append(asset_id)
        
        # Scale down sectors that exceed cap
        for sector, total_weight in sector_weights.items():
            if total_weight > self.max_sector_pct:
                scale_factor = self.max_sector_pct / total_weight
                for asset_id in sector_assets[sector]:
                    result[asset_id] *= scale_factor
        
        return result
    
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


class DrawdownManager:
    """
    Manages portfolio drawdown and implements throttling.
    
    When portfolio drawdown exceeds thresholds, positions are scaled down
    to reduce risk exposure.
    """
    
    def __init__(
        self,
        throttle_threshold_pct: float = 0.15,  # Start throttling at 15% DD
        max_drawdown_pct: float = 0.25,        # Max acceptable DD before full throttle
        min_scale_factor: float = 0.25,        # Minimum position scale at max DD
        recovery_threshold_pct: float = 0.05   # Resume full exposure after DD < 5%
    ):
        self.throttle_threshold_pct = throttle_threshold_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_scale_factor = min_scale_factor
        self.recovery_threshold_pct = recovery_threshold_pct
        
        # State tracking
        self.peak_equity = None
        self.current_scale_factor = 1.0
        self.is_throttled = False
    
    def update(self, current_equity: float) -> float:
        """
        Update drawdown state and return position scale factor.
        
        Args:
            current_equity: Current portfolio equity value
        
        Returns:
            Scale factor to apply to positions (0.0 to 1.0)
        """
        # Initialize peak if first call
        if self.peak_equity is None:
            self.peak_equity = current_equity
            return 1.0
        
        # Update peak (high-water mark)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check for recovery
        if self.is_throttled and drawdown < self.recovery_threshold_pct:
            self.is_throttled = False
            self.current_scale_factor = 1.0
            logger.info(f"Drawdown recovered to {drawdown:.1%}, resuming full exposure")
            return 1.0
        
        # Check if we need to throttle
        if drawdown >= self.throttle_threshold_pct:
            self.is_throttled = True
            
            # Linear interpolation between throttle threshold and max DD
            if drawdown >= self.max_drawdown_pct:
                self.current_scale_factor = self.min_scale_factor
            else:
                # Scale linearly from 1.0 at throttle_threshold to min_scale at max_dd
                dd_range = self.max_drawdown_pct - self.throttle_threshold_pct
                dd_progress = (drawdown - self.throttle_threshold_pct) / dd_range
                self.current_scale_factor = 1.0 - (1.0 - self.min_scale_factor) * dd_progress
            
            logger.warning(
                f"Drawdown throttle active: DD={drawdown:.1%}, "
                f"scale_factor={self.current_scale_factor:.2f}"
            )
        
        return self.current_scale_factor
    
    def get_current_drawdown(self, current_equity: float) -> float:
        """Get current drawdown percentage."""
        if self.peak_equity is None or self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - current_equity) / self.peak_equity
    
    def reset(self, initial_equity: float = None):
        """Reset drawdown tracking state."""
        self.peak_equity = initial_equity
        self.current_scale_factor = 1.0
        self.is_throttled = False
    
    def apply_throttle(
        self,
        weights_dict: Dict[int, float],
        current_equity: float
    ) -> Dict[int, float]:
        """
        Apply drawdown throttle to weights.
        
        Args:
            weights_dict: Target weights before throttling
            current_equity: Current portfolio equity
        
        Returns:
            Throttled weights
        """
        scale_factor = self.update(current_equity)
        
        if scale_factor >= 1.0:
            return weights_dict
        
        # Scale down all positions
        return {k: v * scale_factor for k, v in weights_dict.items()}

