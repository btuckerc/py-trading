"""Loss functions for sequence models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MultiHorizonLoss(nn.Module):
    """Loss function for multi-horizon prediction."""
    
    def __init__(
        self,
        horizons: List[int],
        horizon_weights: Optional[Dict[int, float]] = None,
        loss_type: str = "mse"
    ):
        """
        Args:
            horizons: List of prediction horizons
            horizon_weights: Optional weights for each horizon
            loss_type: "mse" or "gaussian_nll"
        """
        super(MultiHorizonLoss, self).__init__()
        
        self.horizons = horizons
        self.horizon_weights = horizon_weights or {h: 1.0 for h in horizons}
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='mean')
        elif loss_type == "gaussian_nll":
            self.base_loss = self._gaussian_nll
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def _gaussian_nll(self, mean, logvar, target):
        """Gaussian negative log-likelihood."""
        precision = torch.exp(-logvar)
        return 0.5 * (precision * (target - mean) ** 2 + logvar)
    
    def forward(self, predictions: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor]):
        """
        Compute loss.
        
        Args:
            predictions: Dict mapping horizon -> predictions (or (mean, logvar) for gaussian_nll)
            targets: Dict mapping horizon -> target values
        """
        total_loss = 0.0
        
        for horizon in self.horizons:
            if horizon not in predictions or horizon not in targets:
                continue
            
            pred = predictions[horizon]
            target = targets[horizon]
            
            if self.loss_type == "gaussian_nll":
                if isinstance(pred, tuple):
                    mean, logvar = pred
                    loss = self.base_loss(mean, logvar, target).mean()
                else:
                    # Fallback to MSE if not tuple
                    loss = self.base_loss(pred, target)
            else:
                loss = self.base_loss(pred, target)
            
            weight = self.horizon_weights.get(horizon, 1.0)
            total_loss += weight * loss
        
        return total_loss

