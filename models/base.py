"""Base model interface."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit model on training data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def predict_proba(self, X):
        """Predict probabilities (for classification models)."""
        raise NotImplementedError("predict_proba not implemented for this model")


class MultiHorizonModel(BaseModel):
    """Base class for models that predict multiple horizons."""
    
    @abstractmethod
    def predict_multi_horizon(self, X):
        """
        Predict for multiple horizons.
        
        Returns:
            Dictionary mapping horizon -> predictions
        """
        pass

