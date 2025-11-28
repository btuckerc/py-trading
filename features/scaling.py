"""Feature scaling with leakage-free preprocessing."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle


class FeatureScaler:
    """
    Fits scalers on training data only and applies to validation/test.
    
    Prevents data leakage by ensuring scalers are never refit on validation/test data.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Args:
            scaler_type: "standard" or "robust"
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        self.scaler_type = scaler_type
        self.feature_names: Optional[list] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, feature_columns: Optional[list] = None):
        """
        Fit scaler on training data.
        
        Args:
            X: DataFrame with features
            feature_columns: List of column names to scale (if None, scales all numeric columns)
        """
        if feature_columns is None:
            # Select numeric columns only
            feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude date/asset_id columns
            feature_columns = [c for c in feature_columns if c not in ['date', 'asset_id']]
        
        self.feature_names = feature_columns
        
        # Extract feature matrix
        X_features = X[feature_columns].values
        
        # Fit scaler
        self.scaler.fit(X_features)
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: DataFrame with features
        
        Returns:
            DataFrame with scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        
        result = X.copy()
        
        # Extract and scale features
        X_features = X[self.feature_names].values
        X_scaled = self.scaler.transform(X_features)
        
        # Replace original columns with scaled values
        for idx, col in enumerate(self.feature_names):
            result[col] = X_scaled[:, idx]
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, feature_columns: Optional[list] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, feature_columns)
        return self.transform(X)
    
    def save(self, filepath: str):
        """Save scaler to disk."""
        scaler_data = {
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'center_': self.scaler.center_ if hasattr(self.scaler, 'center_') else None,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
    
    def load(self, filepath: str):
        """Load scaler from disk."""
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler_type = scaler_data['scaler_type']
        self.feature_names = scaler_data['feature_names']
        
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        
        # Restore scaler parameters
        if scaler_data['scaler_params']['mean_'] is not None:
            self.scaler.mean_ = scaler_data['scaler_params']['mean_']
        if scaler_data['scaler_params']['scale_'] is not None:
            self.scaler.scale_ = scaler_data['scaler_params']['scale_']
        if scaler_data['scaler_params']['center_'] is not None:
            self.scaler.center_ = scaler_data['scaler_params']['center_']
        
        self.is_fitted = True


class ScalerManager:
    """Manages multiple scalers for different training windows."""
    
    def __init__(self, scalers_dir: str = "artifacts/scalers"):
        self.scalers_dir = Path(scalers_dir)
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        self.scalers: Dict[str, FeatureScaler] = {}
    
    def get_scaler(self, window_name: str, scaler_type: str = "standard") -> FeatureScaler:
        """Get or create scaler for a training window."""
        if window_name not in self.scalers:
            scaler_path = self.scalers_dir / f"{window_name}.pkl"
            if scaler_path.exists():
                scaler = FeatureScaler(scaler_type)
                scaler.load(str(scaler_path))
                self.scalers[window_name] = scaler
            else:
                self.scalers[window_name] = FeatureScaler(scaler_type)
        
        return self.scalers[window_name]
    
    def save_scaler(self, window_name: str, scaler: FeatureScaler):
        """Save scaler for a training window."""
        scaler_path = self.scalers_dir / f"{window_name}.pkl"
        scaler.save(str(scaler_path))
        self.scalers[window_name] = scaler

