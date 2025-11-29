"""Tabular ML models (logistic regression, tree models)."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic regression for binary classification."""
    
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit logistic regression."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Predict class labels."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost for regression or classification."""
    
    def __init__(self, task_type: str = "regression", **kwargs):
        """
        Args:
            task_type: "regression" or "classification"
        """
        self.task_type = task_type
        self.model = None
        self.params = kwargs
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """
        Fit XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            sample_weight: Optional array of sample weights (e.g., time-decay weights)
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if self.task_type == "regression":
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)
        
        eval_set = None
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train, 
            eval_set=eval_set, 
            verbose=False,
            sample_weight=sample_weight
        )
    
    def predict(self, X):
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (for classification)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only available for classification tasks")


class EnsembleXGBoostModel(BaseModel):
    """
    Ensemble of XGBoost models for uncertainty estimation.
    
    Trains multiple XGBoost models with different random seeds and
    uses the variance across predictions as uncertainty estimate.
    
    Supports:
    - Sample weights for time-decay weighting
    - Multi-horizon predictions via predict_multi_horizon
    - Uncertainty estimation via predict_with_uncertainty
    """
    
    def __init__(
        self,
        task_type: str = "regression",
        n_models: int = 5,
        **kwargs
    ):
        self.task_type = task_type
        self.n_models = n_models
        self.models: List[xgb.XGBRegressor] = []
        self.params = kwargs
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """
        Fit ensemble of XGBoost models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            sample_weight: Optional array of sample weights (e.g., time-decay weights)
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        self.models = []
        
        for i in range(self.n_models):
            # Create model with different random seed
            params = self.params.copy()
            params['random_state'] = params.get('random_state', 42) + i
            
            if self.task_type == "regression":
                model = xgb.XGBRegressor(**params)
            else:
                model = xgb.XGBClassifier(**params)
            
            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(
                X_train, y_train, 
                eval_set=eval_set, 
                verbose=False,
                sample_weight=sample_weight
            )
            self.models.append(model)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions (mean of ensemble)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0)
    
    def predict_with_uncertainty(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.array([model.predict(X) for model in self.models])
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Ensure minimum uncertainty to avoid division by zero
        std_pred = np.maximum(std_pred, 1e-6)
        
        return mean_pred, std_pred
    
    def predict_quantiles(self, X, quantiles: List[float] = [0.1, 0.5, 0.9]) -> np.ndarray:
        """
        Predict quantiles from ensemble distribution.
        
        Args:
            X: Features
            quantiles: List of quantiles to compute (e.g., [0.1, 0.5, 0.9])
            
        Returns:
            Array of shape (n_samples, n_quantiles)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Compute quantiles across ensemble
        return np.percentile(predictions, [q * 100 for q in quantiles], axis=0).T
    
    def predict_proba(self, X):
        """Predict probabilities (for classification)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification tasks")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Average probabilities across ensemble
        proba_list = [model.predict_proba(X) for model in self.models]
        return np.mean(proba_list, axis=0)


class LightGBMModel(BaseModel):
    """LightGBM for regression or classification."""
    
    def __init__(self, task_type: str = "regression", **kwargs):
        self.task_type = task_type
        self.model = None
        self.params = kwargs
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """
        Fit LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            sample_weight: Optional array of sample weights (e.g., time-decay weights)
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if self.task_type == "regression":
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
            self.model = lgb.train(self.params, train_data)
        else:
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
            self.model = lgb.train(self.params, train_data)
    
    def predict(self, X):
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (for classification)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.task_type == "classification":
            return self.model.predict(X, raw_score=False)
        raise ValueError("predict_proba only available for classification tasks")


class RandomForestModel(BaseModel):
    """Random Forest for regression or classification."""
    
    def __init__(self, task_type: str = "regression", **kwargs):
        self.task_type = task_type
        if task_type == "regression":
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit Random Forest."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (for classification)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only available for classification tasks")

