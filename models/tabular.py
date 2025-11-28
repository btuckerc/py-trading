"""Tabular ML models (logistic regression, tree models)."""

import pandas as pd
import numpy as np
from typing import Optional
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit XGBoost model."""
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
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
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


class LightGBMModel(BaseModel):
    """LightGBM for regression or classification."""
    
    def __init__(self, task_type: str = "regression", **kwargs):
        self.task_type = task_type
        self.model = None
        self.params = kwargs
        self.feature_names = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit LightGBM model."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if self.task_type == "regression":
            train_data = lgb.Dataset(X_train, label=y_train)
            self.model = lgb.train(self.params, train_data)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
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

