"""Configuration loader with environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class UniverseConfig(BaseModel):
    """Universe configuration."""
    index_name: str = "SP500"
    constituents_csv_path: str = "data/sp500_constituents.csv"
    min_price_usd: float = 3.0
    min_dollar_volume_window: int = 20
    min_dollar_volume_percentile: int = 10
    use_survivorship_bias_free: bool = True


class DataConfig(BaseModel):
    """Data maintenance configuration."""
    min_history_start_date: str = "2020-01-01"
    max_history_lag_days: int = 1
    auto_fetch_on_backtest: bool = True
    auto_fetch_on_live: bool = True
    symbols: list = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    duckdb_path: str = "data/market.duckdb"
    data_root: str = "data"
    raw_vendor_dir: str = "data/raw_vendor"
    normalized_dir: str = "data/normalized"


# =============================================================================
# Retraining Policy Configuration
# =============================================================================

class TimeDecayConfig(BaseModel):
    """Time-decay weighting for training samples."""
    enabled: bool = True
    lambda_: float = Field(default=0.001, alias="lambda")  # Decay rate
    min_weight: float = 0.1  # Floor weight for old samples
    
    model_config = {"populate_by_name": True}


class AdaptiveRetrainingConfig(BaseModel):
    """Adaptive retraining triggers based on performance degradation."""
    enabled: bool = True
    sharpe_floor: float = -0.5  # Retrain if rolling Sharpe drops below
    hit_rate_floor: float = 0.40  # Retrain if hit rate drops below
    calibration_threshold: float = 2.0  # Retrain if variance ratio exceeds
    lookback_days: int = 60  # Window for rolling metrics


class ModelVersioningConfig(BaseModel):
    """Model versioning and stability settings."""
    enabled: bool = True
    artifact_dir: str = "artifacts/models"
    keep_versions: int = 10  # Number of old versions to retain
    stability_penalty: float = 0.1  # L2 penalty on prediction changes


class RetrainingConfig(BaseModel):
    """
    Complete retraining policy configuration.
    
    This controls how and when models are retrained to incorporate new information
    while maintaining stability. The policy applies consistently across backtests,
    simulations, and live trading.
    """
    # Cadence: how often to retrain (in trading days)
    cadence_days: int = 20  # ~monthly
    
    # History window settings
    window_type: str = "rolling"  # "rolling" or "expanding"
    window_years: int = 5  # Years of history to use
    
    # Time-decay weighting (Prophet-style bias to recent data)
    time_decay: TimeDecayConfig = Field(default_factory=TimeDecayConfig)
    
    # Adaptive retraining triggers
    adaptive: AdaptiveRetrainingConfig = Field(default_factory=AdaptiveRetrainingConfig)
    
    # Model versioning
    versioning: ModelVersioningConfig = Field(default_factory=ModelVersioningConfig)
    
    def get_window_start(self, as_of_date, trading_days_func=None) -> 'date':
        """
        Calculate the training window start date.
        
        Args:
            as_of_date: The current date (end of training window)
            trading_days_func: Optional function to count trading days
            
        Returns:
            Start date for training window
        """
        import pandas as pd
        from datetime import timedelta
        
        if self.window_type == "rolling":
            # Rolling window: go back window_years from as_of_date
            return (pd.Timestamp(as_of_date) - pd.DateOffset(years=self.window_years)).date()
        else:
            # Expanding window: use a fixed anchor (or minimum years)
            # For expanding, window_years is the minimum required
            return (pd.Timestamp(as_of_date) - pd.DateOffset(years=self.window_years)).date()
    
    def compute_sample_weights(self, dates: List, as_of_date) -> List[float]:
        """
        Compute time-decay weights for training samples.
        
        Args:
            dates: List of sample dates
            as_of_date: Reference date (most recent)
            
        Returns:
            List of weights, one per sample
        """
        import numpy as np
        from datetime import date as date_type
        
        if not self.time_decay.enabled:
            return [1.0] * len(dates)
        
        # Convert as_of_date to ordinal for calculation
        if hasattr(as_of_date, 'date'):
            as_of_date = as_of_date.date()
        ref_ordinal = as_of_date.toordinal()
        
        weights = []
        for d in dates:
            if hasattr(d, 'date'):
                d = d.date()
            elif isinstance(d, str):
                from datetime import datetime
                d = datetime.strptime(d, "%Y-%m-%d").date()
            
            age_days = ref_ordinal - d.toordinal()
            # Exponential decay: w = exp(-lambda * age)
            w = np.exp(-self.time_decay.lambda_ * age_days)
            # Apply floor
            w = max(w, self.time_decay.min_weight)
            weights.append(w)
        
        return weights
    
    def should_retrain(
        self,
        last_train_date,
        current_date,
        rolling_metrics: Optional[Dict[str, float]] = None
    ) -> tuple:
        """
        Determine if retraining is needed.
        
        Args:
            last_train_date: Date of last model training
            current_date: Current trading date
            rolling_metrics: Optional dict with 'sharpe', 'hit_rate', 'calibration_ratio'
            
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        from datetime import date as date_type
        
        # Convert dates if needed
        if hasattr(last_train_date, 'date'):
            last_train_date = last_train_date.date()
        if hasattr(current_date, 'date'):
            current_date = current_date.date()
        
        days_since_train = (current_date - last_train_date).days
        
        # Check cadence
        if days_since_train >= self.cadence_days:
            return True, f"scheduled (cadence={self.cadence_days} days)"
        
        # Check adaptive triggers
        if self.adaptive.enabled and rolling_metrics:
            sharpe = rolling_metrics.get('sharpe')
            hit_rate = rolling_metrics.get('hit_rate')
            calibration = rolling_metrics.get('calibration_ratio')
            
            if sharpe is not None and sharpe < self.adaptive.sharpe_floor:
                return True, f"adaptive: sharpe={sharpe:.2f} < floor={self.adaptive.sharpe_floor}"
            
            if hit_rate is not None and hit_rate < self.adaptive.hit_rate_floor:
                return True, f"adaptive: hit_rate={hit_rate:.2%} < floor={self.adaptive.hit_rate_floor:.2%}"
            
            if calibration is not None and calibration > self.adaptive.calibration_threshold:
                return True, f"adaptive: calibration={calibration:.2f} > threshold={self.adaptive.calibration_threshold}"
        
        return False, "not due"


class Config(BaseModel):
    """Main configuration model."""
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    universes: Dict[str, Any] = Field(default_factory=dict)  # Named universe definitions
    universe_defaults: Dict[str, str] = Field(default_factory=dict)  # Default universe per mode
    data: DataConfig = Field(default_factory=DataConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vendors: Dict[str, str] = Field(default_factory=dict)
    features: Dict[str, Any] = Field(default_factory=dict)
    labels: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    portfolio: Dict[str, Any] = Field(default_factory=dict)
    costs: Dict[str, Any] = Field(default_factory=dict)
    backtest: Dict[str, Any] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(default_factory=dict)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    live: Dict[str, Any] = Field(default_factory=dict)
    live_gates: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    benchmarks: Dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: str = "configs/base.yaml") -> Config:
    """Load configuration from YAML file with environment variable overrides."""
    # Load environment variables
    load_dotenv()
    
    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Override with environment variables if present
    if "TIINGO_API_KEY" in os.environ:
        config_dict.setdefault("vendors", {})["tiingo_api_key"] = os.environ["TIINGO_API_KEY"]
    if "FINNHUB_API_KEY" in os.environ:
        config_dict.setdefault("vendors", {})["finnhub_api_key"] = os.environ["FINNHUB_API_KEY"]
    if "DUCKDB_PATH" in os.environ:
        config_dict.setdefault("database", {})["duckdb_path"] = os.environ["DUCKDB_PATH"]
    if "DATA_ROOT" in os.environ:
        config_dict.setdefault("database", {})["data_root"] = os.environ["DATA_ROOT"]
    if "LOG_LEVEL" in os.environ:
        config_dict.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]
    
    # Handle retraining config specially to parse nested structures
    if "retraining" in config_dict:
        retraining_dict = config_dict["retraining"]
        # Handle time_decay.lambda -> lambda_ mapping
        if "time_decay" in retraining_dict and "lambda" in retraining_dict["time_decay"]:
            retraining_dict["time_decay"]["lambda_"] = retraining_dict["time_decay"].pop("lambda")
        config_dict["retraining"] = RetrainingConfig(**retraining_dict)
    
    return Config(**config_dict)


# Global config instance (lazy-loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

