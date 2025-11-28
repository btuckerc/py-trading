"""Configuration loader with environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
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
    live: Dict[str, Any] = Field(default_factory=dict)
    live_gates: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


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
    
    return Config(**config_dict)


# Global config instance (lazy-loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

