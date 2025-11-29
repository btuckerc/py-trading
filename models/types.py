"""Type definitions for the models module.

This module provides:
- Structured types for training results, predictions, and configurations
- Domain-specific exceptions for clearer error handling
- TypedDict definitions for loosely-typed dict structures
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np


# =============================================================================
# Prediction Types
# =============================================================================

class PredictionResult(TypedDict):
    """Result of a single prediction with uncertainty."""
    mu: float
    sigma: float


class MultiHorizonPredictions(TypedDict, total=False):
    """Predictions for multiple horizons."""
    # Keys are horizon days (5, 20, etc.)
    # Values are tuples of (predictions_array, uncertainty_array)
    pass


@dataclass
class AssetPrediction:
    """Prediction for a single asset."""
    
    asset_id: int
    score: float
    confidence: float
    
    # Per-horizon predictions (optional)
    horizon_predictions: Optional[Dict[int, PredictionResult]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'asset_id': self.asset_id,
            'score': self.score,
            'confidence': self.confidence,
        }
        if self.horizon_predictions:
            result['horizon_predictions'] = self.horizon_predictions
        return result


@dataclass
class PredictionBatch:
    """Batch of predictions for multiple assets."""
    
    predictions: List[AssetPrediction]
    trading_date: date
    model_version: Optional[str] = None
    horizons: List[int] = field(default_factory=lambda: [20])
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([p.to_dict() for p in self.predictions])
    
    def get_scores_df(self):
        """Get scores DataFrame for strategy."""
        import pandas as pd
        return pd.DataFrame({
            'asset_id': [p.asset_id for p in self.predictions],
            'score': [p.score for p in self.predictions],
            'confidence': [p.confidence for p in self.predictions],
        })


# =============================================================================
# Backtest Result Types
# =============================================================================

class PerformanceMetricsDict(TypedDict, total=False):
    """Performance metrics from a backtest."""
    
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    var_5pct: float
    cvar_5pct: float


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    
    # Identification
    name: str
    description: Optional[str] = None
    
    # Performance
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Equity curve
    equity_curve: Optional[Any] = None  # pd.DataFrame
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional data
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # Error if failed
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if backtest completed successfully."""
        return self.error is None and len(self.metrics) > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'metrics': self.metrics,
            'config': self.config,
            'additional_data': self.additional_data,
            'error': self.error,
            'success': self.success,
        }


@dataclass
class ComparisonSummary:
    """Summary comparing multiple backtest results."""
    
    # Experiment metadata
    experiment_timestamp: str
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    
    # Backtest configuration
    backtest_config: Dict[str, Any] = field(default_factory=dict)
    
    # Results by strategy
    strategies: Dict[str, BacktestResult] = field(default_factory=dict)
    
    # Optional additional analyses
    cost_sensitivity: Optional[List[Dict]] = None
    walk_forward: Optional[Dict] = None
    regime_aware: Optional[Dict] = None
    ablation: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'experiment_metadata': {
                'timestamp': self.experiment_timestamp,
                'git_commit': self.git_commit,
                'config_hash': self.config_hash,
            },
            'backtest_config': self.backtest_config,
            'strategies': {k: v.to_dict() for k, v in self.strategies.items()},
            'cost_sensitivity': self.cost_sensitivity,
            'walk_forward': self.walk_forward,
            'regime_aware': self.regime_aware,
            'ablation': self.ablation,
        }


# =============================================================================
# Live Trading Types
# =============================================================================

@dataclass
class OrderRequest:
    """Request to submit an order."""
    
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str = 'market'
    time_in_force: str = 'day'
    
    # Optional metadata
    asset_id: Optional[int] = None
    target_weight: Optional[float] = None
    current_weight: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'asset_id': self.asset_id,
            'target_weight': self.target_weight,
            'current_weight': self.current_weight,
        }


@dataclass
class DailyTradingResult:
    """Result of a daily trading loop."""
    
    trading_date: date
    regime: str
    exposure_scale: float
    
    # Positions
    target_positions: Dict[int, float]  # asset_id -> weight
    current_positions: Dict[int, float]
    
    # Orders
    orders: List[OrderRequest]
    orders_submitted: int = 0
    
    # Statistics
    positive_score_count: int = 0
    universe_size: int = 0
    
    # Mode
    dry_run: bool = True
    
    # Error if any
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'trading_date': str(self.trading_date),
            'regime': self.regime,
            'exposure_scale': self.exposure_scale,
            'target_positions': self.target_positions,
            'orders_count': len(self.orders),
            'orders_submitted': self.orders_submitted,
            'positive_score_count': self.positive_score_count,
            'universe_size': self.universe_size,
            'dry_run': self.dry_run,
            'error': self.error,
        }


# =============================================================================
# Exception Types
# =============================================================================

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class InsufficientDataError(TrainingError):
    """Raised when there is insufficient data for training."""
    
    def __init__(self, message: str, required: int = 0, available: int = 0):
        self.required = required
        self.available = available
        super().__init__(f"{message} (required: {required}, available: {available})")


class ModelNotFoundError(TrainingError):
    """Raised when a required model is not found."""
    
    def __init__(self, model_path: str, message: Optional[str] = None):
        self.model_path = model_path
        msg = message or f"Model not found at {model_path}"
        super().__init__(msg)


class ModelLoadError(TrainingError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        self.model_path = model_path
        self.cause = cause
        msg = f"Failed to load model from {model_path}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class PredictionError(Exception):
    """Raised when prediction fails."""
    
    def __init__(self, message: str, model_type: Optional[str] = None):
        self.model_type = model_type
        super().__init__(message)


class DataQualityError(Exception):
    """Raised when data quality issues are detected."""
    
    def __init__(self, message: str, issues: Optional[List[str]] = None):
        self.issues = issues or []
        super().__init__(message)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(message)


class BrokerError(Exception):
    """Base exception for broker-related errors."""
    pass


class OrderSubmissionError(BrokerError):
    """Raised when order submission fails."""
    
    def __init__(self, symbol: str, side: str, quantity: int, cause: Optional[str] = None):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.cause = cause
        msg = f"Failed to submit {side} order for {quantity} shares of {symbol}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class InsufficientBuyingPowerError(BrokerError):
    """Raised when there is insufficient buying power."""
    
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient buying power: required ${required:,.2f}, available ${available:,.2f}"
        )


# =============================================================================
# Type Aliases
# =============================================================================

# Asset ID type (for clarity)
AssetId = int

# Symbol type
Symbol = str

# Weight type (portfolio weight as fraction)
Weight = float

# Universe type
Universe = set[AssetId]

# Position mapping
PositionMap = Dict[AssetId, Weight]

# Feature matrix type
FeatureMatrix = Union['pd.DataFrame', 'np.ndarray']

# Prediction array
PredictionArray = np.ndarray

