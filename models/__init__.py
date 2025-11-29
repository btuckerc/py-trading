"""Modeling layer: baselines, sequence models, uncertainty estimation."""

from models.tabular import XGBoostModel, LightGBMModel, EnsembleXGBoostModel
from models.training import WalkForwardRetrainer, SequenceTrainer
from models.monitoring import PerformanceMonitor, ModelPerformanceTracker
from models.versioning import ModelVersionManager, ModelVersion, save_live_model, load_live_model_with_validation
from models.tabular_trainer import (
    TabularTrainer,
    TrainingConfig,
    TrainingResult,
    SamplingStrategy,
    FeatureSchema,
    FeatureSchemaMismatchError,
    validate_feature_schema,
)
from models.types import (
    # Prediction types
    PredictionResult,
    AssetPrediction,
    PredictionBatch,
    # Backtest types
    PerformanceMetricsDict,
    BacktestResult,
    ComparisonSummary,
    # Live trading types
    OrderRequest,
    DailyTradingResult,
    # Exceptions
    TrainingError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelLoadError,
    PredictionError,
    DataQualityError,
    ConfigurationError,
    BrokerError,
    OrderSubmissionError,
    InsufficientBuyingPowerError,
)

__all__ = [
    # Tabular models
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleXGBoostModel",
    # Training
    "TabularTrainer",
    "TrainingConfig",
    "TrainingResult",
    "SamplingStrategy",
    "WalkForwardRetrainer",
    "SequenceTrainer",
    # Feature Schema
    "FeatureSchema",
    "FeatureSchemaMismatchError",
    "validate_feature_schema",
    # Monitoring
    "PerformanceMonitor",
    "ModelPerformanceTracker",
    # Versioning
    "ModelVersionManager",
    "ModelVersion",
    "save_live_model",
    "load_live_model_with_validation",
    # Prediction types
    "PredictionResult",
    "AssetPrediction",
    "PredictionBatch",
    # Backtest types
    "PerformanceMetricsDict",
    "BacktestResult",
    "ComparisonSummary",
    # Live trading types
    "OrderRequest",
    "DailyTradingResult",
    # Exceptions
    "TrainingError",
    "InsufficientDataError",
    "ModelNotFoundError",
    "ModelLoadError",
    "PredictionError",
    "DataQualityError",
    "ConfigurationError",
    "BrokerError",
    "OrderSubmissionError",
    "InsufficientBuyingPowerError",
]

