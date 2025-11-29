"""Centralized tabular model training service.

This module provides a unified training interface used by:
- WalkForwardRetrainer (models/training.py)
- run_backtest.py (scripts)
- run_live_loop.py (scripts)

By consolidating training logic here, we ensure consistent behavior across
backtests, simulations, and live trading.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import numpy as np
import pandas as pd
from loguru import logger

from models.base import BaseModel


# =============================================================================
# Types and Dataclasses
# =============================================================================

@dataclass
class SamplingStrategy:
    """Configuration for how to sample training dates."""
    
    # Sample every N-th day (1 = every day, 5 = every 5th day)
    sample_every_n_days: int = 5
    
    # Minimum samples required to proceed with training
    min_samples: int = 100
    
    # Optional custom date filter function
    date_filter: Optional[Callable[[date], bool]] = None


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    
    # Training window
    window_start: date
    window_end: date
    
    # Horizons to train for
    horizons: List[int] = field(default_factory=lambda: [20])
    
    # Sampling strategy
    sampling: SamplingStrategy = field(default_factory=SamplingStrategy)
    
    # Time-decay weighting
    time_decay_enabled: bool = True
    time_decay_lambda: float = 0.001
    time_decay_min_weight: float = 0.1
    
    # Feature lookback for pipeline
    feature_lookback_days: int = 252
    
    # Label settings
    benchmark_symbol: str = "SPY"
    label_column: str = "target_excess_log_return"


@dataclass
class TrainingResult:
    """Result of a training run, including model and metadata."""
    
    # The trained model
    model: Any
    
    # Feature schema
    feature_names: List[str]
    feature_count: int
    feature_hash: str
    
    # Training metadata
    trained_date: date
    window_start: date
    window_end: date
    horizons: List[int]
    
    # Sample statistics
    num_samples: int
    num_training_dates: int
    
    # Weighting info
    time_decay_enabled: bool
    time_decay_lambda: Optional[float]
    sample_weight_range: Optional[Tuple[float, float]]
    
    # Config snapshot for reproducibility
    config_hash: str
    
    # Optional metrics from training
    training_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'feature_names': self.feature_names,
            'feature_count': self.feature_count,
            'feature_hash': self.feature_hash,
            'trained_date': str(self.trained_date),
            'window_start': str(self.window_start),
            'window_end': str(self.window_end),
            'horizons': self.horizons,
            'num_samples': self.num_samples,
            'num_training_dates': self.num_training_dates,
            'time_decay_enabled': self.time_decay_enabled,
            'time_decay_lambda': self.time_decay_lambda,
            'sample_weight_range': self.sample_weight_range,
            'config_hash': self.config_hash,
            'training_metrics': self.training_metrics,
        }


# =============================================================================
# Feature Schema
# =============================================================================

@dataclass
class FeatureSchema:
    """
    Defines the expected feature schema for a model.
    
    Used for validation to catch schema drift early.
    """
    
    feature_names: List[str]
    feature_hash: str
    horizons: List[int]
    pipeline_version: Optional[str] = None
    created_date: Optional[date] = None
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, horizons: List[int], exclude_cols: Optional[List[str]] = None) -> 'FeatureSchema':
        """Create schema from a features DataFrame."""
        exclude = set(exclude_cols or ['asset_id', 'date', 'target_excess_log_return'])
        feature_names = sorted([c for c in df.columns if c not in exclude])
        feature_hash = cls._compute_hash(feature_names)
        return cls(
            feature_names=feature_names,
            feature_hash=feature_hash,
            horizons=horizons,
            created_date=date.today(),
        )
    
    @staticmethod
    def _compute_hash(feature_names: List[str]) -> str:
        """Compute a hash of feature names for quick comparison."""
        names_str = ','.join(sorted(feature_names))
        return hashlib.md5(names_str.encode()).hexdigest()[:12]
    
    def validate(self, current_features: List[str], strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate current features against this schema.
        
        Args:
            current_features: List of current feature names
            strict: If True, require exact match; if False, allow superset
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        current_set = set(current_features)
        expected_set = set(self.feature_names)
        
        # Check for missing features
        missing = expected_set - current_set
        if missing:
            issues.append(f"Missing features: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        
        # Check for extra features (only in strict mode)
        extra = current_set - expected_set
        if extra and strict:
            issues.append(f"Unexpected features: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
        
        # Check count
        if len(current_features) != len(self.feature_names):
            issues.append(f"Feature count mismatch: expected {len(self.feature_names)}, got {len(current_features)}")
        
        # Check hash
        current_hash = self._compute_hash(current_features)
        if current_hash != self.feature_hash:
            issues.append(f"Feature hash mismatch: expected {self.feature_hash}, got {current_hash}")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'feature_names': self.feature_names,
            'feature_hash': self.feature_hash,
            'horizons': self.horizons,
            'pipeline_version': self.pipeline_version,
            'created_date': str(self.created_date) if self.created_date else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureSchema':
        """Create from dictionary."""
        from datetime import datetime
        created = None
        if data.get('created_date'):
            created = datetime.strptime(data['created_date'], '%Y-%m-%d').date()
        return cls(
            feature_names=data['feature_names'],
            feature_hash=data['feature_hash'],
            horizons=data['horizons'],
            pipeline_version=data.get('pipeline_version'),
            created_date=created,
        )


class FeatureSchemaMismatchError(Exception):
    """Raised when feature schema validation fails."""
    
    def __init__(self, issues: List[str]):
        self.issues = issues
        super().__init__(f"Feature schema mismatch: {'; '.join(issues)}")


# =============================================================================
# TabularTrainer
# =============================================================================

class TabularTrainer:
    """
    Centralized training service for tabular ML models.
    
    Handles the common training loop:
    1. Sample training dates
    2. Build features for each date
    3. Merge with labels
    4. Sanitize numeric data
    5. Compute time-decay sample weights
    6. Fit model
    7. Return structured result with metadata
    
    Example:
        trainer = TabularTrainer(
            feature_pipeline=pipeline,
            label_generator=label_gen,
            storage=storage,
            api=api,
        )
        
        result = trainer.train(
            model_class=XGBoostModel,
            model_params={'task_type': 'regression', 'n_estimators': 100},
            config=TrainingConfig(
                window_start=date(2020, 1, 1),
                window_end=date(2023, 12, 31),
                horizons=[20],
            ),
            universe=universe_set,
        )
    """
    
    def __init__(
        self,
        feature_pipeline,  # FeaturePipeline
        label_generator,   # ReturnLabelGenerator
        storage,           # StorageBackend
        api,               # AsOfQueryAPI
    ):
        self.feature_pipeline = feature_pipeline
        self.label_generator = label_generator
        self.storage = storage
        self.api = api
    
    def train(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        config: TrainingConfig,
        universe: Set[int],
        validation_split: float = 0.0,
    ) -> TrainingResult:
        """
        Train a model using the specified configuration.
        
        Args:
            model_class: Model class to instantiate (e.g., XGBoostModel)
            model_params: Parameters to pass to model constructor
            config: Training configuration
            universe: Set of asset_ids to train on
            validation_split: Fraction of data to hold out for validation (0 = no validation)
            
        Returns:
            TrainingResult with trained model and metadata
            
        Raises:
            ValueError: If insufficient training data
        """
        logger.info(f"TabularTrainer: Starting training from {config.window_start} to {config.window_end}")
        
        # Step 1: Generate labels
        labels_df = self._generate_labels(config, universe)
        if len(labels_df) == 0:
            raise ValueError(f"No labels generated for {config.window_start} to {config.window_end}")
        logger.info(f"Generated {len(labels_df)} labels")
        
        # Step 2: Sample training dates
        train_dates = self._sample_training_dates(config)
        logger.info(f"Sampled {len(train_dates)} training dates")
        
        # Step 3: Build features and merge with labels
        X_list, y_list, date_list = self._build_training_data(
            train_dates, labels_df, config, universe
        )
        
        if len(X_list) == 0:
            raise ValueError("No training data generated")
        
        X_train = pd.concat(X_list, ignore_index=True)
        y_train = np.concatenate(y_list)
        
        logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
        
        # Step 4: Compute sample weights
        sample_weights = None
        weight_range = None
        if config.time_decay_enabled:
            sample_weights = self._compute_time_decay_weights(
                date_list, config.window_end, config.time_decay_lambda, config.time_decay_min_weight
            )
            weight_range = (float(np.min(sample_weights)), float(np.max(sample_weights)))
            logger.info(f"Sample weight range: [{weight_range[0]:.3f}, {weight_range[1]:.3f}]")
        
        # Step 5: Split validation if requested
        X_val, y_val = None, None
        if validation_split > 0:
            split_idx = int(len(X_train) * (1 - validation_split))
            X_val = X_train.iloc[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train[:split_idx]
            if sample_weights is not None:
                sample_weights = sample_weights[:split_idx]
            logger.info(f"Validation split: {len(X_train)} train, {len(X_val)} val")
        
        # Step 6: Train model
        model = model_class(**model_params)
        
        # Check if model supports sample_weight
        fit_kwargs = {}
        if sample_weights is not None and self._model_supports_sample_weight(model):
            fit_kwargs['sample_weight'] = sample_weights
        
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)
        logger.info("Model training complete")
        
        # Step 7: Build result
        feature_names = list(X_train.columns)
        feature_hash = FeatureSchema._compute_hash(feature_names)
        config_hash = self._compute_config_hash(config, model_params)
        
        result = TrainingResult(
            model=model,
            feature_names=feature_names,
            feature_count=len(feature_names),
            feature_hash=feature_hash,
            trained_date=config.window_end,
            window_start=config.window_start,
            window_end=config.window_end,
            horizons=config.horizons,
            num_samples=len(X_train),
            num_training_dates=len(train_dates),
            time_decay_enabled=config.time_decay_enabled,
            time_decay_lambda=config.time_decay_lambda if config.time_decay_enabled else None,
            sample_weight_range=weight_range,
            config_hash=config_hash,
        )
        
        return result
    
    def _generate_labels(self, config: TrainingConfig, universe: Set[int]) -> pd.DataFrame:
        """Generate labels for the training window."""
        return self.label_generator.generate_labels(
            start_date=config.window_start,
            end_date=config.window_end,
            horizons=config.horizons,
            benchmark_symbol=config.benchmark_symbol,
            universe=list(universe)
        )
    
    def _sample_training_dates(self, config: TrainingConfig) -> List[date]:
        """Sample training dates according to the sampling strategy."""
        from data.universe import TradingCalendar
        
        calendar = TradingCalendar()
        all_trading_days = calendar.get_trading_days(config.window_start, config.window_end)
        all_dates = [d.date() for d in all_trading_days]
        
        # Apply sampling
        sampled = all_dates[::config.sampling.sample_every_n_days]
        
        # Apply custom filter if provided
        if config.sampling.date_filter:
            sampled = [d for d in sampled if config.sampling.date_filter(d)]
        
        return sampled
    
    def _build_training_data(
        self,
        train_dates: List[date],
        labels_df: pd.DataFrame,
        config: TrainingConfig,
        universe: Set[int],
    ) -> Tuple[List[pd.DataFrame], List[np.ndarray], List[date]]:
        """Build feature/label pairs for each training date."""
        X_list = []
        y_list = []
        date_list = []
        
        for train_date in train_dates:
            try:
                # Build features
                features_df = self.feature_pipeline.build_features_cross_sectional(
                    as_of_date=train_date,
                    universe=universe,
                    lookback_days=config.feature_lookback_days
                )
                
                if len(features_df) == 0:
                    continue
                
                # Get labels for this date
                date_labels = labels_df[labels_df['date'] == train_date]
                if len(date_labels) == 0:
                    continue
                
                # Merge features with labels
                merged = features_df.merge(
                    date_labels[['asset_id', config.label_column]],
                    on='asset_id',
                    how='inner'
                )
                
                if len(merged) == 0:
                    continue
                
                # Extract features (exclude metadata columns)
                exclude_cols = {'asset_id', 'date', config.label_column}
                feature_cols = [c for c in merged.columns if c not in exclude_cols]
                
                X = merged[feature_cols].copy()
                
                # Sanitize: convert to numeric and fill NaN
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(0)
                
                y = merged[config.label_column].values
                
                X_list.append(X)
                y_list.append(y)
                # Track date for each sample (for time-decay weighting)
                date_list.extend([train_date] * len(merged))
                
            except Exception as e:
                logger.debug(f"Skipping {train_date}: {e}")
                continue
        
        return X_list, y_list, date_list
    
    def _compute_time_decay_weights(
        self,
        dates: List[date],
        reference_date: date,
        lambda_: float,
        min_weight: float,
    ) -> np.ndarray:
        """Compute exponential time-decay weights."""
        ref_ordinal = reference_date.toordinal()
        weights = []
        
        for d in dates:
            if hasattr(d, 'date'):
                d = d.date()
            age_days = ref_ordinal - d.toordinal()
            w = np.exp(-lambda_ * age_days)
            w = max(w, min_weight)
            weights.append(w)
        
        return np.array(weights)
    
    def _model_supports_sample_weight(self, model) -> bool:
        """Check if model's fit method accepts sample_weight."""
        import inspect
        fit_method = getattr(model, 'fit', None)
        if fit_method is None:
            return False
        sig = inspect.signature(fit_method)
        return 'sample_weight' in sig.parameters
    
    def _compute_config_hash(self, config: TrainingConfig, model_params: Dict) -> str:
        """Compute hash of training configuration for reproducibility."""
        config_dict = {
            'window_start': str(config.window_start),
            'window_end': str(config.window_end),
            'horizons': config.horizons,
            'time_decay_enabled': config.time_decay_enabled,
            'time_decay_lambda': config.time_decay_lambda,
            'model_params': model_params,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_feature_schema(
    current_features: List[str],
    saved_schema: FeatureSchema,
    strict: bool = True,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate current features against a saved schema.
    
    Args:
        current_features: List of current feature names
        saved_schema: The expected FeatureSchema
        strict: If True, require exact match; if False, allow superset
        raise_on_error: If True, raise FeatureSchemaMismatchError on failure
        
    Returns:
        Tuple of (is_valid, list of issues)
        
    Raises:
        FeatureSchemaMismatchError: If validation fails and raise_on_error is True
    """
    is_valid, issues = saved_schema.validate(current_features, strict=strict)
    
    if not is_valid and raise_on_error:
        raise FeatureSchemaMismatchError(issues)
    
    return is_valid, issues


def load_training_result_metadata(model_data: Dict) -> Optional[TrainingResult]:
    """
    Extract TrainingResult metadata from a saved model dict.
    
    Args:
        model_data: Dictionary loaded from model pickle
        
    Returns:
        TrainingResult with metadata (model field will be None) or None if not available
    """
    try:
        # Check for new-style metadata
        if 'training_result' in model_data:
            result_dict = model_data['training_result']
            from datetime import datetime
            return TrainingResult(
                model=None,  # Don't include model in metadata-only load
                feature_names=result_dict['feature_names'],
                feature_count=result_dict['feature_count'],
                feature_hash=result_dict['feature_hash'],
                trained_date=datetime.strptime(result_dict['trained_date'], '%Y-%m-%d').date(),
                window_start=datetime.strptime(result_dict['window_start'], '%Y-%m-%d').date(),
                window_end=datetime.strptime(result_dict['window_end'], '%Y-%m-%d').date(),
                horizons=result_dict['horizons'],
                num_samples=result_dict['num_samples'],
                num_training_dates=result_dict['num_training_dates'],
                time_decay_enabled=result_dict['time_decay_enabled'],
                time_decay_lambda=result_dict.get('time_decay_lambda'),
                sample_weight_range=result_dict.get('sample_weight_range'),
                config_hash=result_dict['config_hash'],
                training_metrics=result_dict.get('training_metrics', {}),
            )
        
        # Fall back to legacy format
        if 'feature_cols' in model_data:
            feature_names = model_data['feature_cols']
            return TrainingResult(
                model=None,
                feature_names=feature_names,
                feature_count=len(feature_names),
                feature_hash=FeatureSchema._compute_hash(feature_names),
                trained_date=model_data.get('trained_date', date.today()),
                window_start=model_data.get('effective_start', date.today()),
                window_end=model_data.get('trained_date', date.today()),
                horizons=[model_data.get('horizon', 20)],
                num_samples=model_data.get('training_samples', 0),
                num_training_dates=0,
                time_decay_enabled=model_data.get('retraining_config', {}).get('time_decay_enabled', False),
                time_decay_lambda=model_data.get('retraining_config', {}).get('time_decay_lambda'),
                sample_weight_range=None,
                config_hash=model_data.get('config_hash', ''),
            )
        
        return None
    except Exception as e:
        logger.warning(f"Could not load training result metadata: {e}")
        return None

