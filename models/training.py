"""Training loops for sequence models and walk-forward retraining utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import date, datetime
import numpy as np
import pandas as pd
import pickle
import json
from models.torch.losses import MultiHorizonLoss
from models.splits import TimeSplit
from models.tabular_trainer import (
    TabularTrainer, TrainingConfig, TrainingResult, SamplingStrategy,
    FeatureSchema, validate_feature_schema, FeatureSchemaMismatchError
)


# =============================================================================
# Walk-Forward Retraining Manager
# =============================================================================

class WalkForwardRetrainer:
    """
    Manages walk-forward model retraining with time-decay sample weighting.
    
    This class implements the retraining policy defined in configs/base.yaml,
    ensuring consistent behavior across backtests, simulations, and live trading.
    
    Now delegates training to TabularTrainer for consistency with other training paths.
    """
    
    def __init__(
        self,
        retraining_config,
        model_class,
        model_params: Dict[str, Any],
        feature_pipeline,
        label_generator,
        storage,
        api,
        universe: set,
        horizon: int = 20,
    ):
        """
        Initialize the walk-forward retrainer.
        
        Args:
            retraining_config: RetrainingConfig instance from configs
            model_class: Model class to instantiate (e.g., XGBoostModel)
            model_params: Parameters to pass to model constructor
            feature_pipeline: FeaturePipeline instance
            label_generator: ReturnLabelGenerator instance
            storage: StorageBackend instance
            api: AsOfQueryAPI instance
            universe: Set of asset_ids to train on
            horizon: Prediction horizon in days
        """
        self.config = retraining_config
        self.model_class = model_class
        self.model_params = model_params
        self.feature_pipeline = feature_pipeline
        self.label_generator = label_generator
        self.storage = storage
        self.api = api
        self.universe = universe
        self.horizon = horizon
        
        # Initialize TabularTrainer for delegating training
        self.tabular_trainer = TabularTrainer(
            feature_pipeline=feature_pipeline,
            label_generator=label_generator,
            storage=storage,
            api=api,
        )
        
        # State
        self.current_model = None
        self.current_training_result: Optional[TrainingResult] = None
        self.last_train_date = None
        self.model_version = 0
        self.rolling_metrics = {}
        self.model_history = []  # List of (date, model_path, metrics, training_result)
    
    def get_model_for_date(self, as_of_date: date) -> Tuple[Any, bool]:
        """
        Get the appropriate model for a given date, retraining if necessary.
        
        Args:
            as_of_date: The trading date we need a model for
            
        Returns:
            Tuple of (model, was_retrained)
        """
        # Check if we need to retrain
        if self.current_model is None:
            # First time - must train
            self._train_model(as_of_date)
            return self.current_model, True
        
        should_retrain, reason = self.config.should_retrain(
            self.last_train_date,
            as_of_date,
            self.rolling_metrics
        )
        
        if should_retrain:
            from loguru import logger
            logger.info(f"Retraining model: {reason}")
            self._train_model(as_of_date)
            return self.current_model, True
        
        return self.current_model, False
    
    def get_training_result(self) -> Optional[TrainingResult]:
        """Get the TrainingResult from the most recent training."""
        return self.current_training_result
    
    def _train_model(self, as_of_date: date):
        """
        Train a new model using data up to as_of_date.
        
        Delegates to TabularTrainer for the actual training logic.
        """
        from loguru import logger
        
        # Calculate training window
        window_start = self.config.get_window_start(as_of_date)
        train_end = as_of_date
        
        logger.info(f"Training model on {window_start} to {train_end}")
        
        # Build TrainingConfig from retraining config
        training_config = TrainingConfig(
            window_start=window_start,
            window_end=train_end,
            horizons=[self.horizon],
            sampling=SamplingStrategy(sample_every_n_days=5),
            time_decay_enabled=self.config.time_decay.enabled,
            time_decay_lambda=self.config.time_decay.lambda_,
            time_decay_min_weight=self.config.time_decay.min_weight,
            feature_lookback_days=252,
            benchmark_symbol="SPY",
        )
        
        # Delegate to TabularTrainer
        training_result = self.tabular_trainer.train(
            model_class=self.model_class,
            model_params=self.model_params,
            config=training_config,
            universe=self.universe,
        )
        
        logger.info(f"Training on {training_result.num_samples} samples with {training_result.feature_count} features")
        if training_result.sample_weight_range:
            logger.info(f"Sample weight range: [{training_result.sample_weight_range[0]:.3f}, {training_result.sample_weight_range[1]:.3f}]")
        
        # Update state
        self.current_model = training_result.model
        self.current_training_result = training_result
        self.last_train_date = as_of_date
        self.model_version += 1
        
        # Save model if versioning enabled
        if self.config.versioning.enabled:
            self._save_model_version(as_of_date, training_result)
    
    def _save_model_version(self, as_of_date: date, training_result: TrainingResult):
        """Save a versioned model artifact with full training metadata."""
        from loguru import logger
        
        artifact_dir = Path(self.config.versioning.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versioned filename
        version_str = as_of_date.strftime("%Y-%m-%d")
        model_path = artifact_dir / f"model_{version_str}.pkl"
        
        # Build feature schema for validation
        feature_schema = FeatureSchema(
            feature_names=training_result.feature_names,
            feature_hash=training_result.feature_hash,
            horizons=training_result.horizons,
            created_date=as_of_date,
        )
        
        model_data = {
            'model': self.current_model,
            'trained_date': as_of_date,
            'effective_start': as_of_date,
            'effective_end': None,  # Will be set when next version is trained
            'model_version': self.model_version,
            # Legacy fields for backwards compatibility
            'feature_cols': training_result.feature_names,
            'feature_count': training_result.feature_count,
            'training_samples': training_result.num_samples,
            # New structured metadata
            'training_result': training_result.to_dict(),
            'feature_schema': feature_schema.to_dict(),
            'config': {
                'cadence_days': self.config.cadence_days,
                'window_type': self.config.window_type,
                'window_years': self.config.window_years,
                'time_decay_enabled': self.config.time_decay.enabled,
                'time_decay_lambda': self.config.time_decay.lambda_,
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model version to {model_path}")
        
        # Update previous version's effective_end
        if len(self.model_history) > 0:
            prev_path = self.model_history[-1][1]
            try:
                with open(prev_path, 'rb') as f:
                    prev_data = pickle.load(f)
                prev_data['effective_end'] = as_of_date
                with open(prev_path, 'wb') as f:
                    pickle.dump(prev_data, f)
            except Exception as e:
                logger.warning(f"Could not update previous model version: {e}")
        
        # Track in history with training result
        self.model_history.append((as_of_date, model_path, {}, training_result))
        
        # Clean up old versions
        self._cleanup_old_versions()
    
    def _cleanup_old_versions(self):
        """Remove old model versions beyond keep_versions limit."""
        keep = self.config.versioning.keep_versions
        if len(self.model_history) > keep:
            from loguru import logger
            to_remove = self.model_history[:-keep]
            for _, path, _ in to_remove:
                try:
                    Path(path).unlink()
                    logger.debug(f"Removed old model version: {path}")
                except Exception:
                    pass
            self.model_history = self.model_history[-keep:]
    
    def update_rolling_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        date: date
    ):
        """
        Update rolling performance metrics for adaptive retraining.
        
        Args:
            predictions: Model predictions
            actuals: Actual returns
            date: Date of predictions
        """
        # Store recent predictions and actuals
        if 'history' not in self.rolling_metrics:
            self.rolling_metrics['history'] = []
        
        self.rolling_metrics['history'].append({
            'date': date,
            'predictions': predictions,
            'actuals': actuals
        })
        
        # Keep only lookback_days worth of history
        lookback = self.config.adaptive.lookback_days
        if len(self.rolling_metrics['history']) > lookback:
            self.rolling_metrics['history'] = self.rolling_metrics['history'][-lookback:]
        
        # Compute rolling metrics
        if len(self.rolling_metrics['history']) >= 10:  # Minimum for meaningful stats
            all_preds = np.concatenate([h['predictions'] for h in self.rolling_metrics['history']])
            all_actuals = np.concatenate([h['actuals'] for h in self.rolling_metrics['history']])
            
            # Hit rate: fraction where sign(pred) == sign(actual)
            hit_rate = np.mean(np.sign(all_preds) == np.sign(all_actuals))
            self.rolling_metrics['hit_rate'] = hit_rate
            
            # Simple Sharpe proxy: mean(actual) / std(actual) for top-ranked predictions
            # This is a rough approximation
            top_mask = all_preds > np.median(all_preds)
            if top_mask.sum() > 0:
                top_returns = all_actuals[top_mask]
                if len(top_returns) > 1 and np.std(top_returns) > 0:
                    sharpe_proxy = np.mean(top_returns) / np.std(top_returns) * np.sqrt(252)
                    self.rolling_metrics['sharpe'] = sharpe_proxy
            
            # Calibration: ratio of realized variance to predicted variance
            pred_var = np.var(all_preds)
            actual_var = np.var(all_actuals)
            if pred_var > 0:
                self.rolling_metrics['calibration_ratio'] = actual_var / pred_var
    
    def load_model_for_date(self, target_date: date) -> Optional[Any]:
        """
        Load the appropriate model version for a historical date.
        
        Used in backtests to ensure we use the model that would have been
        available at that point in time.
        
        Args:
            target_date: The date we need a model for
            
        Returns:
            Model if found, None otherwise
        """
        artifact_dir = Path(self.config.versioning.artifact_dir)
        
        # Find all model versions
        model_files = sorted(artifact_dir.glob("model_*.pkl"))
        
        best_model = None
        best_date = None
        
        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                trained_date = model_data.get('trained_date')
                effective_end = model_data.get('effective_end')
                
                # Model is valid if trained before target_date and
                # (no effective_end or effective_end > target_date)
                if trained_date and trained_date <= target_date:
                    if effective_end is None or effective_end > target_date:
                        if best_date is None or trained_date > best_date:
                            best_model = model_data['model']
                            best_date = trained_date
                            
            except Exception:
                continue
        
        if best_model:
            self.current_model = best_model
            self.last_train_date = best_date
        
        return best_model


class SequenceTrainer:
    """Trainer for sequence models."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultiHorizonLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "artifacts/models"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)
            labels = {h: l.to(self.device) for h, l in batch['labels'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(self.device)
                labels = {h: l.to(self.device) for h, l in batch['labels'].items()}
                
                predictions = self.model(sequences)
                loss = self.loss_fn(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """Train model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best:
                        self.save_checkpoint(f"best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.save_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

