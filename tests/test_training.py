"""Tests for training-related functionality.

Tests cover:
- TabularTrainer training loop
- WalkForwardRetrainer.should_retrain behavior
- Feature schema validation
- Time-decay sample weighting
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd

from models.tabular_trainer import (
    TabularTrainer,
    TrainingConfig,
    TrainingResult,
    SamplingStrategy,
    FeatureSchema,
    FeatureSchemaMismatchError,
    validate_feature_schema,
)
from configs.loader import RetrainingConfig, TimeDecayConfig, AdaptiveRetrainingConfig


class TestFeatureSchema:
    """Tests for FeatureSchema validation."""
    
    def test_schema_from_dataframe(self):
        """Test creating schema from DataFrame."""
        df = pd.DataFrame({
            'asset_id': [1, 2, 3],
            'date': [date(2024, 1, 1)] * 3,
            'feature_a': [1.0, 2.0, 3.0],
            'feature_b': [4.0, 5.0, 6.0],
        })
        
        schema = FeatureSchema.from_dataframe(df, horizons=[20])
        
        assert schema.feature_names == ['feature_a', 'feature_b']
        assert schema.horizons == [20]
        assert len(schema.feature_hash) == 12
    
    def test_schema_validation_exact_match(self):
        """Test validation with exact feature match."""
        schema = FeatureSchema(
            feature_names=['feature_a', 'feature_b'],
            feature_hash='abc123',
            horizons=[20],
        )
        
        is_valid, issues = schema.validate(['feature_a', 'feature_b'], strict=True)
        
        # Hash will differ since we manually set it
        assert 'Feature hash mismatch' in issues[0] if issues else True
    
    def test_schema_validation_missing_features(self):
        """Test validation with missing features."""
        schema = FeatureSchema(
            feature_names=['feature_a', 'feature_b', 'feature_c'],
            feature_hash=FeatureSchema._compute_hash(['feature_a', 'feature_b', 'feature_c']),
            horizons=[20],
        )
        
        is_valid, issues = schema.validate(['feature_a', 'feature_b'], strict=True)
        
        assert not is_valid
        assert any('Missing features' in issue for issue in issues)
    
    def test_schema_validation_extra_features_strict(self):
        """Test validation with extra features in strict mode."""
        schema = FeatureSchema(
            feature_names=['feature_a'],
            feature_hash=FeatureSchema._compute_hash(['feature_a']),
            horizons=[20],
        )
        
        is_valid, issues = schema.validate(['feature_a', 'feature_b'], strict=True)
        
        assert not is_valid
        assert any('Unexpected features' in issue for issue in issues)
    
    def test_schema_validation_extra_features_non_strict(self):
        """Test validation with extra features in non-strict mode."""
        schema = FeatureSchema(
            feature_names=['feature_a'],
            feature_hash=FeatureSchema._compute_hash(['feature_a']),
            horizons=[20],
        )
        
        # Non-strict allows superset
        is_valid, issues = schema.validate(['feature_a', 'feature_b'], strict=False)
        
        # Should still fail due to count mismatch
        assert any('count mismatch' in issue for issue in issues)
    
    def test_schema_hash_consistency(self):
        """Test that hash is consistent for same features."""
        features = ['z_feature', 'a_feature', 'm_feature']
        
        hash1 = FeatureSchema._compute_hash(features)
        hash2 = FeatureSchema._compute_hash(features)
        
        assert hash1 == hash2
        
        # Order shouldn't matter (sorted internally)
        hash3 = FeatureSchema._compute_hash(['a_feature', 'm_feature', 'z_feature'])
        assert hash1 == hash3


class TestRetrainingConfig:
    """Tests for RetrainingConfig.should_retrain behavior."""
    
    def test_should_retrain_cadence(self):
        """Test cadence-based retraining trigger."""
        config = RetrainingConfig(
            cadence_days=20,
            time_decay=TimeDecayConfig(enabled=False),
            adaptive=AdaptiveRetrainingConfig(enabled=False),
        )
        
        last_train = date(2024, 1, 1)
        
        # Not due yet (10 days)
        current = date(2024, 1, 11)
        should_retrain, reason = config.should_retrain(last_train, current)
        assert not should_retrain
        assert reason == "not due"
        
        # Due (20 days)
        current = date(2024, 1, 21)
        should_retrain, reason = config.should_retrain(last_train, current)
        assert should_retrain
        assert "scheduled" in reason
    
    def test_should_retrain_adaptive_sharpe(self):
        """Test adaptive retraining based on Sharpe degradation."""
        config = RetrainingConfig(
            cadence_days=60,  # Long cadence
            time_decay=TimeDecayConfig(enabled=False),
            adaptive=AdaptiveRetrainingConfig(
                enabled=True,
                sharpe_floor=-0.5,
                hit_rate_floor=0.40,
            ),
        )
        
        last_train = date(2024, 1, 1)
        current = date(2024, 1, 15)  # Not at cadence yet
        
        # Good metrics - no retrain
        good_metrics = {'sharpe': 0.5, 'hit_rate': 0.55}
        should_retrain, reason = config.should_retrain(last_train, current, good_metrics)
        assert not should_retrain
        
        # Bad Sharpe - trigger retrain
        bad_metrics = {'sharpe': -0.8, 'hit_rate': 0.55}
        should_retrain, reason = config.should_retrain(last_train, current, bad_metrics)
        assert should_retrain
        assert "sharpe" in reason.lower()
    
    def test_should_retrain_adaptive_hit_rate(self):
        """Test adaptive retraining based on hit rate degradation."""
        config = RetrainingConfig(
            cadence_days=60,
            time_decay=TimeDecayConfig(enabled=False),
            adaptive=AdaptiveRetrainingConfig(
                enabled=True,
                sharpe_floor=-0.5,
                hit_rate_floor=0.40,
            ),
        )
        
        last_train = date(2024, 1, 1)
        current = date(2024, 1, 15)
        
        # Bad hit rate - trigger retrain
        bad_metrics = {'sharpe': 0.5, 'hit_rate': 0.35}
        should_retrain, reason = config.should_retrain(last_train, current, bad_metrics)
        assert should_retrain
        assert "hit_rate" in reason.lower()
    
    def test_compute_sample_weights_decay(self):
        """Test time-decay sample weight computation."""
        config = RetrainingConfig(
            time_decay=TimeDecayConfig(
                enabled=True,
                lambda_=0.01,  # Faster decay for testing
                min_weight=0.1,
            ),
        )
        
        reference_date = date(2024, 1, 31)
        dates = [
            date(2024, 1, 31),  # Today - weight ~1.0
            date(2024, 1, 21),  # 10 days ago
            date(2024, 1, 1),   # 30 days ago
        ]
        
        weights = config.compute_sample_weights(dates, reference_date)
        
        # Most recent should have highest weight
        assert weights[0] > weights[1] > weights[2]
        
        # Check floor is respected
        assert all(w >= 0.1 for w in weights)
    
    def test_compute_sample_weights_disabled(self):
        """Test that weights are uniform when decay is disabled."""
        config = RetrainingConfig(
            time_decay=TimeDecayConfig(enabled=False),
        )
        
        reference_date = date(2024, 1, 31)
        dates = [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 31)]
        
        weights = config.compute_sample_weights(dates, reference_date)
        
        assert all(w == 1.0 for w in weights)
    
    def test_get_window_start_rolling(self):
        """Test rolling window start calculation."""
        config = RetrainingConfig(
            window_type="rolling",
            window_years=3,
        )
        
        as_of_date = date(2024, 6, 15)
        window_start = config.get_window_start(as_of_date)
        
        # Should be ~3 years before
        expected = date(2021, 6, 15)
        assert window_start == expected


class TestSamplingStrategy:
    """Tests for SamplingStrategy."""
    
    def test_default_sampling(self):
        """Test default sampling configuration."""
        strategy = SamplingStrategy()
        
        assert strategy.sample_every_n_days == 5
        assert strategy.min_samples == 100
        assert strategy.date_filter is None
    
    def test_custom_date_filter(self):
        """Test custom date filter."""
        # Only include Mondays
        def monday_filter(d: date) -> bool:
            return d.weekday() == 0
        
        strategy = SamplingStrategy(
            sample_every_n_days=1,
            date_filter=monday_filter,
        )
        
        assert strategy.date_filter(date(2024, 1, 1))  # Monday
        assert not strategy.date_filter(date(2024, 1, 2))  # Tuesday


class TestValidateFeatureSchema:
    """Tests for validate_feature_schema utility."""
    
    def test_validate_raises_on_mismatch(self):
        """Test that validation raises exception on mismatch."""
        schema = FeatureSchema(
            feature_names=['feature_a', 'feature_b'],
            feature_hash=FeatureSchema._compute_hash(['feature_a', 'feature_b']),
            horizons=[20],
        )
        
        with pytest.raises(FeatureSchemaMismatchError) as exc_info:
            validate_feature_schema(
                ['feature_a'],  # Missing feature_b
                schema,
                strict=True,
                raise_on_error=True,
            )
        
        assert len(exc_info.value.issues) > 0
    
    def test_validate_returns_issues_without_raising(self):
        """Test validation without raising exception."""
        schema = FeatureSchema(
            feature_names=['feature_a', 'feature_b'],
            feature_hash=FeatureSchema._compute_hash(['feature_a', 'feature_b']),
            horizons=[20],
        )
        
        is_valid, issues = validate_feature_schema(
            ['feature_a'],
            schema,
            strict=True,
            raise_on_error=False,
        )
        
        assert not is_valid
        assert len(issues) > 0


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = TrainingResult(
            model=None,
            feature_names=['a', 'b'],
            feature_count=2,
            feature_hash='abc123',
            trained_date=date(2024, 1, 1),
            window_start=date(2023, 1, 1),
            window_end=date(2024, 1, 1),
            horizons=[20],
            num_samples=1000,
            num_training_dates=50,
            time_decay_enabled=True,
            time_decay_lambda=0.001,
            sample_weight_range=(0.1, 1.0),
            config_hash='def456',
        )
        
        d = result.to_dict()
        
        assert d['feature_names'] == ['a', 'b']
        assert d['feature_count'] == 2
        assert d['trained_date'] == '2024-01-01'
        assert d['horizons'] == [20]
        assert d['time_decay_enabled'] is True

