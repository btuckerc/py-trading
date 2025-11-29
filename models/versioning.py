"""Model versioning with effective dates.

This module provides utilities for managing versioned model artifacts
with explicit effective date ranges, enabling:
- Point-in-time model selection for backtests
- Rollback to previous versions
- Model lineage tracking
"""

import pickle
import json
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class ModelVersion:
    """Represents a single model version with metadata."""
    
    def __init__(
        self,
        version: int,
        model: Any,
        trained_date: date,
        effective_start: date,
        effective_end: Optional[date] = None,
        feature_cols: Optional[List[str]] = None,
        training_samples: int = 0,
        config: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        self.version = version
        self.model = model
        self.trained_date = trained_date
        self.effective_start = effective_start
        self.effective_end = effective_end
        self.feature_cols = feature_cols or []
        self.training_samples = training_samples
        self.config = config or {}
        self.metrics = metrics or {}
        self.created_at = datetime.now()
    
    def is_effective_on(self, query_date: date) -> bool:
        """Check if this version is effective for a given date."""
        if query_date < self.effective_start:
            return False
        if self.effective_end is not None and query_date >= self.effective_end:
            return False
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without model object)."""
        return {
            'version': self.version,
            'trained_date': str(self.trained_date),
            'effective_start': str(self.effective_start),
            'effective_end': str(self.effective_end) if self.effective_end else None,
            'feature_count': len(self.feature_cols),
            'training_samples': self.training_samples,
            'config': self.config,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat()
        }


class ModelVersionManager:
    """
    Manages versioned model artifacts with effective date tracking.
    
    Features:
    - Save/load model versions with metadata
    - Query model by effective date (point-in-time)
    - Automatic version numbering
    - Cleanup of old versions
    - Model comparison and rollback
    """
    
    def __init__(
        self,
        artifact_dir: str = "artifacts/models",
        keep_versions: int = 10,
        model_prefix: str = "model"
    ):
        """
        Initialize the version manager.
        
        Args:
            artifact_dir: Directory to store model artifacts
            keep_versions: Number of old versions to keep
            model_prefix: Prefix for model filenames
        """
        self.artifact_dir = Path(artifact_dir)
        self.keep_versions = keep_versions
        self.model_prefix = model_prefix
        
        # In-memory index of versions
        self.versions: Dict[int, ModelVersion] = {}
        self.current_version: Optional[int] = None
        
        # Load existing index if present
        self._load_index()
    
    def _get_index_path(self) -> Path:
        """Get path to the version index file."""
        return self.artifact_dir / f"{self.model_prefix}_index.json"
    
    def _get_model_path(self, version: int) -> Path:
        """Get path to a specific model version file."""
        return self.artifact_dir / f"{self.model_prefix}_v{version:04d}.pkl"
    
    def _load_index(self):
        """Load version index from disk."""
        index_path = self._get_index_path()
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            self.current_version = index_data.get('current_version')
            
            for v_data in index_data.get('versions', []):
                version = v_data['version']
                # Load model if file exists
                model_path = self._get_model_path(version)
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        self.versions[version] = ModelVersion(
                            version=version,
                            model=model_data.get('model'),
                            trained_date=self._parse_date(v_data['trained_date']),
                            effective_start=self._parse_date(v_data['effective_start']),
                            effective_end=self._parse_date(v_data['effective_end']) if v_data['effective_end'] else None,
                            feature_cols=model_data.get('feature_cols', []),
                            training_samples=v_data.get('training_samples', 0),
                            config=v_data.get('config', {}),
                            metrics=v_data.get('metrics', {})
                        )
                    except Exception as e:
                        logger.warning(f"Could not load model v{version}: {e}")
        except Exception as e:
            logger.warning(f"Could not load version index: {e}")
    
    def _save_index(self):
        """Save version index to disk."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'current_version': self.current_version,
            'versions': [v.to_dict() for v in sorted(self.versions.values(), key=lambda x: x.version)],
            'updated_at': datetime.now().isoformat()
        }
        
        index_path = self._get_index_path()
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object."""
        if isinstance(date_str, date):
            return date_str
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    
    def save_version(
        self,
        model: Any,
        trained_date: date,
        feature_cols: List[str],
        training_samples: int,
        config: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ) -> int:
        """
        Save a new model version.
        
        Args:
            model: Trained model object
            trained_date: Date model was trained
            feature_cols: List of feature column names
            training_samples: Number of training samples used
            config: Training/model configuration
            metrics: Performance metrics
            
        Returns:
            Version number assigned
        """
        # Determine new version number
        if self.versions:
            new_version = max(self.versions.keys()) + 1
        else:
            new_version = 1
        
        # Update effective_end of previous version
        if self.current_version is not None and self.current_version in self.versions:
            self.versions[self.current_version].effective_end = trained_date
        
        # Create new version
        model_version = ModelVersion(
            version=new_version,
            model=model,
            trained_date=trained_date,
            effective_start=trained_date,
            effective_end=None,  # Current version has no end
            feature_cols=feature_cols,
            training_samples=training_samples,
            config=config or {},
            metrics=metrics or {}
        )
        
        # Save model to disk
        model_path = self._get_model_path(new_version)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'version': new_version,
            'trained_date': trained_date,
            'feature_cols': feature_cols,
            'training_samples': training_samples,
            'config': config,
            'metrics': metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Update in-memory state
        self.versions[new_version] = model_version
        self.current_version = new_version
        
        # Save index
        self._save_index()
        
        # Cleanup old versions
        self._cleanup_old_versions()
        
        logger.info(f"Saved model version {new_version} (trained {trained_date})")
        
        return new_version
    
    def get_model_for_date(self, query_date: date) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the model that was effective on a given date.
        
        This is the key method for point-in-time correct backtesting.
        
        Args:
            query_date: The date to query
            
        Returns:
            Tuple of (model, version) or (None, None) if no model found
        """
        # Find the version that was effective on query_date
        effective_version = None
        
        for version, mv in self.versions.items():
            if mv.is_effective_on(query_date):
                if effective_version is None or version > effective_version:
                    effective_version = version
        
        if effective_version is None:
            return None, None
        
        return self.versions[effective_version].model, effective_version
    
    def get_current_model(self) -> Tuple[Optional[Any], Optional[int]]:
        """
        Get the current (latest) model.
        
        Returns:
            Tuple of (model, version) or (None, None)
        """
        if self.current_version is None or self.current_version not in self.versions:
            return None, None
        
        return self.versions[self.current_version].model, self.current_version
    
    def get_version_info(self, version: int) -> Optional[Dict]:
        """Get metadata for a specific version."""
        if version not in self.versions:
            return None
        return self.versions[version].to_dict()
    
    def list_versions(self) -> List[Dict]:
        """List all versions with metadata."""
        return [v.to_dict() for v in sorted(self.versions.values(), key=lambda x: x.version)]
    
    def rollback(self, to_version: int) -> bool:
        """
        Rollback to a previous version.
        
        This sets the specified version as current and removes
        the effective_end date.
        
        Args:
            to_version: Version to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        if to_version not in self.versions:
            logger.error(f"Version {to_version} not found")
            return False
        
        # Mark current version as ended
        if self.current_version is not None and self.current_version in self.versions:
            self.versions[self.current_version].effective_end = date.today()
        
        # Set new current version
        self.versions[to_version].effective_end = None
        self.current_version = to_version
        
        # Save index
        self._save_index()
        
        logger.info(f"Rolled back to version {to_version}")
        return True
    
    def _cleanup_old_versions(self):
        """Remove old versions beyond keep_versions limit."""
        if len(self.versions) <= self.keep_versions:
            return
        
        # Sort by version number
        sorted_versions = sorted(self.versions.keys())
        
        # Keep the most recent keep_versions
        to_remove = sorted_versions[:-self.keep_versions]
        
        for version in to_remove:
            # Don't remove current version
            if version == self.current_version:
                continue
            
            # Remove file
            model_path = self._get_model_path(version)
            try:
                model_path.unlink()
                logger.debug(f"Removed old model version {version}")
            except Exception as e:
                logger.warning(f"Could not remove {model_path}: {e}")
            
            # Remove from index
            del self.versions[version]
        
        # Save updated index
        self._save_index()
    
    def compare_versions(self, version_a: int, version_b: int) -> Dict:
        """
        Compare two model versions.
        
        Args:
            version_a: First version
            version_b: Second version
            
        Returns:
            Comparison dict
        """
        if version_a not in self.versions or version_b not in self.versions:
            return {'error': 'One or both versions not found'}
        
        va = self.versions[version_a]
        vb = self.versions[version_b]
        
        comparison = {
            'version_a': va.to_dict(),
            'version_b': vb.to_dict(),
            'feature_count_diff': len(vb.feature_cols) - len(va.feature_cols),
            'training_samples_diff': vb.training_samples - va.training_samples,
        }
        
        # Compare metrics if available
        if va.metrics and vb.metrics:
            common_metrics = set(va.metrics.keys()) & set(vb.metrics.keys())
            comparison['metric_diffs'] = {}
            for metric in common_metrics:
                val_a = va.metrics[metric]
                val_b = vb.metrics[metric]
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    comparison['metric_diffs'][metric] = val_b - val_a
        
        return comparison


def get_live_model_path(artifact_dir: str = "artifacts/models") -> Path:
    """
    Get the path to the current live model.
    
    This is a convenience function for scripts that need to load
    the current production model.
    """
    return Path(artifact_dir) / "live_model.pkl"


def save_live_model(
    model: Any,
    trained_date: date,
    feature_cols: List[str],
    training_samples: int,
    config: Optional[Dict] = None,
    artifact_dir: str = "artifacts/models"
):
    """
    Save a model as the current live model.
    
    This also saves a versioned copy for history.
    
    Args:
        model: Trained model
        trained_date: Training date
        feature_cols: Feature columns
        training_samples: Number of samples
        config: Training config
        artifact_dir: Artifact directory
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as live model
    live_path = artifact_dir / "live_model.pkl"
    model_data = {
        'model': model,
        'trained_date': trained_date,
        'effective_start': trained_date,
        'feature_cols': feature_cols,
        'feature_count': len(feature_cols),
        'training_samples': training_samples,
        'config': config
    }
    
    with open(live_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Saved live model to {live_path}")
    
    # Also save versioned copy
    version_manager = ModelVersionManager(artifact_dir=str(artifact_dir))
    version_manager.save_version(
        model=model,
        trained_date=trained_date,
        feature_cols=feature_cols,
        training_samples=training_samples,
        config=config
    )

