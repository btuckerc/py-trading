"""Performance monitoring for adaptive retraining triggers.

This module provides tools to monitor model performance in real-time
and trigger early retraining when performance degrades.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
from pathlib import Path
import json
from loguru import logger


class PerformanceMonitor:
    """
    Monitors rolling model performance metrics for adaptive retraining.
    
    Tracks:
    - Rolling Sharpe ratio (proxy)
    - Hit rate (directional accuracy)
    - Calibration ratio (predicted vs realized variance)
    - Prediction stability
    
    These metrics can trigger early retraining when they breach thresholds.
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        sharpe_floor: float = -0.5,
        hit_rate_floor: float = 0.40,
        calibration_threshold: float = 2.0,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize the performance monitor.
        
        Args:
            lookback_days: Number of days for rolling metrics
            sharpe_floor: Minimum acceptable rolling Sharpe
            hit_rate_floor: Minimum acceptable hit rate
            calibration_threshold: Maximum acceptable variance ratio
            log_dir: Optional directory to save monitoring logs
        """
        self.lookback_days = lookback_days
        self.sharpe_floor = sharpe_floor
        self.hit_rate_floor = hit_rate_floor
        self.calibration_threshold = calibration_threshold
        self.log_dir = Path(log_dir) if log_dir else None
        
        # History storage
        self.history: List[Dict] = []
        self.alerts: List[Dict] = []
        
        # Current rolling metrics
        self.rolling_metrics: Dict[str, float] = {}
    
    def record_daily_performance(
        self,
        trading_date: date,
        predictions: np.ndarray,
        actuals: np.ndarray,
        asset_ids: Optional[List[int]] = None,
        model_version: Optional[int] = None
    ):
        """
        Record a day's predictions and actual returns.
        
        Args:
            trading_date: The trading date
            predictions: Model predictions (expected returns)
            actuals: Realized returns
            asset_ids: Optional list of asset IDs for tracking
            model_version: Optional model version number
        """
        # Store daily record
        record = {
            'date': trading_date,
            'predictions': predictions.copy(),
            'actuals': actuals.copy(),
            'n_assets': len(predictions),
            'model_version': model_version,
            'timestamp': datetime.now().isoformat()
        }
        
        if asset_ids is not None:
            record['asset_ids'] = asset_ids
        
        self.history.append(record)
        
        # Keep only lookback_days of history
        if len(self.history) > self.lookback_days:
            self.history = self.history[-self.lookback_days:]
        
        # Update rolling metrics
        self._update_rolling_metrics()
        
        # Check for alerts
        self._check_alerts(trading_date)
        
        # Log to file if configured
        if self.log_dir:
            self._save_daily_log(trading_date, record)
    
    def _update_rolling_metrics(self):
        """Compute rolling performance metrics from history."""
        if len(self.history) < 5:  # Need minimum data
            return
        
        # Aggregate all predictions and actuals
        all_preds = np.concatenate([h['predictions'] for h in self.history])
        all_actuals = np.concatenate([h['actuals'] for h in self.history])
        
        # Hit rate: fraction where sign(pred) == sign(actual)
        # Filter out near-zero predictions/actuals to avoid noise
        mask = (np.abs(all_preds) > 1e-6) & (np.abs(all_actuals) > 1e-6)
        if mask.sum() > 0:
            hit_rate = np.mean(np.sign(all_preds[mask]) == np.sign(all_actuals[mask]))
            self.rolling_metrics['hit_rate'] = hit_rate
        
        # Sharpe proxy for top-ranked predictions
        # Take top 20% by predicted return and compute Sharpe of their actuals
        top_pct = 0.2
        n_top = max(1, int(len(all_preds) * top_pct))
        top_indices = np.argsort(all_preds)[-n_top:]
        top_actuals = all_actuals[top_indices]
        
        if len(top_actuals) > 1 and np.std(top_actuals) > 1e-8:
            # Annualized Sharpe (assuming daily returns)
            sharpe = np.mean(top_actuals) / np.std(top_actuals) * np.sqrt(252)
            self.rolling_metrics['sharpe'] = sharpe
        
        # Calibration: ratio of realized variance to predicted variance
        pred_var = np.var(all_preds)
        actual_var = np.var(all_actuals)
        if pred_var > 1e-10:
            self.rolling_metrics['calibration_ratio'] = actual_var / pred_var
        
        # Prediction stability: correlation of predictions across consecutive days
        if len(self.history) >= 2:
            # Compare most recent two days
            recent_preds = self.history[-1]['predictions']
            prev_preds = self.history[-2]['predictions']
            if len(recent_preds) == len(prev_preds) and len(recent_preds) > 1:
                corr = np.corrcoef(recent_preds, prev_preds)[0, 1]
                if not np.isnan(corr):
                    self.rolling_metrics['prediction_stability'] = corr
        
        # Daily return of top picks
        if len(self.history) >= 1:
            latest = self.history[-1]
            top_k = min(10, len(latest['predictions']))
            top_indices = np.argsort(latest['predictions'])[-top_k:]
            top_return = np.mean(latest['actuals'][top_indices])
            self.rolling_metrics['latest_top_return'] = top_return
    
    def _check_alerts(self, trading_date: date):
        """Check if any metrics breach thresholds and generate alerts."""
        alerts_triggered = []
        
        # Check Sharpe floor
        sharpe = self.rolling_metrics.get('sharpe')
        if sharpe is not None and sharpe < self.sharpe_floor:
            alerts_triggered.append({
                'metric': 'sharpe',
                'value': sharpe,
                'threshold': self.sharpe_floor,
                'message': f"Rolling Sharpe ({sharpe:.2f}) below floor ({self.sharpe_floor})"
            })
        
        # Check hit rate floor
        hit_rate = self.rolling_metrics.get('hit_rate')
        if hit_rate is not None and hit_rate < self.hit_rate_floor:
            alerts_triggered.append({
                'metric': 'hit_rate',
                'value': hit_rate,
                'threshold': self.hit_rate_floor,
                'message': f"Hit rate ({hit_rate:.2%}) below floor ({self.hit_rate_floor:.2%})"
            })
        
        # Check calibration threshold
        calibration = self.rolling_metrics.get('calibration_ratio')
        if calibration is not None and calibration > self.calibration_threshold:
            alerts_triggered.append({
                'metric': 'calibration_ratio',
                'value': calibration,
                'threshold': self.calibration_threshold,
                'message': f"Calibration ratio ({calibration:.2f}) exceeds threshold ({self.calibration_threshold})"
            })
        
        # Log and store alerts
        for alert in alerts_triggered:
            alert['date'] = str(trading_date)
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert['message']}")
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if performance degradation warrants early retraining.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check recent alerts
        if not self.alerts:
            return False, "no alerts"
        
        # Get alerts from last lookback period
        recent_alerts = [a for a in self.alerts if len(self.history) > 0]
        
        if not recent_alerts:
            return False, "no recent alerts"
        
        # Trigger retrain if any threshold is breached
        latest_alert = recent_alerts[-1]
        return True, f"adaptive: {latest_alert['message']}"
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current rolling metrics."""
        return self.rolling_metrics.copy()
    
    def get_summary(self) -> Dict:
        """Get a summary of monitoring state."""
        return {
            'history_length': len(self.history),
            'rolling_metrics': self.rolling_metrics.copy(),
            'num_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'lookback_days': self.lookback_days,
            'thresholds': {
                'sharpe_floor': self.sharpe_floor,
                'hit_rate_floor': self.hit_rate_floor,
                'calibration_threshold': self.calibration_threshold
            }
        }
    
    def _save_daily_log(self, trading_date: date, record: Dict):
        """Save daily monitoring log to file."""
        if not self.log_dir:
            return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save daily record (without numpy arrays)
        log_file = self.log_dir / f"monitor_{trading_date}.json"
        log_data = {
            'date': str(trading_date),
            'n_assets': record['n_assets'],
            'model_version': record['model_version'],
            'rolling_metrics': self.rolling_metrics,
            'alerts': [a for a in self.alerts if a['date'] == str(trading_date)],
            'timestamp': record['timestamp']
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def load_history(self, log_dir: Path, lookback_days: Optional[int] = None):
        """
        Load historical monitoring data from logs.
        
        Args:
            log_dir: Directory containing monitoring logs
            lookback_days: Number of days to load (default: self.lookback_days)
        """
        if lookback_days is None:
            lookback_days = self.lookback_days
        
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return
        
        # Find all monitor files
        monitor_files = sorted(log_dir.glob("monitor_*.json"))
        
        # Load most recent files
        for log_file in monitor_files[-lookback_days:]:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                # Reconstruct record (without predictions/actuals)
                record = {
                    'date': datetime.strptime(log_data['date'], "%Y-%m-%d").date(),
                    'n_assets': log_data['n_assets'],
                    'model_version': log_data.get('model_version'),
                    'timestamp': log_data['timestamp']
                }
                
                # We don't have the actual predictions/actuals, but we have the metrics
                if log_data.get('rolling_metrics'):
                    self.rolling_metrics = log_data['rolling_metrics']
                
                # Load alerts
                for alert in log_data.get('alerts', []):
                    if alert not in self.alerts:
                        self.alerts.append(alert)
                        
            except Exception as e:
                logger.debug(f"Could not load {log_file}: {e}")


class ModelPerformanceTracker:
    """
    Tracks model performance across versions for comparison.
    
    Useful for:
    - Comparing new model vs previous model
    - Detecting model degradation over time
    - A/B testing different configurations
    """
    
    def __init__(self, artifact_dir: Path = None):
        """
        Initialize the tracker.
        
        Args:
            artifact_dir: Directory where model artifacts are stored
        """
        self.artifact_dir = Path(artifact_dir) if artifact_dir else Path("artifacts/models")
        self.version_metrics: Dict[int, Dict] = {}
    
    def record_version_performance(
        self,
        model_version: int,
        metrics: Dict[str, float],
        trading_dates: List[date],
        config: Optional[Dict] = None
    ):
        """
        Record performance metrics for a model version.
        
        Args:
            model_version: Model version number
            metrics: Performance metrics dict
            trading_dates: Dates this version was active
            config: Optional model/training config
        """
        self.version_metrics[model_version] = {
            'metrics': metrics,
            'start_date': str(min(trading_dates)) if trading_dates else None,
            'end_date': str(max(trading_dates)) if trading_dates else None,
            'num_days': len(trading_dates),
            'config': config,
            'recorded_at': datetime.now().isoformat()
        }
    
    def compare_versions(self, version_a: int, version_b: int) -> Dict:
        """
        Compare performance between two model versions.
        
        Args:
            version_a: First version to compare
            version_b: Second version to compare
            
        Returns:
            Dict with comparison results
        """
        if version_a not in self.version_metrics or version_b not in self.version_metrics:
            return {'error': 'One or both versions not found'}
        
        metrics_a = self.version_metrics[version_a]['metrics']
        metrics_b = self.version_metrics[version_b]['metrics']
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'differences': {}
        }
        
        # Compute differences for common metrics
        common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
        for key in common_keys:
            val_a = metrics_a[key]
            val_b = metrics_b[key]
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                pct_change = (diff / val_a * 100) if val_a != 0 else 0
                comparison['differences'][key] = {
                    'absolute': diff,
                    'percent': pct_change
                }
        
        return comparison
    
    def get_best_version(self, metric: str = 'sharpe_ratio') -> Optional[int]:
        """
        Get the best performing model version by a given metric.
        
        Args:
            metric: Metric to compare (default: sharpe_ratio)
            
        Returns:
            Best version number, or None if no versions recorded
        """
        if not self.version_metrics:
            return None
        
        best_version = None
        best_value = float('-inf')
        
        for version, data in self.version_metrics.items():
            value = data['metrics'].get(metric)
            if value is not None and value > best_value:
                best_value = value
                best_version = version
        
        return best_version
    
    def save(self, filepath: Optional[Path] = None):
        """Save tracker state to file."""
        if filepath is None:
            filepath = self.artifact_dir / "model_performance_history.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.version_metrics, f, indent=2, default=str)
    
    def load(self, filepath: Optional[Path] = None):
        """Load tracker state from file."""
        if filepath is None:
            filepath = self.artifact_dir / "model_performance_history.json"
        
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Convert string keys back to int
            self.version_metrics = {int(k): v for k, v in data.items()}

