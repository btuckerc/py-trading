"""Run daily live/paper trading loop.

This script runs a daily trading loop that:
1. Loads a trained model or trains one on historical data
2. Builds features as-of the trading date
3. Gets current regime and computes exposure scaling
4. Generates predictions and portfolio weights
5. Applies regime-aware sector tilts and exposure
6. Submits orders to broker (paper or live)
7. Logs all activity for monitoring

Usage:
    python scripts/run_live_loop.py --trading-date 2024-01-15 --model-type xgboost
    python scripts/run_live_loop.py --dry-run  # Show what would happen without submitting orders
    python scripts/run_live_loop.py --regime-aware  # Enable regime-aware exposure and sector tilts
    python scripts/run_live_loop.py --broker alpaca_paper  # Use Alpaca paper trading
    
For scheduled cloud deployment:
    # Add to crontab for daily runs at 4:30 PM ET (after market close)
    # 30 16 * * 1-5 cd /path/to/py-finance && python scripts/run_live_loop.py --skip-if-logged
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
from datetime import date, datetime
import argparse
import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import signal
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from data.universe import TradingCalendar
from data.maintenance import ensure_data_coverage, prepare_for_trading
from labels.returns import ReturnLabelGenerator
from labels.regimes import RegimeLabelGenerator
from features.pipeline import FeaturePipeline
from features.regime_metrics import RegimeMetricsService
from models.tabular import XGBoostModel, LightGBMModel
from portfolio.strategies import LongTopKStrategy
from portfolio.scores import ScoreConverter
from portfolio.risk import RiskManager, ExposureManager, DrawdownManager, SectorTiltManager
from live.engine import LiveEngine
from live.paper_broker import PaperBroker
from live.alerting import AlertManager, AlertLevel
from configs.loader import get_config
from loguru import logger


# Global flag for graceful shutdown
shutdown_requested = False

# Global alert manager (initialized in main)
alert_manager = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_config_hash(config) -> str:
    """Generate a hash of the config for versioning."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_symbol_to_asset_id(storage: StorageBackend) -> dict:
    """Get mapping from symbol to asset_id."""
    assets_df = storage.query("SELECT asset_id, symbol FROM assets")
    if len(assets_df) == 0:
        return {}
    return dict(zip(assets_df['symbol'], assets_df['asset_id']))


def get_asset_id_to_symbol(storage: StorageBackend) -> dict:
    """Get mapping from asset_id to symbol."""
    assets_df = storage.query("SELECT asset_id, symbol FROM assets")
    if len(assets_df) == 0:
        return {}
    return dict(zip(assets_df['asset_id'], assets_df['symbol']))


def load_or_train_model(
    model_path: Path,
    model_type: str,
    storage: StorageBackend,
    api: AsOfQueryAPI,
    config,
    training_end_date: date,
    universe: set,
    horizon: int = 20,
    train_days: int = 750,  # ~3 years
    force_retrain: bool = False,
    expected_features: list = None,  # Changed from expected_feature_count to full list
    use_retraining_policy: bool = True
) -> object:
    """
    Load model from disk or train a new one.
    
    Now uses the retraining policy from config for:
    - Determining if retraining is needed (cadence)
    - Time-decay sample weighting
    - Training window calculation
    - Feature schema validation (names and order, not just count)
    
    Args:
        model_path: Path to model artifact
        model_type: 'xgboost' or 'lightgbm'
        storage: Storage backend
        api: AsOfQueryAPI
        config: Config object
        training_end_date: End date for training data
        universe: Asset universe to train on
        horizon: Prediction horizon in days
        train_days: Number of days of training data (fallback if no policy)
        force_retrain: If True, retrain even if model exists
        expected_features: If provided, validate model features against this list
        use_retraining_policy: If True, use config.retraining for cadence and weighting
    
    Returns:
        Trained model
    """
    from models.tabular_trainer import (
        TabularTrainer, TrainingConfig, SamplingStrategy, FeatureSchema,
        validate_feature_schema, FeatureSchemaMismatchError
    )
    
    # Get retraining config
    retraining_config = None
    if use_retraining_policy and hasattr(config, 'retraining'):
        retraining_config = config.retraining
    
    # Check if model exists and is recent
    if model_path.exists() and not force_retrain:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if model is recent enough
            model_date = model_data.get('trained_date')
            if model_date:
                # Use retraining policy cadence if available, else fall back to live config
                if retraining_config:
                    should_retrain, reason = retraining_config.should_retrain(
                        model_date, training_end_date
                    )
                    if not should_retrain:
                        # Validate feature schema if expected_features provided
                        if expected_features is not None:
                            schema_issues = _validate_model_features(model_data, expected_features)
                            if schema_issues:
                                logger.warning(
                                    f"Feature schema mismatch: {'; '.join(schema_issues)}. Retraining..."
                                )
                            else:
                                model_age_days = (training_end_date - model_date).days
                                logger.info(f"Loaded existing model from {model_path} (trained {model_age_days} days ago)")
                                return model_data['model']
                        else:
                            model_age_days = (training_end_date - model_date).days
                            logger.info(f"Loaded existing model from {model_path} (trained {model_age_days} days ago)")
                            return model_data['model']
                    else:
                        logger.info(f"Retraining needed: {reason}")
                else:
                    # Legacy behavior: use live config retrain_frequency_days
                    model_age_days = (training_end_date - model_date).days
                    retrain_frequency = config.live.get('retrain_frequency_days', 30) if hasattr(config, 'live') else 30
                    
                    if model_age_days <= retrain_frequency:
                        # Validate feature schema if expected_features provided
                        if expected_features is not None:
                            schema_issues = _validate_model_features(model_data, expected_features)
                            if schema_issues:
                                logger.warning(
                                    f"Feature schema mismatch: {'; '.join(schema_issues)}. Retraining..."
                                )
                            else:
                                logger.info(f"Loaded existing model from {model_path} (trained {model_age_days} days ago)")
                                return model_data['model']
                        else:
                            logger.info(f"Loaded existing model from {model_path} (trained {model_age_days} days ago)")
                            return model_data['model']
                    else:
                        logger.info(f"Model is {model_age_days} days old, retraining...")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")


def _validate_model_features(model_data: dict, expected_features: list) -> list:
    """
    Validate model features against expected features.
    
    Returns list of issues (empty if valid).
    """
    from models.tabular_trainer import FeatureSchema
    
    # Try to get feature schema from model data
    if 'feature_schema' in model_data:
        schema = FeatureSchema.from_dict(model_data['feature_schema'])
        is_valid, issues = schema.validate(expected_features, strict=True)
        return issues if not is_valid else []
    
    # Fall back to feature_cols comparison
    model_features = model_data.get('feature_cols', [])
    if not model_features:
        return []  # Can't validate without stored features
    
    # Build schema and validate
    schema = FeatureSchema(
        feature_names=model_features,
        feature_hash=FeatureSchema._compute_hash(model_features),
        horizons=[model_data.get('horizon', 20)],
    )
    is_valid, issues = schema.validate(expected_features, strict=True)
    return issues if not is_valid else []
    
    # Train new model using TabularTrainer
    logger.info(f"Training new {model_type} model...")
    
    # Calculate training period using retraining config if available
    if retraining_config:
        training_start_date = retraining_config.get_window_start(training_end_date)
        logger.info(f"Using retraining policy: {retraining_config.window_type} window, {retraining_config.window_years} years")
        time_decay_enabled = retraining_config.time_decay.enabled
        time_decay_lambda = retraining_config.time_decay.lambda_
        time_decay_min_weight = retraining_config.time_decay.min_weight
    else:
        training_start_date = (pd.Timestamp(training_end_date) - pd.Timedelta(days=train_days)).date()
        time_decay_enabled = False
        time_decay_lambda = 0.001
        time_decay_min_weight = 0.1
    
    # Initialize components
    label_generator = ReturnLabelGenerator(storage)
    feature_pipeline = FeaturePipeline(api, config.features)
    
    # Build TrainingConfig
    training_config = TrainingConfig(
        window_start=training_start_date,
        window_end=training_end_date,
        horizons=[horizon],
        sampling=SamplingStrategy(sample_every_n_days=5),
        time_decay_enabled=time_decay_enabled,
        time_decay_lambda=time_decay_lambda,
        time_decay_min_weight=time_decay_min_weight,
        feature_lookback_days=252,
        benchmark_symbol="SPY",
    )
    
    # Initialize TabularTrainer
    trainer = TabularTrainer(
        feature_pipeline=feature_pipeline,
        label_generator=label_generator,
        storage=storage,
        api=api,
    )
    
    # Determine model class and params
    if model_type == "xgboost":
        model_class = XGBoostModel
        model_params = {"task_type": "regression", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    elif model_type == "lightgbm":
        model_class = LightGBMModel
        model_params = {"task_type": "regression", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train using TabularTrainer
    training_result = trainer.train(
        model_class=model_class,
        model_params=model_params,
        config=training_config,
        universe=universe,
    )
    
    logger.info(f"Training complete: {training_result.num_samples} samples, {training_result.feature_count} features")
    if training_result.sample_weight_range:
        logger.info(f"Sample weight range: [{training_result.sample_weight_range[0]:.3f}, {training_result.sample_weight_range[1]:.3f}]")
    
    # Build feature schema
    feature_schema = FeatureSchema(
        feature_names=training_result.feature_names,
        feature_hash=training_result.feature_hash,
        horizons=training_result.horizons,
        created_date=training_end_date,
    )
    
    # Save model with full metadata
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'model': training_result.model,
        'trained_date': training_end_date,
        'effective_start': training_end_date,
        'effective_end': None,  # Will be set by next retrain
        'model_type': model_type,
        'horizon': horizon,
        # Legacy fields for backwards compatibility
        'feature_cols': training_result.feature_names,
        'feature_count': training_result.feature_count,
        'training_samples': training_result.num_samples,
        'config_hash': get_config_hash(dict(config.features)) if hasattr(config, 'features') else None,
        # New structured metadata
        'training_result': training_result.to_dict(),
        'feature_schema': feature_schema.to_dict(),
        'retraining_config': {
            'cadence_days': retraining_config.cadence_days if retraining_config else None,
            'window_type': retraining_config.window_type if retraining_config else None,
            'window_years': retraining_config.window_years if retraining_config else None,
            'time_decay_enabled': time_decay_enabled,
            'time_decay_lambda': time_decay_lambda,
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {model_path}")
    
    return training_result.model


def write_heartbeat(heartbeat_path: Path, status: str = "running"):
    """Write a heartbeat file for external monitoring."""
    heartbeat_data = {
        'timestamp': datetime.now().isoformat(),
        'status': status,
        'pid': os.getpid(),
    }
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(heartbeat_path, 'w') as f:
        json.dump(heartbeat_data, f)


def get_sector_mapping(storage: StorageBackend) -> dict:
    """Get mapping from asset_id to sector."""
    try:
        assets_df = storage.query("SELECT asset_id, sector FROM assets WHERE sector IS NOT NULL")
        if len(assets_df) == 0:
            return {}
        return dict(zip(assets_df['asset_id'], assets_df['sector']))
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run daily live/paper trading loop")
    parser.add_argument("--trading-date", type=str, help="Trading date (YYYY-MM-DD, default: last trading day)")
    parser.add_argument("--model-path", type=str, help="Path to trained model (default: artifacts/models/live_model.pkl)")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--horizon", type=int, default=20, help="Prediction horizon (days)")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to trade (default: universe_membership)")
    parser.add_argument("--dry-run", action="store_true", help="Don't submit orders")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--skip-if-logged", action="store_true", help="Skip if log already exists for date")
    parser.add_argument("--force", action="store_true", 
                       help="Force execution: bypass trading day check and idempotency. "
                            "Use for testing or getting insights on weekends/holidays. "
                            "Implies --dry-run unless --no-dry-run is also specified.")
    parser.add_argument("--no-dry-run", action="store_true", 
                       help="Actually submit orders even with --force (use with caution!)")
    
    # Broker selection - defaults to BROKER env var, then "paper"
    default_broker = os.environ.get("BROKER", "paper")
    parser.add_argument("--broker", type=str, default=default_broker, 
                       choices=["paper", "alpaca_paper", "alpaca_live"],
                       help="Broker to use: paper (simulated), alpaca_paper (Alpaca paper trading), alpaca_live (Alpaca live). Can also set via BROKER env var.")
    
    # Top-K - defaults to TOP_K env var, then 10
    default_top_k = int(os.environ.get("TOP_K", "10"))
    parser.add_argument("--top-k", type=int, default=default_top_k, 
                       help="Number of top-scoring assets to hold. Can also set via TOP_K env var.")
    
    # Regime-aware options
    parser.add_argument("--regime-aware", action="store_true", help="Enable regime-aware exposure scaling")
    parser.add_argument("--sector-tilts", action="store_true", help="Enable defensive sector rotation")
    parser.add_argument("--vol-scaling", action="store_true", help="Enable VIX-style volatility scaling")
    parser.add_argument("--drawdown-throttle", action="store_true", help="Enable drawdown throttling")
    parser.add_argument("--regime-model-path", type=str, help="Path to fitted regime model")
    
    # Cloud deployment options
    parser.add_argument("--heartbeat-path", type=str, help="Path to write heartbeat file for monitoring")
    parser.add_argument("--max-runtime-minutes", type=int, default=30, help="Maximum runtime before timeout")
    
    # Alerting options
    parser.add_argument("--enable-alerts", action="store_true", help="Enable email/Slack alerts")
    parser.add_argument("--alert-dry-run", action="store_true", help="Log alerts but don't send them")
    
    # Retraining policy options
    parser.add_argument("--no-retraining-policy", action="store_true",
                       help="Disable retraining policy (use legacy retrain_frequency_days)")
    parser.add_argument("--retrain-cadence", type=int, default=None,
                       help="Override retraining cadence from config (days)")
    parser.add_argument("--time-decay-lambda", type=float, default=None,
                       help="Override time-decay lambda. Higher = more emphasis on recent data")
    
    args = parser.parse_args()
    
    # Initialize alert manager
    global alert_manager
    alert_manager = AlertManager(
        email_enabled=args.enable_alerts,
        slack_enabled=args.enable_alerts,
        dry_run=args.alert_dry_run or args.dry_run
    )
    
    # Write initial heartbeat if path specified
    heartbeat_path = Path(args.heartbeat_path) if args.heartbeat_path else None
    if heartbeat_path:
        write_heartbeat(heartbeat_path, "starting")
    
    # Initialize storage and config first
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    # Determine requested trading date
    calendar = TradingCalendar()
    if args.trading_date:
        requested_date = datetime.strptime(args.trading_date, "%Y-%m-%d").date()
    else:
        requested_date = None  # Let prepare_for_trading determine the best date
    
    # Comprehensive data preparation and validation
    # This handles: bootstrap, top-up, date adjustment, and sufficiency validation
    # In force_mode, skip validation and just use whatever data is available
    logger.info("Preparing data for trading...")
    prep_result = prepare_for_trading(
        storage=storage,
        config=config.__dict__,
        requested_date=requested_date,
        lookback_days=252,  # Feature lookback
        train_days=750,     # Training data requirement
        auto_fetch=config.data.auto_fetch_on_live,
        force_mode=args.force  # Skip validation in force mode
    )
    
    # Use the effective trading date from preparation
    trading_date = prep_result['trading_date']
    
    logger.info(f"{'='*60}")
    logger.info(f"LIVE TRADING LOOP - {trading_date}")
    logger.info(f"{'='*60}")
    
    # Log any warnings
    for warning in prep_result.get('warnings', []):
        logger.warning(warning)
    
    # Check if data is ready
    if not prep_result['ready']:
        for issue in prep_result['issues']:
            logger.error(f"Data issue: {issue}")
        
        if args.enable_alerts and alert_manager:
            alert_manager.send_error_alert(
                error_type="Data Preparation",
                error_message="; ".join(prep_result['issues']),
                context={
                    "trading_date": str(trading_date),
                    "coverage": prep_result.get('coverage'),
                    "fetch_result": prep_result.get('fetch_result')
                }
            )
        storage.close()
        sys.exit(1)  # Exit with error code for cron/scheduler to detect
    
    # Log bootstrap if it happened
    if prep_result.get('bootstrapped'):
        logger.info("Database was auto-bootstrapped with full historical data")
    
    logger.info(f"Data ready: {prep_result['coverage']['num_assets']} assets, "
                f"data from {prep_result['coverage']['min_date']} to {prep_result['coverage']['max_date']}")
    
    # Handle --force flag
    # When --force is used, we bypass safety checks for testing/insight purposes
    force_mode = args.force
    if force_mode:
        # --force implies --dry-run unless --no-dry-run is explicitly set
        if not args.no_dry_run:
            args.dry_run = True
            logger.warning("FORCE MODE: Running with --dry-run (use --no-dry-run to actually submit orders)")
        else:
            logger.warning("FORCE MODE: --no-dry-run specified, orders WILL be submitted!")
    
    # Check if today is a trading day - if not, skip execution
    # This prevents duplicate orders when cron runs on weekends/holidays
    calendar = TradingCalendar()
    today = date.today()
    
    if not calendar.is_trading_day(today) and not force_mode:
        # Today is not a trading day (weekend or holiday)
        # The trading_date will be the last trading day, but we shouldn't execute
        # because we already would have executed on that day
        logger.info(f"Today ({today}) is not a trading day (weekend/holiday). "
                   f"Skipping execution to avoid duplicate orders.")
        logger.info(f"Next trading day: {calendar.next_trading_day(today)}")
        logger.info("Use --force to run anyway (for testing/insights)")
        storage.close()
        return
    elif not calendar.is_trading_day(today) and force_mode:
        logger.warning(f"FORCE MODE: Running on non-trading day ({today}). "
                      f"Using data as-of {trading_date} for insights only.")
    
    # Check for existing log (idempotency)
    # This prevents duplicate orders if the script runs multiple times on the same day
    log_dir = Path("logs") / "live_trading"
    log_file = log_dir / f"daily_log_{trading_date}.json"
    
    if args.skip_if_logged and log_file.exists() and not force_mode:
        # Read the log to check when it was created
        try:
            with open(log_file, 'r') as f:
                existing_log = json.load(f)
            log_timestamp = existing_log.get('timestamp', 'unknown')
            logger.info(f"Log already exists for {trading_date} (created: {log_timestamp}), "
                       f"skipping to avoid duplicate orders (use --force to override)")
        except Exception:
            logger.info(f"Log already exists for {trading_date}, skipping (use --force to override)")
        storage.close()
        return
    elif args.skip_if_logged and log_file.exists() and force_mode:
        logger.warning(f"FORCE MODE: Bypassing idempotency check (log exists for {trading_date})")
    
    # Initialize API
    api = AsOfQueryAPI(storage)
    
    # Get universe
    if args.symbols:
        symbol_to_asset_id_map = get_symbol_to_asset_id(storage)
        universe = {symbol_to_asset_id_map[s] for s in args.symbols if s in symbol_to_asset_id_map}
        logger.info(f"Using specified symbols: {args.symbols}")
    else:
        try:
            universe = api.get_universe_at_date(trading_date, index_name="SP500")
            if len(universe) == 0:
                bars_df = api.get_bars_asof(trading_date)
                universe = set(bars_df['asset_id'].unique())
                logger.info(f"Using all assets in database: {len(universe)}")
            else:
                logger.info(f"Using S&P 500 universe: {len(universe)} assets")
        except Exception as e:
            logger.warning(f"Could not load universe from membership table: {e}")
            bars_df = api.get_bars_asof(trading_date)
            universe = set(bars_df['asset_id'].unique())
            logger.info(f"Using all assets in database: {len(universe)}")
    
    if len(universe) == 0:
        logger.error("No assets found in universe")
        storage.close()
        return
    
    # Get mappings
    asset_id_to_symbol = get_asset_id_to_symbol(storage)
    
    # Load or train model
    model_path = Path(args.model_path) if args.model_path else Path("artifacts/models/live_model.pkl")
    
    # Override retraining config if command-line args provided
    use_retraining_policy = not args.no_retraining_policy
    if use_retraining_policy and hasattr(config, 'retraining'):
        if args.retrain_cadence is not None:
            config.retraining.cadence_days = args.retrain_cadence
            logger.info(f"Overriding retrain cadence to {args.retrain_cadence} days")
        if args.time_decay_lambda is not None:
            config.retraining.time_decay.lambda_ = args.time_decay_lambda
            logger.info(f"Overriding time-decay lambda to {args.time_decay_lambda}")
    
    try:
        model = load_or_train_model(
            model_path=model_path,
            model_type=args.model_type,
            storage=storage,
            api=api,
            config=config,
            training_end_date=(pd.Timestamp(trading_date) - pd.Timedelta(days=1)).date(),
            universe=universe,
            horizon=args.horizon,
            force_retrain=args.force_retrain,
            use_retraining_policy=use_retraining_policy
        )
    except Exception as e:
        logger.error(f"Failed to load/train model: {e}")
        if args.enable_alerts and alert_manager:
            alert_manager.send_error_alert(
                error_type="Model Loading/Training",
                error_message=str(e),
                context={"trading_date": str(trading_date), "model_type": args.model_type}
            )
        storage.close()
        return
    
    # Initialize regime and exposure components
    regime_descriptor = "unknown"
    exposure_scale = 1.0
    regime_metrics_service = RegimeMetricsService(api)
    
    # Always try to get current regime for display/logging (even if not using regime-aware features)
    try:
        regime_id, regime_descriptor = regime_metrics_service.get_current_regime(trading_date)
        logger.info(f"Current market regime: {regime_descriptor}")
    except Exception as e:
        logger.debug(f"Could not determine regime: {e}")
    
    # Use regime model if specified and regime-aware mode enabled
    if args.regime_aware or args.sector_tilts or args.vol_scaling:
        try:
            # Load regime model if path specified
            if args.regime_model_path and Path(args.regime_model_path).exists():
                regime_generator = RegimeLabelGenerator(storage)
                regime_generator.load_model(args.regime_model_path)
                regime_id, regime_descriptor = regime_generator.predict_regime(trading_date)
                logger.info(f"Regime from model: {regime_descriptor}")
        except Exception as e:
            logger.warning(f"Could not determine regime from model: {e}, using database regime")
            if regime_descriptor == "unknown":
                regime_descriptor = "bear_low_vol"  # Conservative default
    
    # Initialize exposure manager
    exposure_manager = None
    if args.regime_aware or args.vol_scaling:
        # Get regime policy from config
        regime_policy = config.portfolio.get('regime_policy', {}).get('exposure_multipliers', {})
        vol_config = config.portfolio.get('volatility_scaling', {})
        
        exposure_manager = ExposureManager(
            regime_policy=regime_policy if regime_policy else None,
            volatility_config=vol_config if vol_config else None,
            enabled=True
        )
        
        # Update regime
        exposure_manager.update_regime(regime_descriptor)
        
        # Update volatility if vol scaling enabled
        if args.vol_scaling:
            realized_vol = regime_metrics_service.get_current_volatility(trading_date)
            exposure_manager.update_volatility(realized_vol)
            logger.info(f"Realized vol (20d): {realized_vol:.2%}")
        
        exposure_scale = exposure_manager.get_combined_scale()
        logger.info(f"Exposure scale (regime + vol): {exposure_scale:.2%}")
    
    # Initialize drawdown manager
    drawdown_manager = None
    if args.drawdown_throttle:
        dd_config = config.portfolio.get('drawdown', {})
        drawdown_manager = DrawdownManager(
            throttle_threshold_pct=dd_config.get('throttle_threshold_pct', 0.15),
            max_drawdown_pct=dd_config.get('max_drawdown_pct', 0.25),
            min_scale_factor=dd_config.get('min_scale_factor', 0.25),
            recovery_threshold_pct=dd_config.get('recovery_threshold_pct', 0.05)
        )
    
    # Initialize sector tilt manager
    sector_tilt_manager = None
    sector_mapping = {}
    if args.sector_tilts:
        sector_policy = config.portfolio.get('sector_policy', {}).get('tilts', {})
        sector_tilt_manager = SectorTiltManager(
            sector_policy=sector_policy if sector_policy else None,
            enabled=True
        )
        sector_tilt_manager.update_regime(regime_descriptor)
        sector_mapping = get_sector_mapping(storage)
        logger.info(f"Sector tilts enabled for regime: {regime_descriptor}")
    
    # Initialize risk manager with sector tilt support
    risk_config = config.portfolio.get('risk', {})
    risk_manager = RiskManager(
        max_position_pct=risk_config.get('max_position_pct', 0.20),
        min_position_pct=risk_config.get('min_position_pct', 0.02),
        max_sector_pct=risk_config.get('max_sector_pct', 0.40),
        max_gross_exposure=risk_config.get('max_gross_exposure', 1.0),
        max_net_exposure=risk_config.get('max_net_exposure', 1.0),
        stop_loss_pct=risk_config.get('stop_loss_pct', 0.10),
        sector_mapping=sector_mapping,
        sector_tilt_manager=sector_tilt_manager
    )
    
    # Initialize components
    feature_pipeline = FeaturePipeline(api, config.features)
    
    # Check if model features match current feature pipeline
    # Build a sample to get current features
    sample_features = feature_pipeline.build_features_cross_sectional(trading_date, universe)
    exclude_cols = {'asset_id', 'date'}
    current_feature_names = sorted([c for c in sample_features.columns if c not in exclude_cols])
    
    # Validate model features against current pipeline
    # This catches schema drift (not just count, but names and order)
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        schema_issues = _validate_model_features(model_data, current_feature_names)
        if schema_issues:
            logger.warning(
                f"Feature schema mismatch detected: {'; '.join(schema_issues[:3])}. Retraining model..."
            )
            # Retrain - let exceptions propagate so we know if it fails
            model = load_or_train_model(
                model_path=model_path,
                model_type=args.model_type,
                storage=storage,
                api=api,
                config=config,
                training_end_date=(pd.Timestamp(trading_date) - pd.Timedelta(days=1)).date(),
                universe=universe,
                horizon=args.horizon,
                force_retrain=True,  # Force retrain due to feature mismatch
                expected_features=current_feature_names,
            )
    except Exception as e:
        logger.debug(f"Could not validate model features: {e}")
    
    strategy = LongTopKStrategy(
        k=args.top_k,
        min_score_threshold=-np.inf,
        risk_manager=risk_manager,
        exposure_manager=exposure_manager
    )
    # Initialize broker based on selection
    if args.broker == "paper":
        broker = PaperBroker(initial_capital=args.initial_capital, api=api, as_of_date=trading_date)
        logger.info("Using PaperBroker (simulated)")
    elif args.broker == "alpaca_paper":
        from live.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(paper=True)
        logger.info("Using AlpacaBroker (PAPER mode)")
    elif args.broker == "alpaca_live":
        from live.alpaca_broker import AlpacaBroker
        # Safety check: require explicit environment variable for live trading
        if os.environ.get("LIVE_TRADING_ENABLED") != "1":
            logger.error("Live trading requires LIVE_TRADING_ENABLED=1 environment variable")
            if args.enable_alerts and alert_manager:
                alert_manager.send_error_alert(
                    error_type="Live Trading Blocked",
                    error_message="Attempted live trading without LIVE_TRADING_ENABLED=1",
                    context={"trading_date": str(trading_date)}
                )
            storage.close()
            return
        broker = AlpacaBroker(paper=False)
        logger.info("Using AlpacaBroker (LIVE mode) - REAL MONEY")
    else:
        logger.error(f"Unknown broker: {args.broker}")
        storage.close()
        return
    
    # Update heartbeat
    if heartbeat_path:
        write_heartbeat(heartbeat_path, "running")
    
    # Create LiveEngine instance with all components
    engine = LiveEngine(
        api=api,
        feature_pipeline=feature_pipeline,
        model=model,
        strategy=strategy,
        broker=broker,
        config=config.__dict__,
        asset_id_to_symbol=asset_id_to_symbol,
        exposure_manager=exposure_manager,
        drawdown_manager=drawdown_manager
    )
    
    # Run the daily loop (this handles features, predictions, position-aware rebalancing, orders, and logging)
    loop_results = None
    try:
        loop_results = engine.run_daily_loop(
            trading_date, 
            dry_run=args.dry_run, 
            universe=universe,
            skip_preflight=args.force  # Skip idempotency check in force mode
        )
    except Exception as e:
        logger.error(f"Error in daily loop: {e}")
        if args.enable_alerts and alert_manager:
            alert_manager.send_error_alert(
                error_type="Daily Loop Execution",
                error_message=str(e),
                context={"trading_date": str(trading_date)}
            )
        raise
    
    # Daily loop completed - LiveEngine handles features, predictions, position-aware rebalancing, orders, and logging
    # The detailed JSON log is saved by LiveEngine._log_daily_results
    
    # Update heartbeat with completion
    if heartbeat_path:
        write_heartbeat(heartbeat_path, "completed")
    
    # Extract results from engine (or use defaults if loop didn't complete)
    if loop_results:
        result_regime = loop_results.get('regime', regime_descriptor)
        result_exposure = loop_results.get('exposure_scale', exposure_scale)
        result_orders = loop_results.get('orders_count', 0)
        result_targets = len(loop_results.get('target_positions', {}))
        result_positive_scores = loop_results.get('positive_score_count', 0)
        result_universe_size = loop_results.get('universe_size', 0)
    else:
        result_regime = regime_descriptor
        result_exposure = exposure_scale
        result_orders = 0
        result_targets = 0
        result_positive_scores = 0
        result_universe_size = 0
    
    # Print summary
    print("\n" + "="*60)
    print("DAILY TRADING SUMMARY")
    print("="*60)
    print(f"Date: {trading_date}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Model: {args.model_type} ({args.horizon}d horizon)")
    print(f"Regime: {result_regime}")
    print(f"Exposure Scale: {result_exposure:.0%}")
    print(f"Account Value: ${broker.get_account_value():,.2f}")
    print(f"Cash: ${broker.get_cash():,.2f}")
    print(f"Buying Power: ${broker.get_buying_power():,.2f}")
    if args.dry_run:
        print(f"Target Positions: {result_targets} (top-k={args.top_k})")
        print(f"Orders (would submit): {result_orders}")
    else:
        print(f"Positions: {len(broker.get_positions())} (top-k={args.top_k})")
        print(f"Orders Submitted: {result_orders}")
    print(f"Positive Scores: {result_positive_scores}/{result_universe_size} assets")
    
    if args.regime_aware:
        print(f"\nRegime-Aware Features:")
        print(f"  Regime Scaling: {'Enabled' if args.regime_aware else 'Disabled'}")
        print(f"  Sector Tilts: {'Enabled' if args.sector_tilts else 'Disabled'}")
        print(f"  Vol Scaling: {'Enabled' if args.vol_scaling else 'Disabled'}")
        print(f"  DD Throttle: {'Enabled' if args.drawdown_throttle else 'Disabled'}")
    
    print("="*60)
    
    # Send daily summary alert if enabled
    if args.enable_alerts and alert_manager:
        alert_manager.send_daily_summary(
            trading_date=str(trading_date),
            regime=regime_descriptor,
            exposure_scale=exposure_scale,
            account_value=broker.get_account_value(),
            positions=len(broker.get_positions()),
            orders=0,  # Order details are logged by LiveEngine
            additional_metrics={
                "Mode": "DRY RUN" if args.dry_run else "LIVE",
                "Model": args.model_type,
            }
        )
    
    storage.close()


if __name__ == "__main__":
    main()
