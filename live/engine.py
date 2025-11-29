"""Live trading engine."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Set, Any
from datetime import date, datetime, time
from pathlib import Path
import json
from data.asof_api import AsOfQueryAPI
from features.pipeline import FeaturePipeline
from features.regime_metrics import RegimeMetricsService, RegimeMetrics
from portfolio.strategies import LongTopKStrategy
from portfolio.scores import ScoreConverter
from portfolio.risk import ExposureManager, DrawdownManager
from live.broker_base import BrokerAPI
from loguru import logger


class LiveEngine:
    """
    Live trading engine that runs daily inference and execution.
    
    Supports:
    - Regime-aware exposure scaling
    - VIX-style volatility-based position sizing
    - Drawdown throttling
    - Defensive sector rotation (via strategy/risk manager)
    """
    
    def __init__(
        self,
        api: AsOfQueryAPI,
        feature_pipeline: FeaturePipeline,
        model,  # Trained model (can be sklearn, torch, or dict of models)
        strategy: LongTopKStrategy,
        broker: BrokerAPI,
        config: Optional[Dict] = None,
        asset_id_to_symbol: Optional[Dict[int, str]] = None,
        exposure_manager: Optional[ExposureManager] = None,
        drawdown_manager: Optional[DrawdownManager] = None
    ):
        self.api = api
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.strategy = strategy
        self.broker = broker
        self.config = config or {}
        
        # Asset mapping - load from DB if not provided
        if asset_id_to_symbol is None:
            self.asset_id_to_symbol = self._load_asset_mapping()
        else:
            self.asset_id_to_symbol = asset_id_to_symbol
        
        self.symbol_to_asset_id = {v: k for k, v in self.asset_id_to_symbol.items()}
        
        # Risk management
        self.exposure_manager = exposure_manager
        self.drawdown_manager = drawdown_manager
        
        # Regime metrics service for getting current regime and volatility
        # This provides a clean public API instead of reaching into private methods
        self.regime_metrics_service = RegimeMetricsService(api)
        
        # Logs directory
        self.logs_dir = Path(self.config.get('logs_dir', 'logs/live_trading'))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_asset_mapping(self) -> Dict[int, str]:
        """Load asset_id to symbol mapping from database."""
        try:
            assets_df = self.api.storage.query("SELECT asset_id, symbol FROM assets")
            return dict(zip(assets_df['asset_id'], assets_df['symbol']))
        except Exception:
            return {}
    
    def run_daily_loop(
        self, 
        trading_date: date, 
        dry_run: bool = False, 
        universe: Optional[Set[int]] = None,
        skip_preflight: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Run daily trading loop.
        
        This should be called after market close on trading_date.
        
        Args:
            trading_date: The trading date to run for
            dry_run: If True, don't submit orders (for testing)
            universe: Optional pre-computed universe. If None, will fetch from API.
            skip_preflight: If True, skip pre-flight checks (for --force mode)
            
        Returns:
            Dict with results including regime, exposure_scale, orders, target_positions
            or None if preflight checks failed
        """
        logger.info(f"Running daily loop for {trading_date}")
        
        # Pre-flight checks (can be skipped in force mode)
        if not skip_preflight and not self._run_preflight_checks(trading_date):
            logger.error("Pre-flight checks failed, aborting")
            return None
        
        # 1. Get latest data
        if universe is None:
            universe = self.api.get_universe_at_date(trading_date)
        logger.info(f"Universe size: {len(universe)}")
        
        # 2. Get current regime and update exposure manager
        current_regime, regime_descriptor = self._get_current_regime(trading_date)
        exposure_scale = self._compute_exposure_scale(trading_date, regime_descriptor)
        
        logger.info(f"Current regime: {regime_descriptor}, exposure scale: {exposure_scale:.2f}")
        
        # 3. Build features
        features_df = self.feature_pipeline.build_features_cross_sectional(
            trading_date,
            universe
        )
        
        if len(features_df) == 0:
            logger.warning("No features generated")
            return None
        
        # 4. Run model inference
        predictions = self._predict(features_df)
        
        # 5. Convert predictions to scores
        scores_df = self._predictions_to_scores(predictions, features_df)
        
        # Count assets with positive scores (would be candidates if unrestricted)
        positive_score_count = len(scores_df[scores_df['score'] > 0]) if 'score' in scores_df.columns else len(scores_df)
        
        # 6. Compute target positions (with exposure scaling and sector tilts)
        current_positions = self._get_current_positions_dict()
        target_weights = self.strategy.compute_weights(
            scores_df,
            current_positions,
            trading_date,
            exposure_scale=exposure_scale,
            current_regime=regime_descriptor
        )
        
        # 7. Generate orders to rebalance
        orders = self._generate_rebalance_orders(target_weights, current_positions)
        
        # 8. Submit orders (unless dry run)
        if not dry_run:
            for order in orders:
                order_id = self.broker.submit_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    quantity=order['quantity'],
                    order_type=order.get('order_type', 'market'),
                    time_in_force=order.get('time_in_force', 'day')
                )
                logger.info(f"Submitted order: {order_id} for {order['symbol']}")
        else:
            logger.info(f"DRY RUN: Would submit {len(orders)} orders")
        
        # 9. Log results
        self._log_daily_results(
            trading_date, predictions, target_weights, orders,
            current_positions=current_positions,
            regime_descriptor=regime_descriptor, exposure_scale=exposure_scale
        )
        
        # 10. Return results for caller
        return {
            'trading_date': trading_date,
            'regime': regime_descriptor,
            'exposure_scale': exposure_scale,
            'target_positions': target_weights,
            'orders': orders,
            'orders_count': len(orders),
            'dry_run': dry_run,
            'positive_score_count': positive_score_count,  # How many assets had positive scores (unrestricted candidates)
            'universe_size': len(universe)
        }
    
    def _run_preflight_checks(self, trading_date: date) -> bool:
        """
        Run pre-flight checks before trading.
        
        Returns:
            True if all checks pass, False otherwise
        """
        # Check if we already ran for this date (idempotency)
        if self.check_log_exists(trading_date):
            logger.warning(f"Log already exists for {trading_date}, skipping to avoid duplicate orders")
            return False
        
        # Check data freshness
        try:
            bars = self.api.get_bars_asof(trading_date, lookback_days=5)
            if len(bars) == 0:
                logger.error("No recent bar data available")
                return False
            
            latest_date = bars['date'].max()
            if hasattr(latest_date, 'date'):
                latest_date = latest_date.date()
            
            days_stale = (trading_date - latest_date).days
            if days_stale > 3:
                logger.warning(f"Data is {days_stale} days stale (latest: {latest_date})")
                # Don't fail, just warn
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return False
        
        # Check broker connection
        try:
            account_value = self.broker.get_account_value()
            if account_value <= 0:
                logger.error("Invalid account value from broker")
                return False
            logger.info(f"Account value: ${account_value:,.2f}")
        except Exception as e:
            logger.error(f"Error connecting to broker: {e}")
            return False
        
        return True
    
    def _get_current_regime(self, trading_date: date) -> Tuple[int, str]:
        """Get current regime ID and descriptor."""
        return self.regime_metrics_service.get_current_regime(trading_date)
    
    def get_regime_metrics(self, trading_date: date) -> RegimeMetrics:
        """
        Get full regime metrics for a trading date.
        
        This is the public API for accessing regime information.
        """
        return self.regime_metrics_service.get_regime_metrics(trading_date)
    
    def _compute_exposure_scale(self, trading_date: date, regime_descriptor: str) -> float:
        """
        Compute combined exposure scale factor.
        
        Combines:
        1. Regime-based scaling
        2. Volatility-based scaling (VIX-style)
        3. Drawdown-based scaling
        """
        if self.exposure_manager is None:
            return 1.0
        
        # Update regime
        self.exposure_manager.update_regime(regime_descriptor)
        
        # Update volatility using the public regime metrics service
        realized_vol = self.regime_metrics_service.get_current_volatility(trading_date)
        self.exposure_manager.update_volatility(realized_vol)
        
        # Get drawdown scale
        drawdown_scale = 1.0
        if self.drawdown_manager is not None:
            current_equity = self.broker.get_account_value()
            drawdown_scale = self.drawdown_manager.update(current_equity)
        
        return self.exposure_manager.get_combined_scale(drawdown_scale)
    
    def _predict(self, features_df: pd.DataFrame) -> Dict:
        """
        Run model inference.
        
        Supports:
        - Dict of sklearn/xgboost models (one per horizon) - multi-horizon tabular
        - Single sklearn/xgboost model - single-horizon tabular
        - EnsembleXGBoostModel with predict_with_uncertainty
        - PyTorch sequence models (ConvLSTM, TCN) with predict_mc - multi-horizon sequence
        
        Returns:
            Dict mapping horizon -> (predictions, uncertainties)
        """
        # Check if this is a sequence model (PyTorch)
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                # Sequence model - need to build sequences
                sequence_length = self.config.get('sequence_length', 60)
                trading_date = features_df['date'].iloc[0] if 'date' in features_df.columns else None
                universe = set(features_df['asset_id'].unique()) if 'asset_id' in features_df.columns else None
                
                if trading_date is None or universe is None:
                    logger.warning("Cannot build sequences without date and universe")
                    return {}
                
                # Build sequence ending at trading_date
                sequence_array = self.feature_pipeline.build_features_sequence(
                    as_of_date=trading_date,
                    sequence_length=sequence_length,
                    universe=universe
                )
                
                if sequence_array.size == 0:
                    logger.warning("No sequences built")
                    return {}
                
                # Convert to tensor
                sequence_tensor = torch.FloatTensor(sequence_array)
                device = next(self.model.parameters()).device
                sequence_tensor = sequence_tensor.to(device)
                
                # Predict with uncertainty if model supports it
                if hasattr(self.model, 'predict_mc'):
                    predictions = self.model.predict_mc(sequence_tensor, n_samples=50)
                else:
                    # Single forward pass
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model(sequence_tensor)
                        predictions = {}
                        for horizon, pred in outputs.items():
                            if isinstance(pred, tuple):
                                # Mean-variance output
                                mean = pred[0].cpu().numpy()
                                logvar = pred[1].cpu().numpy()
                                std = np.sqrt(np.exp(logvar))
                                predictions[horizon] = (mean, std)
                            else:
                                # Single output - use heuristic uncertainty
                                mean = pred.cpu().numpy()
                                std = np.abs(mean) * 0.1 + 0.05
                                predictions[horizon] = (mean, std)
                
                # Map predictions back to asset_ids in features_df
                # sequence_array shape: (num_assets, seq_len, features)
                # features_df has asset_ids - need to align
                if 'asset_id' in features_df.columns:
                    asset_ids = features_df['asset_id'].values
                    # Get asset_ids from sequence (from build_features_sequence)
                    # The sequence order should match the universe order
                    aligned_predictions = {}
                    for horizon in predictions:
                        mu, sigma = predictions[horizon]
                        # If lengths match, use as-is; otherwise need to map
                        if len(mu) == len(asset_ids):
                            aligned_predictions[horizon] = (mu, sigma)
                        else:
                            # Mismatch - try to align by asset_id
                            # For now, use first N or pad
                            n = min(len(mu), len(asset_ids))
                            aligned_predictions[horizon] = (mu[:n], sigma[:n])
                    predictions = aligned_predictions
                
                return predictions
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error in sequence model prediction: {e}")
        
        # Tabular model path
        # Extract feature columns (exclude asset_id, date)
        feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
        X = features_df[feature_cols].copy()
        
        # Ensure numeric - convert to float64 explicitly
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        # Convert to float64 to avoid object dtype issues
        X = X.astype(np.float64)
        
        # Check model type
        if self.model is None:
            logger.error("Model is None - cannot make predictions")
            return {}
        
        if isinstance(self.model, dict):
            # Dict of models (one per horizon) - multi-horizon tabular
            predictions = {}
            for horizon, model in self.model.items():
                if model is None:
                    logger.warning(f"Model for horizon {horizon} is None, skipping")
                    continue
                pred, sigma = self._predict_single_model(model, X)
                predictions[horizon] = (pred, sigma)
            return predictions
        else:
            # Single model - single-horizon tabular
            pred, sigma = self._predict_single_model(self.model, X)
            # Get horizon from config or default to 20
            horizon = self.config.get('horizon', 20)
            return {horizon: (pred, sigma)}
    
    def _predict_single_model(self, model, X: pd.DataFrame) -> tuple:
        """
        Make predictions from a single model with uncertainty estimation.
        
        Args:
            model: Model to use for prediction
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Check if model is None
        if model is None:
            logger.error("Model is None - cannot make predictions")
            raise ValueError("Model is None")
        
        # Ensure X is properly numeric before any model calls
        # Convert to float64 to avoid object dtype issues
        X_clean = X.copy()
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(0).astype(np.float64)
        
        # Check for ensemble model with uncertainty support
        if hasattr(model, 'predict_with_uncertainty'):
            # EnsembleXGBoostModel or similar
            try:
                pred, sigma = model.predict_with_uncertainty(X_clean)
                return pred, sigma
            except Exception as e:
                logger.warning(f"predict_with_uncertainty failed, falling back to predict: {e}")
        
        # Check for sklearn/xgboost style model
        if hasattr(model, 'predict'):
            try:
                pred = model.predict(X_clean)
                
                # Try to estimate uncertainty from model internals
                sigma = self._estimate_uncertainty(model, X_clean, pred)
                return pred, sigma
            except Exception as e:
                logger.warning(f"predict failed: {e}")
        
        # PyTorch model
        try:
            import torch
            # Check if model is actually a PyTorch model
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Model is not a PyTorch module: {type(model)}")
            
            # Ensure X is numeric and convert to numpy array
            X_array = X_clean.values
            # Double-check dtype
            if X_array.dtype == np.object_ or X_array.dtype == object:
                X_array = X_clean.astype(np.float64).values
            X_tensor = torch.FloatTensor(X_array)
            
            if hasattr(model, 'predict_mc'):
                # MC dropout for uncertainty
                mc_predictions = model.predict_mc(X_tensor, n_samples=50)
                if isinstance(mc_predictions, dict):
                    # Return first horizon's predictions
                    first_horizon = list(mc_predictions.keys())[0]
                    return mc_predictions[first_horizon]
                return mc_predictions
            else:
                # Standard prediction
                model.eval()
                with torch.no_grad():
                    outputs = model(X_tensor)
                    if isinstance(outputs, dict):
                        first_horizon = list(outputs.keys())[0]
                        pred = outputs[first_horizon].cpu().numpy()
                    else:
                        pred = outputs.cpu().numpy()
                    sigma = np.ones(len(pred)) * 0.1
                    return pred, sigma
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyTorch model prediction failed: {e}")
            # Don't raise here, fall through to final error
        
        raise ValueError(f"Unknown model type or model has no valid prediction method: {type(model)}")
    
    def _estimate_uncertainty(self, model, X: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty for models without built-in uncertainty.
        
        Uses heuristics based on model type:
        - XGBoost: Use feature importance variance or tree variance
        - Other: Use a constant based on prediction variance
        
        Args:
            model: The model
            X: Features
            predictions: Model predictions
            
        Returns:
            Array of uncertainty estimates
        """
        n_samples = len(predictions)
        
        # Try to get uncertainty from XGBoost tree variance
        if hasattr(model, 'model') and hasattr(model.model, 'get_booster'):
            try:
                booster = model.model.get_booster()
                # Use prediction margin variance as uncertainty proxy
                # This is a rough approximation
                pred_std = np.std(predictions)
                # Scale by prediction magnitude
                sigma = np.abs(predictions - np.mean(predictions)) * 0.1 + pred_std * 0.05
                sigma = np.maximum(sigma, 0.01)  # Minimum uncertainty
                return sigma
            except Exception:
                pass
        
        # Default: use scaled prediction variance
        pred_std = max(np.std(predictions), 0.01)
        sigma = np.ones(n_samples) * pred_std * 0.5
        return sigma
    
    def _predictions_to_scores(
        self,
        predictions: Dict,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert predictions to scores using config-driven multi-horizon blending.
        
        Uses MultiHorizonConfig from config if available, otherwise defaults.
        """
        from portfolio.scores import ScoreConverter, MultiHorizonConfig
        
        # Get multi-horizon config from config
        multi_horizon_config = None
        if 'multi_horizon' in self.config:
            multi_horizon_config = MultiHorizonConfig.from_config(self.config['multi_horizon'])
        elif 'portfolio' in self.config and 'multi_horizon' in self.config['portfolio']:
            multi_horizon_config = MultiHorizonConfig.from_config(self.config['portfolio']['multi_horizon'])
        
        # Create converter (uses defaults if no config)
        converter = ScoreConverter(config=multi_horizon_config)
        
        scores = []
        
        for idx, row in features_df.iterrows():
            asset_id = row['asset_id']
            
            # Combine multi-horizon predictions
            pred_dict = {}
            for horizon, pred in predictions.items():
                if isinstance(pred, tuple):
                    mu, sigma = pred
                    pred_dict[horizon] = {'mu': float(mu[idx]), 'sigma': float(sigma[idx])}
                else:
                    pred_dict[horizon] = {'mu': float(pred[idx]), 'sigma': 0.1}  # Default uncertainty
            
            # Combine scores using config-driven converter
            score = converter.combine_scores(pred_dict)
            
            # Compute confidence (from uncertainty)
            avg_uncertainty = np.mean([p['sigma'] for p in pred_dict.values()])
            confidence = 1.0 / (1.0 + avg_uncertainty)  # Inverse of uncertainty
            
            scores.append({
                'asset_id': asset_id,
                'score': score,
                'confidence': confidence
            })
        
        return pd.DataFrame(scores)
    
    def _get_current_positions_dict(self) -> Dict[int, float]:
        """
        Get current positions as dict of asset_id -> current_weight.
        """
        positions = self.broker.get_positions()
        total_value = self.broker.get_account_value()
        
        if total_value <= 0:
            return {}
        
        result = {}
        for pos in positions:
            symbol = pos['symbol']
            asset_id = self.symbol_to_asset_id.get(symbol)
            if asset_id is not None:
                # Compute weight
                quote = self.broker.get_quote(symbol)
                market_value = pos['quantity'] * quote['last']
                weight = market_value / total_value
                result[asset_id] = weight
        
        return result
    
    def _generate_rebalance_orders(
        self,
        target_weights: pd.DataFrame,
        current_positions: Dict[int, float]
    ) -> List[Dict]:
        """
        Generate orders to rebalance from current to target weights.
        
        Args:
            target_weights: DataFrame with columns: asset_id, weight
            current_positions: Dict of asset_id -> current_weight
        
        Returns:
            List of order dicts with symbol, side, quantity, order_type
        """
        orders = []
        total_value = self.broker.get_account_value()
        
        if total_value <= 0:
            return orders
        
        # Get all asset_ids involved
        target_asset_ids = set(target_weights['asset_id'].values) if len(target_weights) > 0 else set()
        current_asset_ids = set(current_positions.keys())
        all_asset_ids = target_asset_ids | current_asset_ids
        
        for asset_id in all_asset_ids:
            symbol = self.asset_id_to_symbol.get(asset_id)
            if symbol is None:
                logger.warning(f"No symbol mapping for asset_id {asset_id}")
                continue
            
            # Get target weight
            target_weight = 0.0
            if len(target_weights) > 0:
                target_rows = target_weights[target_weights['asset_id'] == asset_id]
                if len(target_rows) > 0:
                    target_weight = target_rows['weight'].iloc[0]
            
            # Get current weight
            current_weight = current_positions.get(asset_id, 0.0)
            
            # Compute weight difference
            weight_diff = target_weight - current_weight
            
            # Skip if difference is negligible (< 0.5% of portfolio)
            if abs(weight_diff) < 0.005:
                continue
            
            # Get current price
            quote = self.broker.get_quote(symbol)
            price = quote['last']
            
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}")
                continue
            
            # Compute shares to trade
            value_diff = weight_diff * total_value
            shares = int(abs(value_diff) / price)
            
            if shares == 0:
                continue
            
            # Create order
            side = 'buy' if weight_diff > 0 else 'sell'
            orders.append({
                'symbol': symbol,
                'side': side,
                'quantity': shares,
                'order_type': 'market',
                'time_in_force': 'day',
                'asset_id': asset_id,
                'target_weight': target_weight,
                'current_weight': current_weight
            })
        
        # Check buying power constraint and scale buy orders if needed
        buying_power = self.broker.get_buying_power()
        safety_factor = self.config.get('buying_power_safety_factor', 0.98)  # Use 98% of buying power as safety margin
        
        # Calculate total buy notional
        total_buy_notional = 0.0
        buy_orders = [o for o in orders if o['side'] == 'buy']
        
        for order in buy_orders:
            quote = self.broker.get_quote(order['symbol'])
            price = quote['last']
            if price > 0:
                total_buy_notional += order['quantity'] * price
        
        # Scale down buy orders if they exceed buying power
        max_buy_notional = buying_power * safety_factor
        if total_buy_notional > max_buy_notional and total_buy_notional > 0:
            scale_factor = max_buy_notional / total_buy_notional
            logger.warning(
                f"Buy orders exceed buying power: ${total_buy_notional:,.2f} > ${max_buy_notional:,.2f}. "
                f"Scaling down buy orders by {scale_factor:.2%}"
            )
            
            # Scale down all buy orders
            for order in buy_orders:
                original_qty = order['quantity']
                order['quantity'] = max(1, int(original_qty * scale_factor))  # At least 1 share
                if order['quantity'] != original_qty:
                    logger.debug(
                        f"Scaled {order['symbol']}: {original_qty} -> {order['quantity']} shares"
                    )
        
        return orders
    
    def _log_daily_results(
        self,
        trading_date: date,
        predictions: Dict,
        target_weights: pd.DataFrame,
        orders: List[Dict],
        current_positions: Optional[Dict[int, float]] = None,
        regime_descriptor: str = "unknown",
        exposure_scale: float = 1.0
    ):
        """Log daily trading results to console and file."""
        logger.info(f"Daily results for {trading_date}:")
        logger.info(f"  Regime: {regime_descriptor}")
        logger.info(f"  Exposure scale: {exposure_scale:.2f}")
        logger.info(f"  Predictions: {len(predictions)} horizons")
        logger.info(f"  Target positions: {len(target_weights)} assets")
        logger.info(f"  Orders submitted: {len(orders)}")
        
        # Get exposure manager state if available
        exposure_state = {}
        if self.exposure_manager is not None:
            exposure_state = self.exposure_manager.get_state()
        
        # Get drawdown state if available
        drawdown_state = {}
        if self.drawdown_manager is not None:
            current_equity = self.broker.get_account_value()
            drawdown_state = {
                'current_drawdown': self.drawdown_manager.get_current_drawdown(current_equity),
                'is_throttled': self.drawdown_manager.is_throttled,
                'scale_factor': self.drawdown_manager.current_scale_factor,
            }
        
        # Save detailed log to file
        log_data = {
            'trading_date': trading_date.isoformat(),
            'timestamp': datetime.now().isoformat(),
            'account_value': self.broker.get_account_value(),
            'cash': self.broker.get_cash(),
            'buying_power': self.broker.get_buying_power(),
            'regime': {
                'descriptor': regime_descriptor,
                'exposure_scale': exposure_scale,
            },
            'exposure_state': exposure_state,
            'drawdown_state': drawdown_state,
            'predictions': {
                str(h): {
                    'count': len(p[0]) if isinstance(p, tuple) else len(p),
                    'mean_prediction': float(np.mean(p[0])) if isinstance(p, tuple) else float(np.mean(p)),
                }
                for h, p in predictions.items()
            },
            'current_positions': [
                {
                    'asset_id': asset_id,
                    'symbol': self.asset_id_to_symbol.get(asset_id, f"Asset_{asset_id}"),
                    'weight': weight
                }
                for asset_id, weight in (current_positions.items() if current_positions else {})
            ],
            'target_weights': target_weights.to_dict('records') if len(target_weights) > 0 else [],
            'orders': [
                {k: (v.isoformat() if isinstance(v, (date, datetime)) else v)
                 for k, v in order.items()}
                for order in orders
            ],
            'positions': self.broker.get_positions()
        }
        
        log_file = self.logs_dir / f"daily_log_{trading_date.isoformat()}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"  Log saved to {log_file}")
    
    def check_log_exists(self, trading_date: date) -> bool:
        """Check if a log already exists for this date (for idempotency)."""
        log_file = self.logs_dir / f"daily_log_{trading_date.isoformat()}.json"
        return log_file.exists()

