"""Live trading engine."""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import date, datetime, time
from data.asof_api import AsOfQueryAPI
from features.pipeline import FeaturePipeline
from portfolio.strategies import LongTopKStrategy
from portfolio.scores import ScoreConverter
from live.broker_base import BrokerAPI
from loguru import logger


class LiveEngine:
    """Live trading engine that runs daily inference and execution."""
    
    def __init__(
        self,
        api: AsOfQueryAPI,
        feature_pipeline: FeaturePipeline,
        model,  # Trained model
        strategy: LongTopKStrategy,
        broker: BrokerAPI,
        config: Optional[Dict] = None
    ):
        self.api = api
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.strategy = strategy
        self.broker = broker
        self.config = config or {}
    
    def run_daily_loop(self, trading_date: date):
        """
        Run daily trading loop.
        
        This should be called after market close on trading_date.
        """
        logger.info(f"Running daily loop for {trading_date}")
        
        # 1. Get latest data
        universe = self.api.get_universe_at_date(trading_date)
        logger.info(f"Universe size: {len(universe)}")
        
        # 2. Build features
        features_df = self.feature_pipeline.build_features_cross_sectional(
            trading_date,
            universe
        )
        
        if len(features_df) == 0:
            logger.warning("No features generated")
            return
        
        # 3. Run model inference
        predictions = self._predict(features_df)
        
        # 4. Convert predictions to scores
        scores_df = self._predictions_to_scores(predictions, features_df)
        
        # 5. Compute target positions
        current_positions = self._get_current_positions_dict()
        target_weights = self.strategy.compute_weights(
            scores_df,
            current_positions,
            trading_date
        )
        
        # 6. Generate orders to rebalance
        orders = self._generate_rebalance_orders(target_weights, current_positions)
        
        # 7. Submit orders
        for order in orders:
            order_id = self.broker.submit_order(
                symbol=order['symbol'],
                side=order['side'],
                quantity=order['quantity'],
                order_type=order.get('order_type', 'market'),
                time_in_force=order.get('time_in_force', 'day')
            )
            logger.info(f"Submitted order: {order_id} for {order['symbol']}")
        
        # 8. Log results
        self._log_daily_results(trading_date, predictions, target_weights, orders)
    
    def _predict(self, features_df: pd.DataFrame) -> Dict:
        """Run model inference."""
        # Extract feature columns (exclude asset_id, date)
        feature_cols = [c for c in features_df.columns if c not in ['asset_id', 'date']]
        X = features_df[feature_cols].values
        
        # Convert to tensor and predict
        import torch
        X_tensor = torch.FloatTensor(X)
        
        if hasattr(self.model, 'predict_mc'):
            # MC dropout for uncertainty
            predictions = self.model.predict_mc(X_tensor, n_samples=50)
        else:
            # Standard prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = {h: pred.cpu().numpy() for h, pred in outputs.items()}
        
        return predictions
    
    def _predictions_to_scores(
        self,
        predictions: Dict,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert predictions to scores."""
        scores = []
        
        for idx, row in features_df.iterrows():
            asset_id = row['asset_id']
            
            # Combine multi-horizon predictions
            pred_dict = {}
            for horizon, pred in predictions.items():
                if isinstance(pred, tuple):
                    mu, sigma = pred
                    pred_dict[horizon] = {'mu': mu[idx], 'sigma': sigma[idx]}
                else:
                    pred_dict[horizon] = {'mu': pred[idx], 'sigma': 0.1}  # Default uncertainty
            
            # Combine scores
            score = ScoreConverter.combine_multi_horizon_scores(pred_dict)
            
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
        """Get current positions as dict."""
        positions = self.broker.get_positions()
        
        # Convert to asset_id -> weight
        # This would need symbol -> asset_id mapping
        return {}
    
    def _generate_rebalance_orders(
        self,
        target_weights: pd.DataFrame,
        current_positions: Dict[int, float]
    ) -> List[Dict]:
        """Generate orders to rebalance to target weights."""
        orders = []
        
        # This is simplified - would need:
        # 1. Map asset_id to symbol
        # 2. Compute current vs target weights
        # 3. Generate buy/sell orders
        
        return orders
    
    def _log_daily_results(
        self,
        trading_date: date,
        predictions: Dict,
        target_weights: pd.DataFrame,
        orders: List[Dict]
    ):
        """Log daily trading results."""
        logger.info(f"Daily results for {trading_date}:")
        logger.info(f"  Predictions: {len(predictions)} horizons")
        logger.info(f"  Target positions: {len(target_weights)} assets")
        logger.info(f"  Orders submitted: {len(orders)}")

