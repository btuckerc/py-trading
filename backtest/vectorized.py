"""Vectorized backtesting engine."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
from datetime import date
from portfolio.costs import TransactionCostModel
from portfolio.risk import DrawdownManager, ExposureManager
from loguru import logger


class VectorizedBacktester:
    """
    Fast vectorized backtester operating on panel arrays.
    
    Inputs:
    - Price panel: (date, asset_id) -> price
    - Predictions panel: (date, asset_id) -> predicted return/score
    - Target weights panel: (date, asset_id) -> target weight
    - Transaction cost model
    - Drawdown manager (optional)
    - Exposure manager (optional, for regime/vol scaling)
    - Regimes DataFrame (optional, for regime-aware scaling)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: Optional[TransactionCostModel] = None,
        drawdown_manager: Optional[DrawdownManager] = None,
        exposure_manager: Optional[ExposureManager] = None,
        use_drawdown_throttle: bool = False,
        use_regime_scaling: bool = False
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.drawdown_manager = drawdown_manager
        self.exposure_manager = exposure_manager
        self.use_drawdown_throttle = use_drawdown_throttle
        self.use_regime_scaling = use_regime_scaling
        
        # Initialize drawdown manager with default settings if throttle enabled but no manager provided
        if self.use_drawdown_throttle and self.drawdown_manager is None:
            self.drawdown_manager = DrawdownManager()
        
        # Initialize exposure manager if regime scaling enabled but no manager provided
        if self.use_regime_scaling and self.exposure_manager is None:
            self.exposure_manager = ExposureManager()
    
    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        execution_lag: int = 1,
        regimes_df: Optional[pd.DataFrame] = None,
        market_vol_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.
        
        Args:
            prices_df: DataFrame with columns: date, asset_id, adj_close
            target_weights_df: DataFrame with columns: date, asset_id, weight
            execution_lag: Days between signal and execution (default 1: signal at close t, execute at open t+1)
            regimes_df: Optional DataFrame with columns: date, regime_id, regime_descriptor
                       Used for regime-aware exposure scaling
            market_vol_df: Optional DataFrame with columns: date, realized_vol_20d
                          Used for VIX-style volatility scaling
        
        Returns:
            DataFrame with columns: date, equity, returns, positions, trades, costs, regime, exposure_scale
        """
        # Build regime lookup if provided
        regime_lookup = {}
        if regimes_df is not None and self.use_regime_scaling:
            regimes_df = regimes_df.copy()
            regimes_df['date'] = pd.to_datetime(regimes_df['date']).dt.date
            regime_lookup = dict(zip(regimes_df['date'], regimes_df['regime_descriptor']))
        
        # Build volatility lookup if provided
        vol_lookup = {}
        if market_vol_df is not None and self.exposure_manager is not None:
            market_vol_df = market_vol_df.copy()
            market_vol_df['date'] = pd.to_datetime(market_vol_df['date']).dt.date
            vol_lookup = dict(zip(market_vol_df['date'], market_vol_df['realized_vol_20d']))
        # Align indices
        prices_df = prices_df.set_index(['date', 'asset_id']).sort_index()
        target_weights_df = target_weights_df.set_index(['date', 'asset_id']).sort_index()
        
        # Get all dates
        all_dates = sorted(set(prices_df.index.get_level_values('date')))
        
        # Initialize tracking
        equity_curve = []
        positions_history = []
        trades_history = []
        
        current_capital = self.initial_capital
        current_positions = {}  # asset_id -> shares
        previous_weights = {}  # asset_id -> weight
        previous_portfolio_value = self.initial_capital  # Track previous day's ending value
        
        for i, current_date in enumerate(all_dates):
            # Get prices on this date
            if current_date in prices_df.index.get_level_values('date'):
                date_prices = prices_df.loc[current_date]
            else:
                continue
            
            if len(date_prices) == 0:
                continue
            
            # Get target weights (with execution lag)
            execution_date_idx = i + execution_lag
            if execution_date_idx < len(all_dates):
                execution_date = all_dates[execution_date_idx]
                if execution_date in target_weights_df.index.get_level_values('date'):
                    target_weights = target_weights_df.loc[execution_date]
                    # Ensure it's a DataFrame, not a Series
                    if isinstance(target_weights, pd.Series):
                        target_weights = target_weights.to_frame().T
                        if 'asset_id' in target_weights.columns:
                            target_weights = target_weights.set_index('asset_id')
                else:
                    target_weights = pd.DataFrame(columns=['weight'])
            else:
                target_weights = pd.DataFrame(columns=['weight'])
            
            # Compute current portfolio value
            portfolio_value = current_capital
            for asset_id, shares in current_positions.items():
                if asset_id in date_prices.index:
                    portfolio_value += shares * date_prices.loc[asset_id, 'adj_close']
            
            # Track exposure scaling for logging
            current_regime = regime_lookup.get(current_date, 'unknown') if regime_lookup else 'unknown'
            exposure_scale = 1.0
            
            # Apply regime-aware exposure scaling if enabled
            if self.use_regime_scaling and self.exposure_manager is not None:
                # Update regime
                self.exposure_manager.update_regime(current_regime)
                
                # Update volatility if available
                if current_date in vol_lookup:
                    self.exposure_manager.update_volatility(vol_lookup[current_date])
            
            # Apply drawdown throttle if enabled
            drawdown_scale = 1.0
            if self.use_drawdown_throttle and self.drawdown_manager is not None:
                # Get the scale factor based on current drawdown
                drawdown_scale = self.drawdown_manager.update(portfolio_value)
            
            # Get combined exposure scale
            if self.exposure_manager is not None:
                exposure_scale = self.exposure_manager.get_combined_scale(drawdown_scale)
            elif drawdown_scale < 1.0:
                exposure_scale = drawdown_scale
            
            # Scale target weights by combined exposure factor
            if exposure_scale < 1.0 and len(target_weights) > 0:
                if isinstance(target_weights, pd.DataFrame) and 'weight' in target_weights.columns:
                    target_weights = target_weights.copy()
                    target_weights['weight'] = target_weights['weight'] * exposure_scale
            
            # Update positions to target weights
            target_shares = {}
            trades = {}
            
            if isinstance(date_prices, pd.Series):
                # Single asset case
                asset_id = date_prices.name if hasattr(date_prices, 'name') else date_prices.index[0]
                current_price = date_prices['adj_close'] if isinstance(date_prices, pd.Series) else date_prices.iloc[0]['adj_close']
                current_shares = current_positions.get(asset_id, 0)
                current_value = current_shares * current_price
                
                # Target weight
                if len(target_weights) > 0 and asset_id in target_weights.index:
                    target_weight = target_weights.loc[asset_id, 'weight']
                else:
                    target_weight = 0.0
                
                target_value = portfolio_value * target_weight
                target_shares_new = target_value / current_price if current_price > 0 else 0
                target_shares_new = int(target_shares_new)
                target_shares[asset_id] = target_shares_new
                
                trade_shares = target_shares_new - current_shares
                if trade_shares != 0:
                    trades[asset_id] = trade_shares
            else:
                # Multiple assets
                for asset_id in date_prices.index:
                    if isinstance(date_prices.index, pd.MultiIndex):
                        # Handle MultiIndex case
                        asset_id = asset_id[1] if isinstance(asset_id, tuple) else asset_id
                    
                    current_price = date_prices.loc[asset_id, 'adj_close'] if hasattr(date_prices, 'loc') else date_prices[asset_id]['adj_close']
                    current_shares = current_positions.get(asset_id, 0)
                    current_value = current_shares * current_price
                    
                    # Target weight
                    if len(target_weights) > 0:
                        if asset_id in target_weights.index:
                            target_weight = target_weights.loc[asset_id, 'weight']
                        elif isinstance(target_weights, pd.DataFrame) and 'asset_id' in target_weights.columns:
                            target_weight_rows = target_weights[target_weights['asset_id'] == asset_id]
                            if len(target_weight_rows) > 0:
                                target_weight = target_weight_rows.iloc[0]['weight']
                            else:
                                target_weight = 0.0
                        else:
                            target_weight = 0.0
                    else:
                        target_weight = 0.0
                    
                    target_value = portfolio_value * target_weight
                    target_shares_new = target_value / current_price if current_price > 0 else 0
                    target_shares_new = int(target_shares_new)
                    target_shares[asset_id] = target_shares_new
                    
                    trade_shares = target_shares_new - current_shares
                    if trade_shares != 0:
                        trades[asset_id] = trade_shares
            
            # Execute trades and update cash
            # First, sell existing positions that need to be reduced/closed
            for asset_id, trade_shares in trades.items():
                if trade_shares < 0:  # Selling
                    # Get price - handle both Series and DataFrame cases
                    if isinstance(date_prices, pd.Series):
                        price = date_prices['adj_close'] if 'adj_close' in date_prices.index else date_prices.iloc[0]
                    else:
                        price = date_prices.loc[asset_id, 'adj_close'] if asset_id in date_prices.index else date_prices.iloc[0]['adj_close']
                    
                    sell_value = abs(trade_shares) * price
                    price_dict = {'adj_close': price}
                    cost = self.cost_model.compute_cost(sell_value, price_dict)
                    current_capital += sell_value - cost  # Add proceeds minus costs
            
            # Then, buy new positions
            for asset_id, trade_shares in trades.items():
                if trade_shares > 0:  # Buying
                    # Get price - handle both Series and DataFrame cases
                    if isinstance(date_prices, pd.Series):
                        price = date_prices['adj_close'] if 'adj_close' in date_prices.index else date_prices.iloc[0]
                    else:
                        price = date_prices.loc[asset_id, 'adj_close'] if asset_id in date_prices.index else date_prices.iloc[0]['adj_close']
                    
                    buy_value = trade_shares * price
                    price_dict = {'adj_close': price}
                    cost = self.cost_model.compute_cost(buy_value, price_dict)
                    current_capital -= (buy_value + cost)  # Subtract cost plus transaction costs
            
            # Update positions
            current_positions = target_shares.copy()
            
            # Compute new portfolio value (cash + positions value)
            new_portfolio_value = current_capital
            for asset_id, shares in current_positions.items():
                if asset_id in date_prices.index:
                    new_portfolio_value += shares * date_prices.loc[asset_id, 'adj_close']
            
            # Calculate total costs for reporting
            total_costs = 0.0
            for asset_id, trade_shares in trades.items():
                # Get price - handle both Series and DataFrame cases
                if isinstance(date_prices, pd.Series):
                    price = date_prices['adj_close'] if 'adj_close' in date_prices.index else date_prices.iloc[0]
                    price_dict = {'adj_close': price}
                else:
                    price = date_prices.loc[asset_id, 'adj_close'] if asset_id in date_prices.index else date_prices.iloc[0]['adj_close']
                    price_dict = {'adj_close': price}
                
                trade_value = abs(trade_shares) * price
                cost = self.cost_model.compute_cost(trade_value, price_dict)
                total_costs += cost
            
            # Record results
            # Calculate return from previous day's ending value to this day's ending value
            daily_return = (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value if previous_portfolio_value > 0 else 0
            
            equity_curve.append({
                'date': current_date,
                'equity': new_portfolio_value,
                'returns': daily_return,
                'costs': total_costs,
                'cash': current_capital,
                'regime': current_regime,
                'exposure_scale': exposure_scale
            })
            
            # Update previous portfolio value for next iteration
            previous_portfolio_value = new_portfolio_value
            
            positions_history.append({
                'date': current_date,
                'positions': current_positions.copy()
            })
            
            if len(trades) > 0:
                trades_history.append({
                    'date': current_date,
                    'trades': trades.copy()
                })
        
        equity_df = pd.DataFrame(equity_curve)
        return equity_df

