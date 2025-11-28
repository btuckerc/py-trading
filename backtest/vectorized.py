"""Vectorized backtesting engine."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
from datetime import date
from portfolio.costs import TransactionCostModel


class VectorizedBacktester:
    """
    Fast vectorized backtester operating on panel arrays.
    
    Inputs:
    - Price panel: (date, asset_id) -> price
    - Predictions panel: (date, asset_id) -> predicted return/score
    - Target weights panel: (date, asset_id) -> target weight
    - Transaction cost model
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: Optional[TransactionCostModel] = None
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
    
    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        execution_lag: int = 1
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.
        
        Args:
            prices_df: DataFrame with columns: date, asset_id, adj_close
            target_weights_df: DataFrame with columns: date, asset_id, weight
            execution_lag: Days between signal and execution (default 1: signal at close t, execute at open t+1)
        
        Returns:
            DataFrame with columns: date, equity, returns, positions, trades, costs
        """
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
                target_weights = target_weights_df.loc[execution_date] if execution_date in target_weights_df.index.get_level_values('date') else pd.Series()
            else:
                target_weights = pd.Series()
            
            # Compute current portfolio value
            portfolio_value = current_capital
            for asset_id, shares in current_positions.items():
                if asset_id in date_prices.index:
                    portfolio_value += shares * date_prices.loc[asset_id, 'adj_close']
            
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
                        target_weight_rows = target_weights[target_weights['asset_id'] == asset_id]
                        if len(target_weight_rows) > 0:
                            target_weight = target_weight_rows.iloc[0]['weight']
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
            
            # Apply transaction costs
            total_costs = 0.0
            for asset_id, trade_shares in trades.items():
                trade_value = abs(trade_shares) * date_prices.loc[asset_id, 'adj_close']
                cost = self.cost_model.compute_cost(trade_value, date_prices.loc[asset_id].to_dict())
                total_costs += cost
            
            # Update capital and positions
            current_capital -= total_costs
            current_positions = target_shares.copy()
            
            # Compute new portfolio value
            new_portfolio_value = current_capital
            for asset_id, shares in current_positions.items():
                if asset_id in date_prices.index:
                    new_portfolio_value += shares * date_prices.loc[asset_id, 'adj_close']
            
            # Record results
            equity_curve.append({
                'date': current_date,
                'equity': new_portfolio_value,
                'returns': (new_portfolio_value - portfolio_value) / portfolio_value if portfolio_value > 0 else 0,
                'costs': total_costs,
                'cash': current_capital
            })
            
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

