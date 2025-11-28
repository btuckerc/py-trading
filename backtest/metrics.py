"""Performance metrics for backtesting."""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import date


class PerformanceMetrics:
    """Computes performance metrics from equity curves."""
    
    @staticmethod
    def compute_metrics(equity_curve: pd.DataFrame) -> dict:
        """
        Compute comprehensive performance metrics.
        
        Args:
            equity_curve: DataFrame with columns: date, equity, returns
        
        Returns:
            Dictionary of metrics
        """
        if len(equity_curve) == 0:
            return {}
        
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve = equity_curve.sort_values('date')
        
        equity = equity_curve['equity'].values
        returns = equity_curve['returns'].values
        
        # Total return
        total_return = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        
        # Annualized return (CAGR)
        num_years = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days / 365.25
        if num_years > 0:
            cagr = ((equity[-1] / equity[0]) ** (1.0 / num_years)) - 1 if equity[0] > 0 else 0
        else:
            cagr = 0
        
        # Annualized volatility
        if len(returns) > 1:
            daily_vol = np.std(returns)
            annualized_vol = daily_vol * np.sqrt(252)  # Trading days
        else:
            annualized_vol = 0
        
        # Sharpe ratio
        if annualized_vol > 0:
            sharpe = cagr / annualized_vol
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        if max_drawdown > 0:
            calmar = cagr / max_drawdown
        else:
            calmar = 0
        
        # Hit rate
        positive_returns = (returns > 0).sum()
        hit_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        # Turnover (approximate from returns volatility)
        turnover = np.std(returns) * np.sqrt(252)  # Rough approximation
        
        # VaR and CVaR (5% and 1%)
        if len(returns) > 0:
            var_5 = np.percentile(returns, 5)
            var_1 = np.percentile(returns, 1)
            cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else 0
            cvar_1 = returns[returns <= var_1].mean() if len(returns[returns <= var_1]) > 0 else 0
        else:
            var_5 = var_1 = cvar_5 = cvar_1 = 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'hit_rate': hit_rate,
            'turnover': turnover,
            'var_5pct': var_5,
            'var_1pct': var_1,
            'cvar_5pct': cvar_5,
            'cvar_1pct': cvar_1,
            'num_trading_days': len(equity_curve)
        }
    
    @staticmethod
    def compute_regime_metrics(
        equity_curve: pd.DataFrame,
        regimes_df: pd.DataFrame
    ) -> dict:
        """
        Compute metrics conditioned on market regimes.
        
        Args:
            equity_curve: DataFrame with date, equity, returns
            regimes_df: DataFrame with date, regime_id
        
        Returns:
            Dictionary mapping regime_id -> metrics
        """
        equity_curve = equity_curve.copy()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        regimes_df = regimes_df.copy()
        regimes_df['date'] = pd.to_datetime(regimes_df['date'])
        
        # Merge
        merged = equity_curve.merge(regimes_df, on='date', how='left')
        
        regime_metrics = {}
        
        for regime_id, regime_data in merged.groupby('regime_id'):
            if len(regime_data) > 0:
                regime_equity = regime_data[['date', 'equity', 'returns']]
                regime_metrics[regime_id] = PerformanceMetrics.compute_metrics(regime_equity)
        
        return regime_metrics

