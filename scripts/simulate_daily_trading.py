#!/usr/bin/env python3
"""
Simulate daily trading decisions and compare performance to SPY.

This script simulates what the live trading loop would have done for each
trading day in a date range, then calculates returns and compares to SPY.

IMPORTANT: Point-in-time correctness
- Model is trained ONLY on data before the simulation period
- Each day's features use ONLY data available as of that day's close
- No future data leakage

Usage:
    # Last 2 weeks
    python scripts/simulate_daily_trading.py --start-date 2025-11-12 --end-date 2025-11-26

    # Custom range with different top-k
    python scripts/simulate_daily_trading.py --start-date 2025-06-01 --end-date 2025-06-30 --top-k 10

    # Output to JSON for further analysis
    python scripts/simulate_daily_trading.py --start-date 2025-11-01 --end-date 2025-11-26 --output-json

    # Verbose mode shows daily stock returns
    python scripts/simulate_daily_trading.py --start-date 2025-11-12 --end-date 2025-11-26 --verbose
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import get_config, RetrainingConfig
from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from data.universe import TradingCalendar
from features.pipeline import FeaturePipeline
from models.tabular import XGBoostModel, LightGBMModel
from models.training import WalkForwardRetrainer
from labels.returns import ReturnLabelGenerator
from portfolio.strategies import LongTopKStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate daily trading and compare to SPY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--start-date", required=True,
        help="Start date for simulation (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", required=True,
        help="End date for simulation (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top stocks to hold (default: 5)"
    )
    parser.add_argument(
        "--train-days", type=int, default=252,
        help="Number of trading days to use for training (default: 252, ~1 year)"
    )
    parser.add_argument(
        "--retrain-frequency", type=int, default=0,
        help="Retrain model every N days (0 = train once at start, default: 0). "
             "Use --use-policy to use config-driven retraining instead."
    )
    parser.add_argument(
        "--use-policy", action="store_true",
        help="Use retraining policy from config (cadence, time-decay weighting)"
    )
    parser.add_argument(
        "--retrain-cadence", type=int, default=None,
        help="Override retraining cadence from config (days)"
    )
    parser.add_argument(
        "--time-decay-lambda", type=float, default=None,
        help="Override time-decay lambda. Higher = more emphasis on recent data"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000,
        help="Initial portfolio value (default: 100000)"
    )
    parser.add_argument(
        "--output-json", action="store_true",
        help="Output results as JSON (for programmatic use)"
    )
    parser.add_argument(
        "--save-report", type=str, default=None,
        help="Save report to specified file path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed daily returns for each stock"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output (just final summary)"
    )
    return parser.parse_args()


class DailyTradingSimulator:
    """
    Simulates daily trading decisions with strict point-in-time correctness.
    
    Supports two retraining modes:
    1. Simple: retrain every N days (--retrain-frequency)
    2. Policy-driven: use config-based cadence, time-decay weighting (--use-policy)
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        config,
        top_k: int = 5,
        train_days: int = 252,
        initial_capital: float = 100000,
        use_policy: bool = False,
        retraining_config: RetrainingConfig = None
    ):
        self.storage = storage
        self.config = config
        self.api = AsOfQueryAPI(storage)
        self.calendar = TradingCalendar()
        self.top_k = top_k
        self.train_days = train_days
        self.initial_capital = initial_capital
        self.use_policy = use_policy
        self.retraining_config = retraining_config
        
        # Will be set during simulation
        self.model = None
        self.feature_pipeline = None
        self.universe = None
        self.symbol_map = None  # asset_id -> symbol
        self.retrainer = None  # WalkForwardRetrainer for policy mode
        self.retraining_history = []  # Track when retraining occurred
        
    def _get_universe(self, as_of_date: date) -> set:
        """Get universe as of a specific date (point-in-time correct)."""
        try:
            universe = self.api.get_universe_at_date(as_of_date, index_name="SP500")
            if len(universe) > 0:
                return universe
        except Exception:
            pass
        
        # Fallback: all assets with data
        bars = self.api.get_bars_asof(as_of_date, lookback_days=5)
        return set(bars['asset_id'].unique())
    
    def _build_symbol_map(self) -> Dict[int, str]:
        """Build asset_id to symbol mapping."""
        assets = self.storage.query("SELECT asset_id, symbol FROM assets")
        return dict(zip(assets['asset_id'], assets['symbol']))
    
    def _ensure_benchmark_data(self, benchmark_symbols: List[str], start_date: date, end_date: date):
        """
        Ensure benchmark symbols exist in the database with price data.
        
        This handles the case where a fresh Docker container doesn't have
        benchmark ETFs (SPY, DIA, QQQ) in the assets table because they're
        not part of the S&P 500 constituents CSV.
        """
        # Check which benchmarks are missing
        missing_symbols = []
        for symbol in benchmark_symbols:
            result = self.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{symbol}'")
            if len(result) == 0:
                missing_symbols.append(symbol)
            else:
                # Check if we have price data for the period
                asset_id = result.iloc[0]['asset_id']
                bars = self.storage.query(f"""
                    SELECT COUNT(*) as cnt FROM bars_daily 
                    WHERE asset_id = {asset_id} 
                    AND date >= '{start_date}' AND date <= '{end_date}'
                """)
                if bars.iloc[0]['cnt'] == 0:
                    missing_symbols.append(symbol)
        
        if not missing_symbols:
            return
        
        logger.info(f"Fetching missing benchmark data for: {missing_symbols}")
        
        try:
            # Use the data maintenance system to fetch benchmark data
            from data.maintenance import DataMaintenanceManager
            
            maintenance = DataMaintenanceManager(self.storage)
            
            # Fetch data for missing benchmarks
            # Use a buffer before start_date for training data
            fetch_start = start_date - timedelta(days=400)  # ~1.5 years buffer for training
            
            maintenance.ensure_coverage(
                mode="date-range",
                target_start=fetch_start,
                target_end=end_date,
                symbols=missing_symbols,
                auto_fetch=True,
                bootstrap_universe=False  # Don't rebuild universe, just fetch these symbols
            )
            
            logger.info(f"Successfully fetched benchmark data for {missing_symbols}")
            
        except Exception as e:
            logger.warning(f"Could not fetch benchmark data: {e}. Benchmark comparisons may be unavailable.")
    
    def _get_price_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Get price data for the simulation period plus buffer."""
        # Get all bars in the range
        query = f"""
            SELECT b.asset_id, b.date, b.close, a.symbol
            FROM bars_daily b
            JOIN assets a ON b.asset_id = a.asset_id
            WHERE b.date >= '{start_date}' AND b.date <= '{end_date}'
            ORDER BY b.date, a.symbol
        """
        return self.storage.query(query)
    
    def _train_model(self, train_end_date: date, universe: set) -> Tuple[XGBoostModel, FeaturePipeline]:
        """
        Train model using only data up to train_end_date.
        This ensures no future data leakage.
        
        If use_policy is True, uses time-decay weighting from retraining_config.
        """
        # Calculate training period
        if self.use_policy and self.retraining_config:
            # Use window from config
            train_start = self.retraining_config.get_window_start(train_end_date)
        else:
            # Use fixed train_days
            train_start = train_end_date - timedelta(days=self.train_days * 2)
        
        trading_days = self.calendar.get_trading_days(train_start, train_end_date)
        trading_days = [d.date() if hasattr(d, 'date') else d for d in trading_days]
        
        if not self.use_policy and len(trading_days) < self.train_days:
            raise ValueError(f"Not enough trading days for training. Need {self.train_days}, got {len(trading_days)}")
        
        # Use last train_days trading days (or all if using policy)
        if self.use_policy:
            train_dates = trading_days
        else:
            train_dates = trading_days[-self.train_days:]
        
        train_start = train_dates[0]
        train_end = train_dates[-1]
        
        logger.info(f"Training model on {train_start} to {train_end} ({len(train_dates)} days)")
        if self.use_policy and self.retraining_config and self.retraining_config.time_decay.enabled:
            logger.info(f"Time-decay weighting enabled (lambda={self.retraining_config.time_decay.lambda_})")
        
        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline(self.api)
        
        # Get training data - sample dates for efficiency
        sample_size = min(20, len(train_dates))
        sample_indices = np.linspace(0, len(train_dates) - 1, sample_size, dtype=int)
        sample_dates = [train_dates[i] for i in sample_indices]
        
        # Build features for sampled dates
        features_list = []
        labels_list = []
        date_list = []  # Track dates for time-decay weighting
        
        for sample_date in sample_dates:
            try:
                # Get features as of this date
                features_df = feature_pipeline.build_features_cross_sectional(
                    as_of_date=sample_date,
                    universe=set(universe),
                    lookback_days=252
                )
                
                if features_df.empty:
                    continue
                
                # Get forward returns (20-day) for labels
                # This is the ONLY place we look forward, and it's for training labels
                future_date = sample_date + timedelta(days=30)  # ~20 trading days
                future_bars = self.api.get_bars_asof(future_date, universe=universe, lookback_days=5)
                current_bars = self.api.get_bars_asof(sample_date, universe=universe, lookback_days=5)
                
                if future_bars.empty or current_bars.empty:
                    continue
                
                # Calculate returns
                current_prices = current_bars.groupby('asset_id')['close'].last()
                future_prices = future_bars.groupby('asset_id')['close'].last()
                
                common_assets = current_prices.index.intersection(future_prices.index)
                returns = (future_prices[common_assets] - current_prices[common_assets]) / current_prices[common_assets]
                
                # Merge with features
                features_df = features_df[features_df['asset_id'].isin(common_assets)]
                features_df = features_df.set_index('asset_id')
                features_df['forward_return'] = returns
                features_df = features_df.dropna(subset=['forward_return'])
                
                if not features_df.empty:
                    features_list.append(features_df.reset_index())
                    # Track dates for each sample
                    date_list.extend([sample_date] * len(features_df))
                    
            except Exception as e:
                logger.debug(f"Skipping {sample_date}: {e}")
                continue
        
        if not features_list:
            raise ValueError("Could not generate any training data")
        
        # Combine all training data
        train_df = pd.concat(features_list, ignore_index=True)
        
        # Prepare X and y
        feature_cols = [c for c in train_df.columns if c not in ['asset_id', 'date', 'forward_return', 'symbol']]
        X = train_df[feature_cols].values
        y = train_df['forward_return'].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Compute time-decay sample weights if using policy
        sample_weights = None
        if self.use_policy and self.retraining_config and self.retraining_config.time_decay.enabled:
            sample_weights = self.retraining_config.compute_sample_weights(date_list, train_end_date)
            sample_weights = np.array(sample_weights)
            logger.info(f"Sample weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        
        logger.info(f"Training on {len(train_df)} samples with {len(feature_cols)} features")
        
        # Train model with sample weights
        model = XGBoostModel(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y, sample_weight=sample_weights)
        
        # Store feature columns for prediction
        model.feature_columns = feature_cols
        
        return model, feature_pipeline
    
    def _get_daily_predictions(
        self,
        trading_date: date,
        model: XGBoostModel,
        feature_pipeline: FeaturePipeline,
        universe: set
    ) -> Dict[int, float]:
        """
        Get model predictions for a specific date.
        Uses only data available as of that date.
        """
        # Build features as of trading_date
        features_df = feature_pipeline.build_features_cross_sectional(
            as_of_date=trading_date,
            universe=set(universe),
            lookback_days=252
        )
        
        if features_df.empty:
            return {}
        
        # Prepare features
        feature_cols = model.feature_columns
        missing_cols = set(feature_cols) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0
        
        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Return as dict: asset_id -> predicted_return
        return dict(zip(features_df['asset_id'], predictions))
    
    def _select_portfolio(self, predictions: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Select top-k stocks and assign equal weights.
        Returns list of (asset_id, weight) tuples.
        """
        if not predictions:
            return []
        
        # Sort by predicted return (descending)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_k = sorted_preds[:self.top_k]
        
        # Equal weight
        weight = 1.0 / len(top_k) if top_k else 0
        
        return [(asset_id, weight) for asset_id, _ in top_k]
    
    def simulate(
        self,
        start_date: date,
        end_date: date,
        retrain_frequency: int = 0,
        verbose: bool = False
    ) -> dict:
        """
        Run the full simulation.
        
        Args:
            start_date: First trading date of simulation
            end_date: Last trading date of simulation
            retrain_frequency: Retrain model every N days (0 = train once)
            verbose: Print detailed progress
            
        Returns:
            Dictionary with simulation results
        """
        # Get trading days
        all_trading_days = self.calendar.get_trading_days(
            start_date - timedelta(days=10),  # Buffer
            end_date + timedelta(days=5)
        )
        all_trading_days = [d.date() if hasattr(d, 'date') else d for d in all_trading_days]
        
        # Filter to simulation period
        trading_days = [d for d in all_trading_days if start_date <= d <= end_date]
        
        if not trading_days:
            raise ValueError(f"No trading days found between {start_date} and {end_date}")
        
        logger.info(f"Simulating {len(trading_days)} trading days: {trading_days[0]} to {trading_days[-1]}")
        
        # Build symbol map
        self.symbol_map = self._build_symbol_map()
        
        # Get universe (use first day's universe for training)
        self.universe = self._get_universe(trading_days[0])
        logger.info(f"Universe size: {len(self.universe)} assets")
        
        # Train initial model (using data BEFORE simulation starts)
        train_end = trading_days[0] - timedelta(days=1)
        self.model, self.feature_pipeline = self._train_model(train_end, self.universe)
        
        # Get all price data for the period
        price_data = self._get_price_data(
            start_date - timedelta(days=5),
            end_date + timedelta(days=5)
        )
        price_data['date'] = pd.to_datetime(price_data['date']).dt.date
        
        # Build price lookup: (date, asset_id) -> close
        price_lookup = {}
        for _, row in price_data.iterrows():
            price_lookup[(row['date'], row['asset_id'])] = row['close']
        
        # Get benchmark symbols from config (default to SPY, DIA, QQQ)
        from configs.loader import get_config
        config = get_config()
        benchmark_config = getattr(config, 'benchmarks', {})
        benchmark_definitions = benchmark_config.get('definitions', {})
        default_benchmarks = benchmark_config.get('default', ['sp500'])
        
        # Map benchmark names to tickers
        benchmark_symbols = []
        benchmark_names = {}
        for bench_name in default_benchmarks:
            if bench_name in benchmark_definitions:
                ticker = benchmark_definitions[bench_name]['ticker']
                name = benchmark_definitions[bench_name]['name']
                benchmark_symbols.append(ticker)
                benchmark_names[ticker] = name
        
        # Fallback to SPY if no benchmarks configured
        if not benchmark_symbols:
            benchmark_symbols = ['SPY']
            benchmark_names['SPY'] = 'S&P 500'
        
        # Ensure benchmark symbols exist in the database (auto-fetch if missing)
        self._ensure_benchmark_data(benchmark_symbols, start_date, end_date)
        
        # Get asset_ids for all benchmarks
        benchmark_asset_ids = {}
        for symbol in benchmark_symbols:
            result = self.storage.query(f"SELECT asset_id FROM assets WHERE symbol = '{symbol}'")
            if len(result) > 0:
                benchmark_asset_ids[symbol] = result.iloc[0]['asset_id']
            else:
                logger.warning(f"Benchmark {symbol} not found in assets table after fetch attempt")
        
        # Backward compatibility: keep SPY asset_id for spy_return
        spy_asset_id = benchmark_asset_ids.get('SPY', None)
        
        if not benchmark_asset_ids:
            logger.warning("No benchmark data available - benchmark comparisons will be skipped")
        
        # Simulation state
        portfolio_value = self.initial_capital
        daily_results = []
        last_train_day = 0
        
        # Run simulation
        for i, trading_date in enumerate(trading_days[:-1]):  # Skip last day (no next-day return)
            next_date = trading_days[i + 1]
            
            # Check if we need to retrain
            should_retrain = False
            retrain_reason = ""
            
            if self.use_policy and self.retraining_config:
                # Policy-driven retraining
                if self.model is None:
                    should_retrain = True
                    retrain_reason = "initial training"
                else:
                    last_train_date = trading_days[last_train_day] if last_train_day >= 0 else trading_days[0]
                    should_retrain, retrain_reason = self.retraining_config.should_retrain(
                        last_train_date, trading_date
                    )
            elif retrain_frequency > 0 and i > 0 and (i - last_train_day) >= retrain_frequency:
                # Simple frequency-based retraining
                should_retrain = True
                retrain_reason = f"frequency ({retrain_frequency} days)"
            
            if should_retrain:
                logger.info(f"Retraining model at day {i} ({trading_date}): {retrain_reason}")
                train_end = trading_date - timedelta(days=1)
                self.model, self.feature_pipeline = self._train_model(train_end, self.universe)
                last_train_day = i
                self.retraining_history.append({
                    'date': str(trading_date),
                    'day_index': i,
                    'reason': retrain_reason
                })
            
            # Get predictions for this date
            try:
                predictions = self._get_daily_predictions(
                    trading_date, self.model, self.feature_pipeline, self.universe
                )
            except Exception as e:
                logger.warning(f"Failed to get predictions for {trading_date}: {e}")
                predictions = {}
            
            # Select portfolio
            portfolio = self._select_portfolio(predictions)
            
            # Calculate returns
            day_return = 0
            position_returns = []
            
            for asset_id, weight in portfolio:
                price_today = price_lookup.get((trading_date, asset_id))
                price_tomorrow = price_lookup.get((next_date, asset_id))
                
                if price_today and price_tomorrow:
                    stock_return = (price_tomorrow - price_today) / price_today
                    weighted_return = weight * stock_return
                    day_return += weighted_return
                    
                    symbol = self.symbol_map.get(asset_id, f"ID:{asset_id}")
                    position_returns.append({
                        "symbol": symbol,
                        "weight": weight,
                        "return": stock_return
                    })
            
            # Update portfolio value
            portfolio_value *= (1 + day_return)
            
            # Get benchmark returns for comparison
            benchmark_returns = {}
            spy_return = None  # Backward compatibility
            
            for symbol, asset_id in benchmark_asset_ids.items():
                today_price = price_lookup.get((trading_date, asset_id))
                tomorrow_price = price_lookup.get((next_date, asset_id))
                if today_price and tomorrow_price:
                    ret = (tomorrow_price - today_price) / today_price
                    benchmark_returns[symbol] = ret
                    # Keep SPY return for backward compatibility
                    if symbol == 'SPY':
                        spy_return = ret
            
            # Record results
            daily_results.append({
                "date": str(trading_date),
                "next_date": str(next_date),
                "positions": position_returns,
                "portfolio_return": day_return,
                "portfolio_value": portfolio_value,
                "spy_return": spy_return,  # Backward compatibility
                "benchmark_returns": benchmark_returns,  # New multi-benchmark support
                "cumulative_return": (portfolio_value - self.initial_capital) / self.initial_capital
            })
            
            if verbose:
                positions_str = ", ".join([
                    f"{p['symbol']}({p['weight']*100:.0f}%):{p['return']*100:+.2f}%"
                    for p in position_returns
                ])
                logger.info(f"{trading_date}: {positions_str} -> {day_return*100:+.2f}%")
        
        # Calculate summary statistics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate benchmark total returns
        # Use the actual dates we have data for, not just trading_days[0] and trading_days[-1]
        benchmark_total_returns = {}
        spy_total_return = None  # Backward compatibility
        
        for symbol, asset_id in benchmark_asset_ids.items():
            # Find first and last dates with price data for this benchmark
            start_price = None
            end_price = None
            
            # Search forward from start for first available price
            for d in trading_days:
                price = price_lookup.get((d, asset_id))
                if price is not None:
                    start_price = price
                    break
            
            # Search backward from end for last available price
            for d in reversed(trading_days):
                price = price_lookup.get((d, asset_id))
                if price is not None:
                    end_price = price
                    break
            
            if start_price and end_price:
                bench_return = (end_price - start_price) / start_price
                benchmark_total_returns[symbol] = bench_return
                if symbol == 'SPY':
                    spy_total_return = bench_return
            else:
                logger.debug(f"No price data found for benchmark {symbol} in the simulation period")
        
        # Calculate additional metrics
        daily_returns = [r["portfolio_return"] for r in daily_results]
        spy_daily_returns = [r["spy_return"] for r in daily_results if r["spy_return"] is not None]
        
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod([1 + r for r in daily_returns])
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        wins = sum(1 for r in daily_returns if r > 0)
        win_rate = wins / len(daily_returns) if daily_returns else 0
        
        # Calculate alpha vs primary benchmark (SPY by default)
        primary_benchmark = benchmark_config.get('primary', 'sp500')
        primary_ticker = benchmark_definitions.get(primary_benchmark, {}).get('ticker', 'SPY')
        primary_return = benchmark_total_returns.get(primary_ticker, spy_total_return)
        alpha_pct = (total_return - primary_return) * 100 if primary_return else None
        
        # Build retraining info
        retraining_info = {
            "mode": "policy" if self.use_policy else "frequency",
            "num_retrains": len(self.retraining_history),
            "history": self.retraining_history
        }
        if self.use_policy and self.retraining_config:
            retraining_info["config"] = {
                "cadence_days": self.retraining_config.cadence_days,
                "window_type": self.retraining_config.window_type,
                "window_years": self.retraining_config.window_years,
                "time_decay_enabled": self.retraining_config.time_decay.enabled,
                "time_decay_lambda": self.retraining_config.time_decay.lambda_,
            }
        
        return {
            "summary": {
                "start_date": str(trading_days[0]),
                "end_date": str(trading_days[-1]),
                "trading_days": len(trading_days),
                "universe_size": len(self.universe),
                "top_k": self.top_k,
                "initial_capital": self.initial_capital,
                "final_value": portfolio_value,
                "total_return_pct": total_return * 100,
                "spy_return_pct": spy_total_return * 100 if spy_total_return else None,  # Backward compatibility
                "benchmark_returns_pct": {k: v * 100 for k, v in benchmark_total_returns.items()},  # New
                "alpha_pct": alpha_pct,
                "annualized_volatility_pct": volatility * 100,
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_drawdown * 100,
                "win_rate_pct": win_rate * 100,
            },
            "daily_results": daily_results,
            "retraining": retraining_info
        }


def format_report(results: dict, verbose: bool = False) -> str:
    """Format simulation results as a readable report."""
    summary = results["summary"]
    daily = results["daily_results"]
    
    lines = []
    lines.append("=" * 70)
    lines.append("DAILY TRADING SIMULATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Period: {summary['start_date']} to {summary['end_date']}")
    lines.append(f"Trading Days: {summary['trading_days']}")
    lines.append(f"Universe: {summary['universe_size']} assets")
    lines.append(f"Strategy: Long top-{summary['top_k']} (equal weight, daily rebalance)")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Initial Capital:     ${summary['initial_capital']:,.2f}")
    lines.append(f"  Final Value:         ${summary['final_value']:,.2f}")
    lines.append("")
    lines.append(f"  ML Strategy Return:  {summary['total_return_pct']:+.2f}%")
    if summary['spy_return_pct'] is not None:
        lines.append(f"  S&P 500 (SPY):       {summary['spy_return_pct']:+.2f}%")
        lines.append(f"  Alpha (excess):      {summary['alpha_pct']:+.2f}%")
    lines.append("")
    lines.append(f"  Volatility (ann.):   {summary['annualized_volatility_pct']:.2f}%")
    lines.append(f"  Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
    lines.append(f"  Max Drawdown:        {summary['max_drawdown_pct']:.2f}%")
    lines.append(f"  Win Rate:            {summary['win_rate_pct']:.1f}%")
    lines.append("")
    
    # Performance assessment
    lines.append("-" * 70)
    lines.append("ASSESSMENT")
    lines.append("-" * 70)
    if summary['alpha_pct'] is not None:
        if summary['alpha_pct'] > 1:
            lines.append("✅ ML Strategy OUTPERFORMED the S&P 500")
        elif summary['alpha_pct'] < -1:
            lines.append("❌ ML Strategy UNDERPERFORMED the S&P 500")
        else:
            lines.append("➖ ML Strategy roughly matched the S&P 500")
    lines.append("")
    
    # Daily breakdown
    lines.append("-" * 70)
    lines.append("DAILY BREAKDOWN")
    lines.append("-" * 70)
    lines.append(f"{'Date':<12} {'Positions':<40} {'Return':>10} {'Cumulative':>12}")
    lines.append("-" * 70)
    
    for day in daily:
        positions = day["positions"]
        if positions:
            pos_str = ", ".join([f"{p['symbol']}({p['weight']*100:.0f}%)" for p in positions[:4]])
            if len(positions) > 4:
                pos_str += "..."
        else:
            pos_str = "(no positions)"
        
        lines.append(
            f"{day['date']:<12} {pos_str:<40} "
            f"{day['portfolio_return']*100:>+9.2f}% "
            f"{day['cumulative_return']*100:>+11.2f}%"
        )
    
    lines.append("")
    
    if verbose:
        lines.append("-" * 70)
        lines.append("DETAILED POSITION RETURNS")
        lines.append("-" * 70)
        for day in daily:
            lines.append(f"\n{day['date']}:")
            for pos in day["positions"]:
                lines.append(f"  {pos['symbol']:6s} ({pos['weight']*100:5.1f}%): {pos['return']*100:+7.2f}%")
            lines.append(f"  {'Portfolio':6s}: {day['portfolio_return']*100:+7.2f}%")
            if day['spy_return'] is not None:
                lines.append(f"  {'SPY':6s}:      {day['spy_return']*100:+7.2f}%")
    
    lines.append("")
    
    # Retraining info
    if "retraining" in results:
        rt = results["retraining"]
        lines.append("-" * 70)
        lines.append("RETRAINING INFO")
        lines.append("-" * 70)
        lines.append(f"  Mode: {rt['mode']}")
        lines.append(f"  Number of Retrains: {rt['num_retrains']}")
        if rt['mode'] == 'policy' and 'config' in rt:
            cfg = rt['config']
            lines.append(f"  Cadence: {cfg['cadence_days']} days")
            lines.append(f"  Window: {cfg['window_type']}, {cfg['window_years']} years")
            if cfg['time_decay_enabled']:
                lines.append(f"  Time-decay: lambda={cfg['time_decay_lambda']}")
        if rt['history']:
            lines.append(f"  Retrain Dates:")
            for h in rt['history'][:5]:
                lines.append(f"    {h['date']} - {h['reason']}")
            if len(rt['history']) > 5:
                lines.append(f"    ... and {len(rt['history']) - 5} more")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("=" * 70)
    lines.append("• Model trained on data BEFORE simulation period (no lookahead)")
    lines.append("• Features computed using only data available as of each date")
    lines.append("• Returns assume execution at close prices (no slippage/costs)")
    lines.append("• Universe: Curated subset of S&P 500 large-cap stocks")
    lines.append("")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    # Load config and storage
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    try:
        # Get retraining config if using policy mode
        retraining_config = None
        if args.use_policy:
            retraining_config = config.retraining
            # Override with command-line args if provided
            if args.retrain_cadence is not None:
                retraining_config.cadence_days = args.retrain_cadence
                logger.info(f"Overriding retrain cadence to {args.retrain_cadence} days")
            if args.time_decay_lambda is not None:
                retraining_config.time_decay.lambda_ = args.time_decay_lambda
                logger.info(f"Overriding time-decay lambda to {args.time_decay_lambda}")
        
        # Create simulator
        simulator = DailyTradingSimulator(
            storage=storage,
            config=config,
            top_k=args.top_k,
            train_days=args.train_days,
            initial_capital=args.initial_capital,
            use_policy=args.use_policy,
            retraining_config=retraining_config
        )
        
        # Run simulation
        results = simulator.simulate(
            start_date=start_date,
            end_date=end_date,
            retrain_frequency=args.retrain_frequency,
            verbose=args.verbose and not args.output_json
        )
        
        # Output results
        if args.output_json:
            print(json.dumps(results, indent=2))
        else:
            report = format_report(results, verbose=args.verbose)
            print(report)
            
            # Save report if requested
            if args.save_report:
                report_path = Path(args.save_report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                if args.save_report.endswith('.json'):
                    with open(report_path, 'w') as f:
                        json.dump(results, f, indent=2)
                else:
                    with open(report_path, 'w') as f:
                        f.write(report)
                
                logger.info(f"Report saved to {report_path}")
    
    finally:
        storage.close()


if __name__ == "__main__":
    main()

