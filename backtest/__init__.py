"""Backtesting framework."""

from backtest.vectorized import VectorizedBacktester
from backtest.metrics import PerformanceMetrics
from backtest.benchmarks import BenchmarkStrategies
from backtest.data_prep import (
    BacktestDataConfig,
    get_universe_asset_ids,
    sample_trading_dates,
    build_training_data,
    prepare_test_data,
    check_data_coverage,
)
from backtest.experiments import (
    ExperimentConfig,
    ExperimentResult,
    run_walk_forward_evaluation,
    run_policy_driven_backtest,
    run_regime_aware_backtest,
)

__all__ = [
    # Core backtest
    "VectorizedBacktester",
    "PerformanceMetrics",
    "BenchmarkStrategies",
    # Data prep
    "BacktestDataConfig",
    "get_universe_asset_ids",
    "sample_trading_dates",
    "build_training_data",
    "prepare_test_data",
    "check_data_coverage",
    # Experiments
    "ExperimentConfig",
    "ExperimentResult",
    "run_walk_forward_evaluation",
    "run_policy_driven_backtest",
    "run_regime_aware_backtest",
]
