"""Smoke tests to verify core modules import and basic functionality works."""

import pytest
from datetime import date


class TestImports:
    """Test that all core modules can be imported."""

    def test_import_data_vendors(self):
        """Test data vendor imports."""
        from data.vendors.yahoo import YahooClient
        from data.vendors.base import BaseVendorClient
        
        assert YahooClient is not None
        assert BaseVendorClient is not None

    def test_import_data_modules(self):
        """Test data module imports."""
        from data.storage import StorageBackend
        from data.normalize import DataNormalizer
        from data.quality import DataQualityChecker
        from data.clock import SimulationClock
        from data.universe import UniverseManager
        
        assert StorageBackend is not None
        assert DataNormalizer is not None
        assert DataQualityChecker is not None
        assert SimulationClock is not None
        assert UniverseManager is not None

    def test_import_features(self):
        """Test feature module imports."""
        from features.technical import TechnicalFeatureBuilder
        from features.cross_sectional import CrossSectionalFeatureBuilder
        from features.calendar import CalendarFeatureBuilder
        from features.pipeline import FeaturePipeline
        
        assert TechnicalFeatureBuilder is not None
        assert CrossSectionalFeatureBuilder is not None
        assert CalendarFeatureBuilder is not None
        assert FeaturePipeline is not None

    def test_import_labels(self):
        """Test label module imports."""
        from labels.returns import ReturnLabelGenerator
        from labels.regimes import RegimeLabelGenerator
        
        assert ReturnLabelGenerator is not None
        assert RegimeLabelGenerator is not None

    def test_import_models(self):
        """Test model module imports."""
        from models.base import BaseModel, MultiHorizonModel
        from models.baselines import persistence_strategy, momentum_strategy
        from models.tabular import XGBoostModel, LightGBMModel, LogisticRegressionModel
        from models.splits import TimeSplit
        
        assert BaseModel is not None
        assert MultiHorizonModel is not None
        assert persistence_strategy is not None
        assert momentum_strategy is not None
        assert XGBoostModel is not None
        assert LightGBMModel is not None
        assert LogisticRegressionModel is not None
        assert TimeSplit is not None

    def test_import_torch_models(self):
        """Test PyTorch model imports."""
        from models.torch.conv_lstm import ConvLSTMModel
        from models.torch.tcn import TCNModel
        from models.torch.transformer import TransformerModel
        from models.torch.dataset import SequenceDataset
        
        assert ConvLSTMModel is not None
        assert TCNModel is not None
        assert TransformerModel is not None
        assert SequenceDataset is not None

    def test_import_portfolio(self):
        """Test portfolio module imports."""
        from portfolio.strategies import LongTopKStrategy, LongShortDecilesStrategy
        from portfolio.risk import RiskManager
        from portfolio.costs import TransactionCostModel
        from portfolio.scores import ScoreConverter
        
        assert LongTopKStrategy is not None
        assert LongShortDecilesStrategy is not None
        assert RiskManager is not None
        assert TransactionCostModel is not None
        assert ScoreConverter is not None

    def test_import_backtest(self):
        """Test backtest module imports."""
        from backtest.vectorized import VectorizedBacktester
        from backtest.metrics import PerformanceMetrics
        from backtest.benchmarks import BenchmarkStrategies
        from backtest.validators import BacktestValidator
        
        assert VectorizedBacktester is not None
        assert PerformanceMetrics is not None
        assert BenchmarkStrategies is not None
        assert BacktestValidator is not None

    def test_import_live(self):
        """Test live trading module imports."""
        from live.engine import LiveEngine
        from live.broker_base import BrokerAPI
        from live.paper_broker import PaperBroker
        
        assert LiveEngine is not None
        assert BrokerAPI is not None
        assert PaperBroker is not None

    def test_import_configs(self):
        """Test config module imports."""
        from configs.loader import get_config
        
        assert get_config is not None


class TestYahooClient:
    """Test Yahoo Finance client basic functionality."""

    def test_yahoo_client_init(self):
        """Test YahooClient can be instantiated."""
        from data.vendors.yahoo import YahooClient
        
        client = YahooClient()
        assert client.name == "yahoo"

    def test_yahoo_client_fetch_empty_symbols(self):
        """Test YahooClient returns empty DataFrame for empty symbol list."""
        from data.vendors.yahoo import YahooClient
        
        client = YahooClient()
        result = client.fetch_daily_bars(
            symbols=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        assert len(result) == 0
        assert list(result.columns) == ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        """Test that config loads without errors."""
        from configs.loader import get_config
        
        config = get_config()
        assert config is not None

    def test_config_has_expected_sections(self):
        """Test config has expected top-level sections."""
        from configs.loader import get_config
        
        config = get_config()
        
        # Check for expected attributes (depends on config structure)
        assert hasattr(config, 'universe') or 'universe' in dir(config)


class TestSimulationClock:
    """Test simulation clock functionality."""

    def test_clock_initialization(self):
        """Test SimulationClock can be initialized with date range."""
        from data.clock import SimulationClock
        
        clock = SimulationClock(
            start_date=date(2024, 1, 2),  # Use a trading day
            end_date=date(2024, 1, 31)
        )
        assert clock.start_date == date(2024, 1, 2)
        assert clock.end_date == date(2024, 1, 31)

    def test_clock_iteration(self):
        """Test that clock can iterate over trading days."""
        from data.clock import SimulationClock
        
        clock = SimulationClock(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 10)
        )
        
        # Should have some trading days
        trading_days = list(clock)
        assert len(trading_days) > 0
        
        # All days should be within range
        for day in trading_days:
            assert date(2024, 1, 2) <= day <= date(2024, 1, 10)
