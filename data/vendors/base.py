"""Base vendor client interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date
import pandas as pd


class BaseVendorClient(ABC):
    """Abstract base class for vendor data clients."""
    
    @abstractmethod
    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars.
        
        Returns DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
        """
        pass
    
    @abstractmethod
    def fetch_corporate_actions(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch corporate actions (splits, dividends).
        
        Returns DataFrame with columns: symbol, date, split_factor, dividend_amount
        """
        pass
    
    @abstractmethod
    def fetch_fundamentals(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch fundamental data.
        
        Returns DataFrame with columns: symbol, period_end_date, report_release_date, eps, revenue, etc.
        """
        pass
    
    @abstractmethod
    def fetch_news(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch news/sentiment data.
        
        Returns DataFrame with columns: symbol, timestamp, headline, source, url, sentiment_score
        """
        pass

