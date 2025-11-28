"""Tiingo vendor client (placeholder for when API key is available)."""

import pandas as pd
from typing import List, Optional
from datetime import date
import os
import requests
from .base import BaseVendorClient


class TiingoClient(BaseVendorClient):
    """Tiingo API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "tiingo"
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("Tiingo API key required. Set TIINGO_API_KEY environment variable.")
        self.base_url = "https://api.tiingo.com/tiingo"
    
    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Tiingo."""
        all_bars = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_url}/daily/{symbol}/prices"
                params = {
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "token": self.api_key
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if len(data) == 0:
                    continue
                
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df['date'] = pd.to_datetime(df['date']).dt.date
                
                bars = pd.DataFrame({
                    'symbol': df['symbol'],
                    'date': df['date'],
                    'open': df['open'],
                    'high': df['high'],
                    'low': df['low'],
                    'close': df['close'],
                    'adj_close': df.get('adjClose', df['close']),
                    'volume': df['volume']
                })
                
                all_bars.append(bars)
            except Exception as e:
                print(f"Error fetching {symbol} from Tiingo: {e}")
                continue
        
        if len(all_bars) == 0:
            return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        
        result = pd.concat(all_bars, ignore_index=True)
        return result
    
    def fetch_corporate_actions(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Fetch corporate actions from Tiingo."""
        # Tiingo provides this via a separate endpoint
        # Placeholder implementation
        return pd.DataFrame(columns=['symbol', 'date', 'split_factor', 'dividend_amount'])
    
    def fetch_fundamentals(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch fundamental data from Tiingo."""
        # Tiingo fundamentals endpoint
        # Placeholder implementation
        return pd.DataFrame(columns=[
            'symbol', 'period_end_date', 'report_release_date',
            'eps', 'revenue', 'ebitda', 'total_debt', 'total_equity', 'free_cash_flow'
        ])
    
    def fetch_news(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Fetch news from Tiingo."""
        # Tiingo news endpoint
        # Placeholder implementation
        return pd.DataFrame(columns=['symbol', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'])

