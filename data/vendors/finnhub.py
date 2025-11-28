"""Finnhub vendor client (placeholder for news/sentiment)."""

import pandas as pd
from typing import List, Optional
from datetime import date
import os
import requests
from .base import BaseVendorClient


class FinnhubClient(BaseVendorClient):
    """Finnhub API client for news and sentiment."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "finnhub"
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("Finnhub API key required. Set FINNHUB_API_KEY environment variable.")
        self.base_url = "https://finnhub.io/api/v1"
    
    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Finnhub doesn't provide historical daily bars easily - use Yahoo/Tiingo instead."""
        return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
    
    def fetch_corporate_actions(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Finnhub doesn't provide corporate actions."""
        return pd.DataFrame(columns=['symbol', 'date', 'split_factor', 'dividend_amount'])
    
    def fetch_fundamentals(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch fundamentals from Finnhub."""
        # Placeholder - would need to implement Finnhub fundamentals API
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
        """Fetch company news from Finnhub."""
        all_news = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_url}/company-news"
                params = {
                    "symbol": symbol,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat(),
                    "token": self.api_key
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if len(data) == 0:
                    continue
                
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df['headline'] = df.get('headline', '')
                df['source'] = df.get('source', '')
                df['url'] = df.get('url', '')
                df['vendor_sentiment_score'] = None  # Would need sentiment endpoint
                
                news_df = df[[
                    'symbol', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'
                ]]
                all_news.append(news_df)
            except Exception as e:
                print(f"Error fetching news for {symbol} from Finnhub: {e}")
                continue
        
        if len(all_news) == 0:
            return pd.DataFrame(columns=['symbol', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'])
        
        result = pd.concat(all_news, ignore_index=True)
        return result

