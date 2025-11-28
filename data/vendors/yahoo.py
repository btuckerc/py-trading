"""Yahoo Finance vendor client via yfinance."""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict
from datetime import date, datetime
from .base import BaseVendorClient
from loguru import logger


class YahooClient(BaseVendorClient):
    """Yahoo Finance client using yfinance library."""
    
    def __init__(self):
        self.name = "yahoo"
    
    def fetch_asset_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch asset metadata including sector and industry from Yahoo Finance.
        
        Args:
            symbols: List of ticker symbols
        
        Returns:
            DataFrame with columns: symbol, sector, industry, exchange, currency, name
        """
        all_info = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract relevant fields with fallbacks
                asset_info = {
                    'symbol': symbol,
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'exchange': info.get('exchange'),
                    'currency': info.get('currency', 'USD'),
                    'name': info.get('longName') or info.get('shortName'),
                    'quote_type': info.get('quoteType'),  # EQUITY, ETF, etc.
                }
                
                # Handle ETFs which don't have sector/industry
                if asset_info['quote_type'] == 'ETF':
                    asset_info['sector'] = 'ETF'
                    asset_info['industry'] = info.get('category', 'Index ETF')
                
                all_info.append(asset_info)
                logger.debug(f"Fetched info for {symbol}: sector={asset_info['sector']}, industry={asset_info['industry']}")
                
            except Exception as e:
                logger.warning(f"Error fetching info for {symbol}: {e}")
                # Add placeholder entry
                all_info.append({
                    'symbol': symbol,
                    'sector': None,
                    'industry': None,
                    'exchange': None,
                    'currency': 'USD',
                    'name': None,
                    'quote_type': None,
                })
        
        if len(all_info) == 0:
            return pd.DataFrame(columns=['symbol', 'sector', 'industry', 'exchange', 'currency', 'name', 'quote_type'])
        
        return pd.DataFrame(all_info)
    
    def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Yahoo Finance."""
        all_bars = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if len(hist) == 0:
                    continue
                
                # Reset index to get date as column
                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['date'] = pd.to_datetime(hist['Date']).dt.date
                
                # Rename columns to match schema
                bars = pd.DataFrame({
                    'symbol': hist['symbol'],
                    'date': hist['date'],
                    'open': hist['Open'],
                    'high': hist['High'],
                    'low': hist['Low'],
                    'close': hist['Close'],
                    'adj_close': hist.get('Adj Close', hist['Close']),  # Fallback to Close if no Adj Close
                    'volume': hist['Volume']
                })
                
                all_bars.append(bars)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
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
        """Fetch corporate actions (splits and dividends) from Yahoo Finance."""
        all_actions = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get splits
                splits = ticker.splits
                if len(splits) > 0:
                    splits_df = splits.reset_index()
                    splits_df['symbol'] = symbol
                    splits_df['date'] = pd.to_datetime(splits_df['Date']).dt.date
                    splits_df['split_factor'] = splits_df['Stock Splits']
                    splits_df['dividend_amount'] = 0.0
                    splits_df = splits_df[
                        (splits_df['date'] >= start_date) & 
                        (splits_df['date'] <= end_date)
                    ]
                    if len(splits_df) > 0:
                        all_actions.append(splits_df[['symbol', 'date', 'split_factor', 'dividend_amount']])
                
                # Get dividends
                dividends = ticker.dividends
                if len(dividends) > 0:
                    div_df = dividends.reset_index()
                    div_df['symbol'] = symbol
                    div_df['date'] = pd.to_datetime(div_df['Date']).dt.date
                    div_df['split_factor'] = 1.0
                    div_df['dividend_amount'] = div_df['Dividends']
                    div_df = div_df[
                        (div_df['date'] >= start_date) & 
                        (div_df['date'] <= end_date)
                    ]
                    if len(div_df) > 0:
                        all_actions.append(div_df[['symbol', 'date', 'split_factor', 'dividend_amount']])
            except Exception as e:
                print(f"Error fetching corporate actions for {symbol}: {e}")
                continue
        
        if len(all_actions) == 0:
            return pd.DataFrame(columns=['symbol', 'date', 'split_factor', 'dividend_amount'])
        
        result = pd.concat(all_actions, ignore_index=True)
        # Group by symbol and date, sum dividends and multiply split factors
        result = result.groupby(['symbol', 'date']).agg({
            'split_factor': 'prod',
            'dividend_amount': 'sum'
        }).reset_index()
        return result
    
    def fetch_fundamentals(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch fundamental data from Yahoo Finance."""
        # Yahoo Finance doesn't provide historical fundamentals easily
        # This is a placeholder that would need to be implemented with another vendor
        # or by scraping financial statements
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
        """Fetch news data from Yahoo Finance."""
        all_news = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                if len(news) > 0:
                    news_df = pd.DataFrame(news)
                    news_df['symbol'] = symbol
                    news_df['timestamp'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
                    news_df['headline'] = news_df.get('title', '')
                    news_df['source'] = news_df.get('publisher', '')
                    news_df['url'] = news_df.get('link', '')
                    news_df['vendor_sentiment_score'] = None  # Yahoo doesn't provide sentiment
                    
                    news_df = news_df[
                        (news_df['timestamp'].dt.date >= start_date) &
                        (news_df['timestamp'].dt.date <= end_date)
                    ]
                    
                    if len(news_df) > 0:
                        all_news.append(news_df[[
                            'symbol', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'
                        ]])
            except Exception as e:
                print(f"Error fetching news for {symbol}: {e}")
                continue
        
        if len(all_news) == 0:
            return pd.DataFrame(columns=['symbol', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'])
        
        result = pd.concat(all_news, ignore_index=True)
        return result

