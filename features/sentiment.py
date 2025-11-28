"""Sentiment and news-based features."""

import pandas as pd
import numpy as np
from typing import Optional, Set
from datetime import date, timedelta
from data.asof_api import AsOfQueryAPI


class SentimentFeatureBuilder:
    """Builds sentiment features from news data."""
    
    def __init__(self, api: AsOfQueryAPI):
        self.api = api
    
    def aggregate_sentiment(
        self,
        news_df: pd.DataFrame,
        decay_factor: float = 0.95
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores with time decay.
        
        Args:
            news_df: DataFrame with columns: asset_id, timestamp, vendor_sentiment_score
            decay_factor: Exponential decay factor per day
        
        Returns:
            DataFrame with aggregated sentiment features per asset_id and date
        """
        if len(news_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'sentiment_mean', 'sentiment_std', 'news_count'])
        
        news_df = news_df.copy()
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        
        # Compute time weights (more recent = higher weight)
        max_date = news_df['date'].max()
        news_df['days_ago'] = (max_date - news_df['date']).apply(lambda x: x.days)
        news_df['time_weight'] = decay_factor ** news_df['days_ago']
        
        # Aggregate per asset and date
        aggregated = []
        
        for (asset_id, date), group in news_df.groupby(['asset_id', 'date']):
            if 'vendor_sentiment_score' in group.columns:
                scores = group['vendor_sentiment_score'].dropna()
                if len(scores) > 0:
                    weighted_scores = scores * group.loc[scores.index, 'time_weight']
                    
                    aggregated.append({
                        'asset_id': asset_id,
                        'date': date,
                        'sentiment_mean': weighted_scores.mean(),
                        'sentiment_std': weighted_scores.std(),
                        'news_count': len(group),
                        'bullish_share': (scores > 0).sum() / len(scores) if len(scores) > 0 else 0,
                        'bearish_share': (scores < 0).sum() / len(scores) if len(scores) > 0 else 0
                    })
            else:
                aggregated.append({
                    'asset_id': asset_id,
                    'date': date,
                    'sentiment_mean': np.nan,
                    'sentiment_std': np.nan,
                    'news_count': len(group),
                    'bullish_share': np.nan,
                    'bearish_share': np.nan
                })
        
        if len(aggregated) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'sentiment_mean', 'sentiment_std', 'news_count'])
        
        return pd.DataFrame(aggregated)
    
    def build_features(
        self,
        as_of_date: date,
        lookback_days: int = 30,
        universe: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """
        Build sentiment features as-of a specific date.
        """
        # Get news within lookback window
        news_df = self.api.get_news_asof(as_of_date, lookback_days, universe)
        
        if len(news_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date'])
        
        # Aggregate sentiment
        sentiment_features = self.aggregate_sentiment(news_df)
        
        # Add date column for merging
        sentiment_features['date'] = as_of_date
        
        return sentiment_features

