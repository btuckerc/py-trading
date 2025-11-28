"""Data normalization and point-in-time table construction."""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .storage import StorageBackend


class DataNormalizer:
    """Normalizes vendor data into canonical schemas."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.symbol_to_asset_id: Dict[str, int] = {}
        self._load_asset_mapping()
    
    def _load_asset_mapping(self):
        """Load existing symbol to asset_id mapping from database."""
        try:
            assets_df = self.storage.query("SELECT asset_id, symbol FROM assets")
            if len(assets_df) > 0:
                self.symbol_to_asset_id = dict(zip(assets_df['symbol'], assets_df['asset_id']))
        except Exception:
            # Table might not exist yet
            pass
    
    def _get_or_create_asset_id(self, symbol: str, exchange: Optional[str] = None) -> int:
        """Get or create asset_id for a symbol."""
        if symbol in self.symbol_to_asset_id:
            return self.symbol_to_asset_id[symbol]
        
        # Create new asset_id
        max_id = max(self.symbol_to_asset_id.values()) if self.symbol_to_asset_id else 0
        asset_id = max_id + 1
        
        # Insert into assets table
        asset_df = pd.DataFrame([{
            'asset_id': asset_id,
            'symbol': symbol,
            'exchange': exchange or 'NYSE',
            'currency': 'USD',
            'is_active': True
        }])
        self.storage.insert_dataframe('assets', asset_df)
        
        self.symbol_to_asset_id[symbol] = asset_id
        return asset_id
    
    def normalize_bars(
        self,
        bars_df: pd.DataFrame,
        vendor: str = "yahoo"
    ) -> pd.DataFrame:
        """
        Normalize bars DataFrame to canonical schema.
        
        Input: DataFrame with columns [symbol, date, open, high, low, close, adj_close, volume]
        Output: DataFrame with columns [asset_id, date, open, high, low, close, adj_close, volume, data_vendor, ingestion_timestamp]
        """
        if len(bars_df) == 0:
            return pd.DataFrame(columns=[
                'asset_id', 'date', 'open', 'high', 'low', 'close', 'adj_close', 
                'volume', 'data_vendor', 'ingestion_timestamp'
            ])
        
        normalized = bars_df.copy()
        
        # Map symbols to asset_ids
        normalized['asset_id'] = normalized['symbol'].apply(
            lambda s: self._get_or_create_asset_id(s)
        )
        
        # Ensure date is date type
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date']).dt.date
        
        # Add vendor and timestamp
        normalized['data_vendor'] = vendor
        normalized['ingestion_timestamp'] = datetime.now()
        
        # Select and order columns
        result = normalized[[
            'asset_id', 'date', 'open', 'high', 'low', 'close', 'adj_close',
            'volume', 'data_vendor', 'ingestion_timestamp'
        ]]
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['asset_id', 'date'])
        
        return result
    
    def normalize_corporate_actions(
        self,
        actions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize corporate actions DataFrame.
        
        Input: DataFrame with columns [symbol, date, split_factor, dividend_amount]
        Output: DataFrame with columns [asset_id, date, split_factor, dividend_amount, special_dividend]
        """
        if len(actions_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'split_factor', 'dividend_amount', 'special_dividend'])
        
        normalized = actions_df.copy()
        
        # Map symbols to asset_ids
        normalized['asset_id'] = normalized['symbol'].apply(
            lambda s: self._get_or_create_asset_id(s)
        )
        
        # Ensure date is date type
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date']).dt.date
        
        # Add special_dividend column (default 0)
        if 'special_dividend' not in normalized.columns:
            normalized['special_dividend'] = 0.0
        
        # Select columns
        result = normalized[[
            'asset_id', 'date', 'split_factor', 'dividend_amount', 'special_dividend'
        ]]
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['asset_id', 'date'])
        
        return result
    
    def normalize_fundamentals(
        self,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize fundamentals DataFrame.
        
        Input: DataFrame with columns including symbol, period_end_date, report_release_date, and financial metrics
        Output: DataFrame with canonical schema
        """
        if len(fundamentals_df) == 0:
            return pd.DataFrame(columns=[
                'asset_id', 'period_end_date', 'report_release_date',
                'eps', 'eps_estimate', 'revenue', 'ebitda', 'total_debt',
                'total_equity', 'free_cash_flow'
            ])
        
        normalized = fundamentals_df.copy()
        
        # Map symbols to asset_ids
        if 'symbol' in normalized.columns:
            normalized['asset_id'] = normalized['symbol'].apply(
                lambda s: self._get_or_create_asset_id(s)
            )
        
        # Ensure dates are date type
        for date_col in ['period_end_date', 'report_release_date']:
            if date_col in normalized.columns:
                normalized[date_col] = pd.to_datetime(normalized[date_col]).dt.date
        
        # Ensure report_release_date exists (critical for point-in-time correctness)
        if 'report_release_date' not in normalized.columns:
            # If not provided, assume release date is 45 days after period end (typical for quarterly reports)
            if 'period_end_date' in normalized.columns:
                normalized['report_release_date'] = pd.to_datetime(normalized['period_end_date']) + pd.Timedelta(days=45)
                normalized['report_release_date'] = normalized['report_release_date'].dt.date
            else:
                raise ValueError("Must provide either report_release_date or period_end_date")
        
        # Select columns (fill missing with None)
        result_cols = [
            'asset_id', 'period_end_date', 'report_release_date',
            'eps', 'eps_estimate', 'revenue', 'ebitda', 'total_debt',
            'total_equity', 'free_cash_flow'
        ]
        
        result = pd.DataFrame()
        for col in result_cols:
            if col in normalized.columns:
                result[col] = normalized[col]
            else:
                result[col] = None
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['asset_id', 'period_end_date'])
        
        return result
    
    def normalize_news(
        self,
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize news DataFrame.
        
        Input: DataFrame with columns [symbol, timestamp, headline, source, url, vendor_sentiment_score]
        Output: DataFrame with canonical schema
        """
        if len(news_df) == 0:
            return pd.DataFrame(columns=[
                'asset_id', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'
            ])
        
        normalized = news_df.copy()
        
        # Map symbols to asset_ids
        if 'symbol' in normalized.columns:
            normalized['asset_id'] = normalized['symbol'].apply(
                lambda s: self._get_or_create_asset_id(s)
            )
        
        # Ensure timestamp is datetime
        if 'timestamp' in normalized.columns:
            normalized['timestamp'] = pd.to_datetime(normalized['timestamp'])
        
        # Select columns
        result = normalized[[
            'asset_id', 'timestamp', 'headline', 'source', 'url', 'vendor_sentiment_score'
        ]]
        
        # Remove duplicates (same asset, same timestamp)
        result = result.drop_duplicates(subset=['asset_id', 'timestamp'])
        
        return result
    
    def validate_adj_close_consistency(
        self,
        bars_df: pd.DataFrame,
        actions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Validate that adj_close is consistent with corporate actions.
        
        Returns DataFrame with validation results (asset_id, date, issue_type, message)
        """
        issues = []
        
        # Group actions by asset and date
        if len(actions_df) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'issue_type', 'message'])
        
        for (asset_id, action_date), group in actions_df.groupby(['asset_id', 'date']):
            # Get bars before and after this action
            asset_bars = bars_df[bars_df['asset_id'] == asset_id].sort_values('date')
            
            if len(asset_bars) == 0:
                continue
            
            # Check if there's a bar on the action date
            action_bars = asset_bars[asset_bars['date'] == action_date]
            
            if len(action_bars) == 0:
                issues.append({
                    'asset_id': asset_id,
                    'date': action_date,
                    'issue_type': 'missing_bar',
                    'message': f'No bar data on corporate action date'
                })
                continue
            
            # Check split adjustment
            split_factor = group['split_factor'].iloc[0]
            if split_factor != 1.0:
                # Get price before and after split
                before_bars = asset_bars[asset_bars['date'] < action_date]
                after_bars = asset_bars[asset_bars['date'] > action_date]
                
                if len(before_bars) > 0 and len(after_bars) > 0:
                    price_before = before_bars.iloc[-1]['close']
                    price_after = after_bars.iloc[0]['close']
                    expected_ratio = split_factor
                    actual_ratio = price_after / price_before if price_before > 0 else 0
                    
                    # Allow some tolerance (e.g., 5%)
                    if abs(actual_ratio - expected_ratio) / expected_ratio > 0.05:
                        issues.append({
                            'asset_id': asset_id,
                            'date': action_date,
                            'issue_type': 'split_mismatch',
                            'message': f'Split factor {split_factor} but price ratio is {actual_ratio:.4f}'
                        })
        
        if len(issues) == 0:
            return pd.DataFrame(columns=['asset_id', 'date', 'issue_type', 'message'])
        
        return pd.DataFrame(issues)

