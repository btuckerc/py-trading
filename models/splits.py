"""Time-series train/validation/test splitting."""

import pandas as pd
from typing import List, Dict, Tuple
from datetime import date


class TimeSplit:
    """Manages time-respecting train/val/test splits."""
    
    def __init__(self, splits: List[Dict]):
        """
        Args:
            splits: List of split configs, each with:
                - name: str
                - train_start: date
                - train_end: date
                - val_start: date
                - val_end: date
                - test_start: date
                - test_end: date
        """
        self.splits = splits
    
    def get_split(self, split_name: str) -> Dict:
        """Get a specific split by name."""
        for split in self.splits:
            if split['name'] == split_name:
                return split
        raise ValueError(f"Split {split_name} not found")
    
    def get_date_ranges(self, split_name: str) -> Tuple[date, date, date, date, date, date]:
        """Get date ranges for a split."""
        split = self.get_split(split_name)
        return (
            pd.to_datetime(split['train_start']).date(),
            pd.to_datetime(split['train_end']).date(),
            pd.to_datetime(split['val_start']).date(),
            pd.to_datetime(split['val_end']).date(),
            pd.to_datetime(split['test_start']).date(),
            pd.to_datetime(split['test_end']).date()
        )
    
    def filter_dataframe(
        self,
        df: pd.DataFrame,
        split_name: str,
        split_type: str,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Filter DataFrame to a specific split.
        
        Args:
            df: DataFrame with date column
            split_name: Name of split
            split_type: "train", "val", or "test"
            date_col: Name of date column
        """
        train_start, train_end, val_start, val_end, test_start, test_end = self.get_date_ranges(split_name)
        
        if split_type == "train":
            start_date, end_date = train_start, train_end
        elif split_type == "val":
            start_date, end_date = val_start, val_end
        elif split_type == "test":
            start_date, end_date = test_start, test_end
        else:
            raise ValueError(f"Unknown split_type: {split_type}")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        filtered = df[
            (df[date_col] >= pd.Timestamp(start_date)) &
            (df[date_col] <= pd.Timestamp(end_date))
        ]
        
        return filtered

