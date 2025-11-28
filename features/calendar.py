"""Calendar and temporal features."""

import pandas as pd
import numpy as np
from datetime import date
from typing import Optional


class CalendarFeatureBuilder:
    """Builds calendar and temporal features."""
    
    def build_features(
        self,
        dates: pd.Series,
        earnings_dates: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build calendar features.
        
        Args:
            dates: Series of dates
            earnings_dates: Optional DataFrame with columns: asset_id, earnings_date
        
        Returns:
            DataFrame with calendar features
        """
        features = pd.DataFrame(index=dates.index)
        features['date'] = dates
        
        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = pd.to_datetime(dates).dt.dayofweek
        
        # Day of month
        features['day_of_month'] = pd.to_datetime(dates).dt.day
        
        # Month
        features['month'] = pd.to_datetime(dates).dt.month
        
        # Quarter
        features['quarter'] = pd.to_datetime(dates).dt.quarter
        
        # Is month end
        features['is_month_end'] = pd.to_datetime(dates).dt.is_month_end
        
        # Is quarter end
        features['is_quarter_end'] = pd.to_datetime(dates).dt.is_quarter_end
        
        # Days to next earnings (if earnings dates provided)
        if earnings_dates is not None and len(earnings_dates) > 0:
            # This would need to be merged per asset_id
            # For now, placeholder
            features['days_to_earnings'] = np.nan
            features['days_since_earnings'] = np.nan
        
        # Holiday flags (simplified - would need actual holiday calendar)
        # FOMC dates (simplified - would need actual FOMC calendar)
        features['is_holiday'] = False  # Placeholder
        features['is_fomc_day'] = False  # Placeholder
        
        return features

