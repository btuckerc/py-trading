"""Simulation clock for point-in-time correctness."""

from datetime import date
from typing import Iterator, Optional
from .universe import TradingCalendar


class SimulationClock:
    """
    Simulation clock that iterates over trading dates.
    
    Ensures all queries are "as-of" a specific date to prevent lookahead bias.
    """
    
    def __init__(self, start_date: date, end_date: date):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = TradingCalendar()
        self.trading_days = self.calendar.get_trading_days(start_date, end_date)
        self.current_index = 0
    
    def __iter__(self) -> Iterator[date]:
        """Iterate over trading days."""
        for trading_day in self.trading_days:
            yield trading_day.date()
    
    def __len__(self) -> int:
        """Number of trading days."""
        return len(self.trading_days)
    
    @property
    def now(self) -> date:
        """Current simulation date."""
        if self.current_index < len(self.trading_days):
            return self.trading_days[self.current_index].date()
        return self.end_date
    
    def advance(self) -> bool:
        """Advance to next trading day. Returns False if at end."""
        if self.current_index < len(self.trading_days) - 1:
            self.current_index += 1
            return True
        return False
    
    def reset(self):
        """Reset clock to start."""
        self.current_index = 0
    
    def set_date(self, target_date: date):
        """Set clock to a specific date."""
        import pandas as pd
        try:
            idx = self.trading_days.get_loc(pd.Timestamp(target_date))
            self.current_index = idx
        except KeyError:
            raise ValueError(f"{target_date} is not a trading day in the range")
    
    def get_lookback_dates(self, lookback_days: int) -> list[date]:
        """Get list of dates going back lookback_days trading days from now."""
        if self.current_index < lookback_days:
            return [d.date() for d in self.trading_days[:self.current_index + 1]]
        start_idx = self.current_index - lookback_days + 1
        return [d.date() for d in self.trading_days[start_idx:self.current_index + 1]]

