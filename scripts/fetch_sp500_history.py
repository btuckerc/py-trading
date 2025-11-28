"""Fetch historical S&P 500 constituents data.

This script downloads historical S&P 500 membership data from available sources
and creates a CSV file that can be used by build_universe.py.

Sources:
1. Wikipedia (current constituents)
2. fja05680/sp500 GitHub repo (historical changes since 1996)
3. Manual fallback with major constituents

The output CSV has format:
    date,symbol,action
    2020-01-01,AAPL,added
    2020-06-01,XYZ,removed
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import argparse
import pandas as pd
import requests
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def fetch_from_github_fja05680() -> pd.DataFrame:
    """
    Fetch historical S&P 500 changes from fja05680/sp500 GitHub repo.
    
    This repo maintains historical membership changes since 1996.
    """
    base_url = "https://raw.githubusercontent.com/fja05680/sp500/master"
    
    try:
        # Try to get the S&P 500 changes file
        changes_url = f"{base_url}/S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2024).csv"
        logger.info(f"Fetching S&P 500 historical changes from GitHub...")
        
        response = requests.get(changes_url, timeout=30)
        
        if response.status_code != 200:
            # Try alternative URL patterns
            alt_urls = [
                f"{base_url}/S%26P%20500%20Historical%20Components%20%26%20Changes.csv",
                f"{base_url}/sp500_changes.csv",
            ]
            for alt_url in alt_urls:
                response = requests.get(alt_url, timeout=30)
                if response.status_code == 200:
                    break
        
        if response.status_code != 200:
            logger.warning(f"Could not fetch from GitHub (status {response.status_code})")
            return pd.DataFrame()
        
        # Parse the CSV
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Fetched {len(df)} records from GitHub")
        
        return df
        
    except Exception as e:
        logger.warning(f"Error fetching from GitHub: {e}")
        return pd.DataFrame()


def fetch_current_from_wikipedia() -> list:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    
    Returns list of current ticker symbols.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        logger.info("Fetching current S&P 500 constituents from Wikipedia...")
        
        tables = pd.read_html(url)
        
        if len(tables) > 0:
            # First table is usually the current constituents
            df = tables[0]
            
            # Find the symbol column
            symbol_col = None
            for col in df.columns:
                if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                    symbol_col = col
                    break
            
            if symbol_col is None and len(df.columns) > 0:
                symbol_col = df.columns[0]
            
            if symbol_col is not None:
                symbols = df[symbol_col].tolist()
                # Clean up symbols (remove class designations, etc.)
                symbols = [str(s).split('.')[0].strip() for s in symbols if pd.notna(s)]
                logger.info(f"Found {len(symbols)} current S&P 500 constituents")
                return symbols
        
        return []
        
    except Exception as e:
        logger.warning(f"Error fetching from Wikipedia: {e}")
        return []


def create_fallback_constituents() -> pd.DataFrame:
    """
    Create a fallback list of major S&P 500 constituents.
    
    This is used when online sources are unavailable.
    Includes the largest and most stable S&P 500 members.
    """
    logger.info("Using fallback list of major S&P 500 constituents...")
    
    # Major S&P 500 constituents that have been members for most of the period
    # This is a simplified list - actual membership changes are more complex
    major_constituents = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 
        'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM',
        'IBM', 'TXN', 'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'SNPS',
        
        # Finance
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C',
        'AXP', 'BLK', 'SCHW', 'SPGI', 'CME', 'ICE', 'MCO', 'CB',
        
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
        'DHR', 'BMY', 'AMGN', 'MDT', 'GILD', 'ISRG', 'CVS', 'CI',
        
        # Consumer
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE',
        'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR', 'CMG', 'YUM',
        
        # Industrial
        'CAT', 'DE', 'UPS', 'RTX', 'HON', 'BA', 'GE', 'LMT',
        'MMM', 'UNP', 'ADP', 'ITW', 'EMR', 'ETN', 'PH', 'ROK',
        
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
        
        # Communication
        'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR',
        
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX',
        
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O',
        
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL',
        
        # Index ETF (benchmark)
        'SPY'
    ]
    
    # Create a simple membership file assuming all were members from 2010
    records = []
    start_date = date(2010, 1, 1)
    
    for symbol in major_constituents:
        records.append({
            'date': start_date,
            'symbol': symbol,
            'action': 'added'
        })
    
    return pd.DataFrame(records)


def process_github_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the fja05680/sp500 GitHub data into our format.
    
    The GitHub data typically has columns like:
    - date: date of change
    - tickers: comma-separated list of current tickers OR
    - added/removed columns
    """
    if len(df) == 0:
        return pd.DataFrame(columns=['date', 'symbol', 'action'])
    
    records = []
    
    # Check column names (they vary by file version)
    columns_lower = [c.lower() for c in df.columns]
    
    if 'tickers' in columns_lower:
        # Format: date, tickers (comma-separated list of all current members)
        ticker_col = df.columns[columns_lower.index('tickers')]
        date_col = df.columns[0] if 'date' not in columns_lower else df.columns[columns_lower.index('date')]
        
        # Sort by date
        df = df.sort_values(date_col)
        
        previous_tickers = set()
        for _, row in df.iterrows():
            try:
                row_date = pd.to_datetime(row[date_col]).date()
                current_tickers = set(str(row[ticker_col]).split(','))
                current_tickers = {t.strip() for t in current_tickers if t.strip()}
                
                # Find additions
                added = current_tickers - previous_tickers
                for symbol in added:
                    records.append({
                        'date': row_date,
                        'symbol': symbol,
                        'action': 'added'
                    })
                
                # Find removals
                removed = previous_tickers - current_tickers
                for symbol in removed:
                    records.append({
                        'date': row_date,
                        'symbol': symbol,
                        'action': 'removed'
                    })
                
                previous_tickers = current_tickers
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
    
    elif 'added' in columns_lower or 'removed' in columns_lower:
        # Format: date, added, removed (explicit changes)
        date_col = df.columns[0] if 'date' not in columns_lower else df.columns[columns_lower.index('date')]
        added_col = df.columns[columns_lower.index('added')] if 'added' in columns_lower else None
        removed_col = df.columns[columns_lower.index('removed')] if 'removed' in columns_lower else None
        
        for _, row in df.iterrows():
            try:
                row_date = pd.to_datetime(row[date_col]).date()
                
                if added_col and pd.notna(row[added_col]):
                    added_symbols = str(row[added_col]).split(',')
                    for symbol in added_symbols:
                        symbol = symbol.strip()
                        if symbol:
                            records.append({
                                'date': row_date,
                                'symbol': symbol,
                                'action': 'added'
                            })
                
                if removed_col and pd.notna(row[removed_col]):
                    removed_symbols = str(row[removed_col]).split(',')
                    for symbol in removed_symbols:
                        symbol = symbol.strip()
                        if symbol:
                            records.append({
                                'date': row_date,
                                'symbol': symbol,
                                'action': 'removed'
                            })
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
    
    else:
        logger.warning(f"Unknown GitHub data format. Columns: {list(df.columns)}")
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Fetch historical S&P 500 constituents data")
    parser.add_argument("--output", type=str, default="data/sp500_constituents.csv", 
                        help="Output CSV path (default: data/sp500_constituents.csv)")
    parser.add_argument("--source", type=str, choices=['github', 'wikipedia', 'fallback', 'auto'],
                        default='auto', help="Data source (default: auto)")
    parser.add_argument("--start-date", type=str, help="Start date for constituents (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(columns=['date', 'symbol', 'action'])
    
    if args.source == 'auto' or args.source == 'github':
        # Try GitHub first
        github_df = fetch_from_github_fja05680()
        if len(github_df) > 0:
            result_df = process_github_data(github_df)
    
    if len(result_df) == 0 and (args.source == 'auto' or args.source == 'wikipedia'):
        # Try Wikipedia for current constituents
        current_symbols = fetch_current_from_wikipedia()
        if len(current_symbols) > 0:
            # Create a simple file with current constituents as of today
            today = date.today()
            # Assume they were all added at start_date or 2010-01-01
            start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else date(2010, 1, 1)
            
            records = [{'date': start, 'symbol': s, 'action': 'added'} for s in current_symbols]
            result_df = pd.DataFrame(records)
    
    if len(result_df) == 0 or args.source == 'fallback':
        # Use fallback
        result_df = create_fallback_constituents()
    
    if len(result_df) == 0:
        logger.error("Failed to fetch any S&P 500 constituents data")
        return
    
    # Clean up
    result_df['symbol'] = result_df['symbol'].str.strip().str.upper()
    result_df['symbol'] = result_df['symbol'].str.replace('.', '-', regex=False)  # BRK.B -> BRK-B
    result_df = result_df.drop_duplicates()
    result_df = result_df.sort_values(['date', 'symbol'])
    
    # Save
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(result_df)} records to {output_path}")
    
    # Summary
    logger.info("\nSummary:")
    logger.info(f"  Total records: {len(result_df)}")
    logger.info(f"  Unique symbols: {result_df['symbol'].nunique()}")
    logger.info(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    logger.info(f"  Additions: {len(result_df[result_df['action'] == 'added'])}")
    logger.info(f"  Removals: {len(result_df[result_df['action'] == 'removed'])}")


if __name__ == "__main__":
    main()

