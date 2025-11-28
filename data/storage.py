"""Storage backend using DuckDB and Parquet."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq


class StorageBackend:
    """DuckDB + Parquet storage backend."""
    
    def __init__(self, db_path: str = "data/market.duckdb", data_root: str = "data"):
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.raw_vendor_dir = self.data_root / "raw_vendor"
        self.normalized_dir = self.data_root / "normalized"
        
        # Create directories
        self.raw_vendor_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema with table definitions."""
        # Create assets table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                asset_id INTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                exchange VARCHAR,
                currency VARCHAR DEFAULT 'USD',
                first_trade_date DATE,
                last_trade_date DATE,
                sector VARCHAR,
                industry VARCHAR,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(symbol)
            )
        """)
        
        # Create bars_daily table (will be a view over Parquet)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bars_daily (
                asset_id INTEGER NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                data_vendor VARCHAR,
                ingestion_timestamp TIMESTAMP,
                PRIMARY KEY (asset_id, date)
            )
        """)
        
        # Create corporate_actions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                asset_id INTEGER NOT NULL,
                date DATE NOT NULL,
                split_factor DOUBLE DEFAULT 1.0,
                dividend_amount DOUBLE DEFAULT 0.0,
                special_dividend DOUBLE DEFAULT 0.0,
                PRIMARY KEY (asset_id, date)
            )
        """)
        
        # Create fundamentals table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                asset_id INTEGER NOT NULL,
                period_end_date DATE NOT NULL,
                report_release_date DATE NOT NULL,
                eps DOUBLE,
                eps_estimate DOUBLE,
                revenue DOUBLE,
                ebitda DOUBLE,
                total_debt DOUBLE,
                total_equity DOUBLE,
                free_cash_flow DOUBLE,
                PRIMARY KEY (asset_id, period_end_date)
            )
        """)
        
        # Create news_events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_events (
                asset_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                headline VARCHAR,
                source VARCHAR,
                url VARCHAR,
                vendor_sentiment_score DOUBLE,
                PRIMARY KEY (asset_id, timestamp)
            )
        """)
        
        # Create universe_membership table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS universe_membership (
                date DATE NOT NULL,
                asset_id INTEGER NOT NULL,
                index_name VARCHAR NOT NULL,
                in_index BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (date, asset_id, index_name)
            )
        """)
    
    def save_parquet(self, df: pd.DataFrame, table_name: str, partition_by: Optional[List[str]] = None):
        """Save DataFrame to Parquet file."""
        parquet_path = self.normalized_dir / f"{table_name}.parquet"
        
        if partition_by and len(partition_by) > 0:
            # Partitioned write
            df.to_parquet(
                parquet_path.parent / table_name,
                partition_cols=partition_by,
                engine='pyarrow',
                index=False
            )
        else:
            # Single file write
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    def load_parquet(self, table_name: str) -> pd.DataFrame:
        """Load DataFrame from Parquet file."""
        parquet_path = self.normalized_dir / f"{table_name}.parquet"
        
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif (self.normalized_dir / table_name).is_dir():
            # Partitioned table
            return pd.read_parquet(self.normalized_dir / table_name)
        else:
            return pd.DataFrame()
    
    def register_parquet_view(self, table_name: str, parquet_path: Optional[str] = None):
        """Register a Parquet file as a DuckDB view."""
        if parquet_path is None:
            parquet_path = str(self.normalized_dir / f"{table_name}.parquet")
        
        view_name = f"{table_name}_view"
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.conn.execute(sql).df()
    
    def insert_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = "append"):
        """Insert DataFrame into DuckDB table."""
        if if_exists == "replace":
            self.conn.execute(f"DELETE FROM {table_name}")
        
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df", parameters={"df": df})
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()

