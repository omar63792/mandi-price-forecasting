"""DuckDB storage.

DuckDB is a database that lives in a single file (data/mandi.duckdb), like
SQLite, but is built for analytics. No server to install: `pip install duckdb`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from mandi.config import DB_PATH, SQL_DIR


def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (or create) the database and make sure tables and views exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    for sql_file in sorted(SQL_DIR.glob("*.sql")):  # 01_schema.sql, then 02_views.sql
        con.execute(sql_file.read_text())
    return con


def upsert_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert cleaned rows. A row that already exists (same primary key) is replaced,
    so loading the same file twice is safe. This property is called 'idempotent'."""
    df = df.copy()
    # Primary-key columns cannot be NULL in DuckDB, so use '' for missing variety/grade.
    for col in ["variety", "grade"]:
        df[col] = df[col].fillna("")
    if "source_file" not in df.columns:
        df["source_file"] = None
    con.register("incoming", df)
    con.execute("""
        INSERT OR REPLACE INTO raw_prices
            (state, district, market, commodity, variety, grade, price_date,
             min_price, max_price, modal_price, arrivals_tonnes, quality_flag, source_file)
        SELECT state, district, market, commodity, variety, grade, price_date,
               min_price, max_price, modal_price, arrivals_tonnes, quality_flag, source_file
        FROM incoming
    """)
    con.unregister("incoming")
    return len(df)


def load_daily(con: duckdb.DuckDBPyConnection, level: str = "state") -> pd.DataFrame:
    """Read the state-level or national daily view into pandas."""
    view = {"state": "state_daily", "national": "national_daily"}[level]
    return con.execute(f"SELECT * FROM {view} ORDER BY state, commodity, price_date").df()
