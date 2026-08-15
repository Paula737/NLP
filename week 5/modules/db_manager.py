"""
modules/db_manager.py
Loads an uploaded CSV/Excel file into SQLite, and exposes schema info
for the SQL-generation step.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "./data/analysis.db"
TABLE_NAME = "dataset"


def load_file_to_dataframe(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .xlsx, or .xls")


def store_dataframe_in_sqlite(df: pd.DataFrame, table_name: str = TABLE_NAME) -> dict:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
    conn.close()

    return {"table_name": table_name, "row_count": len(df), "columns": columns}


def get_current_schema(table_name: str = TABLE_NAME) -> dict:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("No dataset has been uploaded yet.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]

    if not columns:
        conn.close()
        raise ValueError(f"Table '{table_name}' does not exist. Upload a dataset first.")

    # Sample a few distinct values for TEXT columns, so the SQL generator
    # can match the user's spoken words to actual stored values
    # (handles language mismatches, casing, abbreviations, etc.)
    for col in columns:
        if col["type"] == "TEXT":
            sample_cursor = conn.execute(
                f'SELECT DISTINCT "{col["name"]}" FROM {table_name} LIMIT 10'
            )
            col["sample_values"] = [row[0] for row in sample_cursor.fetchall()]

    conn.close()
    return {"table_name": table_name, "columns": columns}