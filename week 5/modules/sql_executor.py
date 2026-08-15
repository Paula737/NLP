"""
modules/sql_executor.py
Validates a SQL query (read-only, single-statement, correct table)
before executing it against SQLite. This is the safety gate between
LLM-generated SQL and your actual database.
"""

import sqlite3
import re
from modules.db_manager import DB_PATH, TABLE_NAME

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter",
    "create", "truncate", "attach", "detach", "pragma",
]


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    """
    sql_lower = sql.lower().strip()

    if not sql_lower.startswith("select"):
        return False, "Only SELECT queries are allowed."

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_lower):
            return False, f"Query contains a forbidden keyword: '{keyword}'."

    if ";" in sql.strip().rstrip(";"):
        return False, "Multiple statements are not allowed."

    return True, ""


def execute_sql(sql: str) -> dict:
    """
    Validates and executes the SQL, returns rows as list of dicts.
    """
    is_valid, error = validate_sql(sql)
    if not is_valid:
        return {"success": False, "error": error, "rows": [], "columns": []}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        return {"success": True, "error": None, "rows": rows, "columns": columns}
    except sqlite3.Error as e:
        return {"success": False, "error": str(e), "rows": [], "columns": []}