import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE
}


class SqlToolError(Exception):
    """A query was rejected or failed. The message is safe to show to the LLM."""


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


def authorizer(action, arg1, arg2, db_name, source) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_query(
    sql: str,
    *,
    db_path: Path,
    max_rows: int = 50,
    timeout_s: float = 2.0
) -> QueryResult:
    statement = sql.strip().rstrip(";").strip()
    if not statement.lower().startswith(("select", "with")):
        raise SqlToolError("Only SELECT queries are allowed.")

    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.set_authorizer(authorizer)
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        try:
            cursor = conn.execute(statement)
            fetched = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            raise SqlToolError(f"Query failed: {exc}") from exc
        columns = [column[0] for column in cursor.description]
    finally:
        conn.close()

    return QueryResult(
        columns=columns,
        rows=[list(row) for row in fetched[:max_rows]],
        truncated=len(fetched) > max_rows,
    )