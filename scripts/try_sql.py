"""Usage: uv run python scripts/try_sql.py "SELECT ..."  -> runs one read-only query"""

import sys
from pathlib import Path

from project_assistant.sql_tool import SqlToolError, run_query

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "project.db"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    try:
        result = run_query(sys.argv[1], db_path=DB_PATH)
    except SqlToolError as exc:
        raise SystemExit(f"Rejected: {exc}")
    print(" | ".join(result.columns))
    for row in result.rows:
        print(" | ".join("NULL" if value is None else str(value) for value in row))
    if result.truncated:
        print("... (truncated)")


if __name__ == "__main__":
    main()