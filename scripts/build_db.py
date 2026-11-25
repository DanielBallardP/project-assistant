"""Usage: uv run python scripts/build_db.py  -> builds data/project.db"""

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "data" / "schema.sql"
CSV_DIR = ROOT / "data" / "sql"
DB_PATH = ROOT / "data" / "project.db"

TABLES = ["kunden", "projekte", "stunden", "material", "offene_punkte"]


def read_rows(table: str) -> list[dict]:
    with (CSV_DIR / f"{table}.csv").open(encoding="utf-8", newline="") as file:
        # An empty CSV cell means NULL (for example offene_punkte.faellig_am).
        return [
            {key: value or None for key, value in row.items()} for row in csv.DictReader(file)
        ]


def main() -> None:
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        for table in TABLES:
            rows = read_rows(table)
            columns = list(rows[0])
            placeholders = ", ".join(f":{column}" for column in columns)
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                rows,
            )
            print(f"{table}: {len(rows)} rows")
        conn.commit()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise SystemExit(f"Foreign key violations: {violations}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()