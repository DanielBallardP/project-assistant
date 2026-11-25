import sqlite3
from pathlib import Path

import pytest

from project_assistant.sql_tool import SqlToolError, run_query


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE projekte (id TEXT PRIMARY KEY, name TEXT NOT NULL);
        INSERT INTO projekte VALUES ('P-1', 'Alpha'), ('P-2', 'Beta'), ('P-3', 'Gamma');
        """
    )
    conn.commit()
    conn.close()
    return path


def count_projects(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT count(*) FROM projekte").fetchone()[0]
    finally:
        conn.close()


def test_returns_columns_and_rows(db_path):
    result = run_query("SELECT id, name FROM projekte ORDER BY id", db_path=db_path)

    assert result.columns == ["id", "name"]
    assert result.rows == [["P-1", "Alpha"], ["P-2", "Beta"], ["P-3", "Gamma"]]
    assert result.truncated is False


def test_accepts_cte_and_trailing_semicolon(db_path):
    result = run_query(
        "WITH p AS (SELECT id FROM projekte) SELECT count(*) FROM p;", db_path=db_path
    )

    assert result.rows == [[3]]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM projekte",
        "UPDATE projekte SET name = 'x'",
        "INSERT INTO projekte VALUES ('P-9', 'Neu')",
        "DROP TABLE projekte",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA table_info(projekte)",
    ],
)
def test_rejects_statements_that_are_not_select(db_path, sql):
    with pytest.raises(SqlToolError, match="Only SELECT"):
        run_query(sql, db_path=db_path)

    assert count_projects(db_path) == 3


def test_database_blocks_a_write_hidden_inside_a_cte(db_path):
    with pytest.raises(SqlToolError, match="not authorized"):
        run_query("WITH x AS (SELECT 1) DELETE FROM projekte", db_path=db_path)

    assert count_projects(db_path) == 3


def test_database_blocks_pragma_functions_inside_select(db_path):
    with pytest.raises(SqlToolError, match="not authorized"):
        run_query("SELECT * FROM pragma_table_info('projekte')", db_path=db_path)


def test_rejects_multiple_statements(db_path):
    with pytest.raises(SqlToolError):
        run_query("SELECT 1; DROP TABLE projekte", db_path=db_path)

    assert count_projects(db_path) == 3


def test_truncates_at_max_rows(db_path):
    result = run_query("SELECT id FROM projekte", db_path=db_path, max_rows=2)

    assert len(result.rows) == 2
    assert result.truncated is True


def test_exactly_max_rows_is_not_truncated(db_path):
    result = run_query("SELECT id FROM projekte", db_path=db_path, max_rows=3)

    assert len(result.rows) == 3
    assert result.truncated is False


def test_aborts_runaway_query(db_path):
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )

    with pytest.raises(SqlToolError, match="interrupted"):
        run_query(runaway, db_path=db_path, timeout_s=0.2)


def test_sql_errors_are_reported_so_the_llm_can_correct_itself(db_path):
    with pytest.raises(SqlToolError, match="no such column"):
        run_query("SELECT nope FROM projekte", db_path=db_path)