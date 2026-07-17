"""
MCP server that exposes a SQL/CSV query tool.

Any MCP-compatible client (including CrewAI agents via MCPServerAdapter)
can connect to this server over stdio and call the exposed tools whenever
they need to run a SQL query against a CSV file or a SQLite database.

Run standalone for a quick manual check:
    python mcp_server.py
(it will just idle, waiting for a client to connect over stdio)
"""

import sqlite3
from pathlib import Path
from typing import Optional

import duckdb
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sql-csv-query-server")


# ---------------------------------------------------------------------------
# Tool 1: run a SQL query against one or more CSV files
# ---------------------------------------------------------------------------
@mcp.tool()
def query_csv(sql_query: str, csv_path: str, table_alias: str = "data") -> str:
    """
    Run a SQL query against a CSV file using DuckDB.

    Args:
        sql_query: The SQL query to run. Reference the CSV using the name
            given in `table_alias` (default: "data"), e.g.
            "SELECT category, SUM(sales) FROM data GROUP BY category".
        csv_path: Absolute or relative path to the CSV file to query.
        table_alias: The table name to use for the CSV inside the SQL
            query. Defaults to "data".

    Returns:
        The query result formatted as a markdown table, or an error
        message if the query/path is invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        return f"Error: CSV file not found at '{csv_path}'."

    try:
        conn = duckdb.connect(database=":memory:")
        conn.execute(
            f"CREATE VIEW {table_alias} AS SELECT * FROM read_csv_auto(?)",
            [str(path)],
        )
        result_df = conn.execute(sql_query).fetchdf()
        conn.close()

        if result_df.empty:
            return "Query ran successfully but returned no rows."

        return result_df.to_markdown(index=False)

    except Exception as exc:
        return f"Error running query: {exc}"


# ---------------------------------------------------------------------------
# Tool 2: run a SQL query against a SQLite database file
# ---------------------------------------------------------------------------
@mcp.tool()
def query_sqlite(sql_query: str, db_path: str) -> str:
    """
    Run a read-only SQL query against a SQLite database file.

    Args:
        sql_query: The SQL query to run (e.g. "SELECT * FROM customers LIMIT 10").
        db_path: Path to the .db / .sqlite file.

    Returns:
        The query result formatted as a markdown table, or an error message.
    """
    path = Path(db_path)
    if not path.exists():
        return f"Error: database file not found at '{db_path}'."

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cursor = conn.execute(sql_query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query ran successfully but returned no rows."

        header = " | ".join(columns)
        separator = " | ".join(["---"] * len(columns))
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{separator}\n{body}"

    except Exception as exc:
        return f"Error running query: {exc}"


# ---------------------------------------------------------------------------
# Tool 3: inspect schema / columns before writing a query
# ---------------------------------------------------------------------------
@mcp.tool()
def describe_source(source_path: str) -> str:
    """
    Inspect a CSV or SQLite file and return its schema (table/column names
    and types) so an agent can write a correct SQL query.

    Args:
        source_path: Path to a .csv, .db, or .sqlite file.

    Returns:
        A human-readable description of the schema, or an error message.
    """
    path = Path(source_path)
    if not path.exists():
        return f"Error: file not found at '{source_path}'."

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            conn = duckdb.connect(database=":memory:")
            conn.execute("CREATE VIEW data AS SELECT * FROM read_csv_auto(?)", [str(path)])
            schema_df = conn.execute("DESCRIBE data").fetchdf()
            conn.close()
            return "CSV columns (table alias: data):\n" + schema_df.to_markdown(index=False)

        elif suffix in (".db", ".sqlite", ".sqlite3"):
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()

            lines = []
            for (table_name,) in tables:
                cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
                lines.append(f"- {table_name}: {col_desc}")

            conn.close()
            return "Tables found:\n" + "\n".join(lines) if lines else "No tables found."

        else:
            return f"Error: unsupported file type '{suffix}'. Use .csv, .db, or .sqlite."

    except Exception as exc:
        return f"Error describing source: {exc}"


if __name__ == "__main__":
    # stdio transport: the client (CrewAI's MCPServerAdapter) launches this
    # script as a subprocess and talks to it over stdin/stdout.
    mcp.run(transport="stdio")
