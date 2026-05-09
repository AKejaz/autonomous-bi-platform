"""
core/database.py

Handles all database connectivity. Supports both local PostgreSQL and
Supabase (hosted Postgres). Uses SQLAlchemy under the hood so the rest
of the platform stays database-agnostic.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Thin wrapper around SQLAlchemy that gives the rest of the app
    a clean interface for running queries and introspecting the schema.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError(
                "No database URL found. Set DATABASE_URL in your .env file."
            )
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,   # detect stale connections
                pool_recycle=300,     # recycle every 5 minutes
                connect_args={"connect_timeout": 10},
            )
        return self._engine

    def test_connection(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def run_query(self, sql: str, params: dict = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.
        All agent queries funnel through here.
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            logger.error(f"Query failed: {e}\nSQL: {sql}")
            raise

    def get_schema_info(self) -> dict:
        """
        Introspect the database and return table/column metadata.
        Fed to the SQL agent as context so it can build accurate queries.
        """
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)
            schema[table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                    }
                    for col in columns
                ],
                "primary_keys": pk.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "column": fk["constrained_columns"],
                        "references": f"{fk['referred_table']}.{fk['referred_columns']}",
                    }
                    for fk in fks
                ],
            }
        return schema

    def get_schema_as_text(self) -> str:
        """
        Human-readable schema string fed into the SQL agent's system prompt.
        """
        schema = self.get_schema_info()
        lines = ["DATABASE SCHEMA\n" + "=" * 60]
        for table, meta in schema.items():
            lines.append(f"\nTABLE: {table}")
            lines.append("  Columns:")
            for col in meta["columns"]:
                nullable = "" if col["nullable"] else " NOT NULL"
                lines.append(f"    - {col['name']} ({col['type']}){nullable}")
            if meta["primary_keys"]:
                lines.append(f"  Primary Key: {', '.join(meta['primary_keys'])}")
            if meta["foreign_keys"]:
                for fk in meta["foreign_keys"]:
                    lines.append(
                        f"  Foreign Key: {fk['column']} → {fk['references']}"
                    )
        return "\n".join(lines)

    def get_table_sample(self, table_name: str, n: int = 5) -> pd.DataFrame:
        return self.run_query(f"SELECT * FROM {table_name} LIMIT {n}")


@lru_cache(maxsize=1)
def get_db() -> DatabaseConnection:
    """
    Singleton accessor. Cached so we reuse the connection pool
    across all agents within a session.
    """
    return DatabaseConnection()
