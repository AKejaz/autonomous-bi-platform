"""
agents/sql_agent.py

Agent 1 — The SQL Agent

Responsibility: translate a plain-English business question into a valid,
safe SQL query, execute it against the live database, and return a clean
DataFrame along with the generated SQL for transparency.

Uses Groq's LLaMA 3.3 70b model via LangChain. The schema is injected
into the system prompt so the model knows exactly what tables and columns
exist — no hallucinated column names.
"""

import logging
import re
from typing import Optional

import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from core.database import DatabaseConnection
from core.memory import PlatformMemory

logger = logging.getLogger(__name__)


SQL_SYSTEM_PROMPT = """You are an expert SQL analyst embedded in an enterprise business intelligence platform.

Your ONLY job is to translate natural language questions into safe, correct PostgreSQL queries.

STRICT RULES:
1. Return ONLY the raw SQL query — no markdown fences, no explanations, no commentary.
2. Never use DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER, or any write operations.
3. Always use table aliases for readability.
4. Limit results to 500 rows maximum unless the user explicitly asks for more.
5. Use meaningful column aliases (e.g., AVG(salary) AS avg_salary).
6. When filtering by string values, use ILIKE for case-insensitive matching.
7. Always handle NULLs gracefully — use COALESCE where appropriate.
8. If the question cannot be answered with the available schema, return exactly: CANNOT_ANSWER

Available schema:
{schema}

{context}
"""


class SQLAgent:
    """
    Converts natural language questions to executable SQL and runs them.
    The generated SQL is always exposed in the response so users can
    audit what was actually run — transparency is non-negotiable in
    enterprise settings.
    """

    def __init__(
        self,
        db: DatabaseConnection,
        memory: PlatformMemory,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ):
        self.db = db
        self.memory = memory
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=0,          # deterministic for SQL generation
            max_tokens=1024,
        )
        self._schema_cache: Optional[str] = None

    def _get_schema(self) -> str:
        if not self._schema_cache:
            self._schema_cache = self.db.get_schema_as_text()
        return self._schema_cache

    def _build_system_prompt(self, user_question: str) -> str:
        schema = self._get_schema()
        context = self.memory.get_context_summary(user_question)
        return SQL_SYSTEM_PROMPT.format(schema=schema, context=context)

    def _clean_sql(self, raw: str) -> str:
        """Strip any markdown fences the model might add despite instructions."""
        cleaned = re.sub(r"```sql\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)
        return cleaned.strip()

    def generate_and_run(self, question: str) -> dict:
        """
        Main entry point. Returns a dict with keys:
          - sql          : the generated query
          - dataframe    : results as a pandas DataFrame (or None)
          - row_count    : number of rows returned
          - error        : error message if something went wrong (or None)
          - answerable   : bool — False if schema can't answer the question
        """
        logger.info(f"SQL Agent processing: {question}")

        system_prompt = self._build_system_prompt(question)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_sql = response.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._error_result(str(e))

        if raw_sql == "CANNOT_ANSWER":
            return {
                "sql": "",
                "dataframe": None,
                "row_count": 0,
                "error": "This question cannot be answered with the available data.",
                "answerable": False,
            }

        sql = self._clean_sql(raw_sql)
        logger.debug(f"Generated SQL:\n{sql}")

        try:
            df = self.db.run_query(sql)
            return {
                "sql": sql,
                "dataframe": df,
                "row_count": len(df),
                "error": None,
                "answerable": True,
            }
        except Exception as e:
            # Second attempt: feed the error back and ask for a correction
            logger.warning(f"First SQL attempt failed: {e}. Trying self-correction.")
            return self._self_correct(question, sql, str(e), system_prompt)

    def _self_correct(
        self, question: str, bad_sql: str, error_msg: str, system_prompt: str
    ) -> dict:
        """
        If the first query errors out, give the model the error message and
        ask it to try again. One retry only — avoids infinite loops.
        """
        correction_prompt = (
            f"The following SQL query produced an error.\n\n"
            f"Original question: {question}\n\n"
            f"Failed SQL:\n{bad_sql}\n\n"
            f"Error: {error_msg}\n\n"
            f"Please write a corrected SQL query that fixes this error. "
            f"Return ONLY the SQL — no explanation."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=correction_prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            corrected_sql = self._clean_sql(response.content.strip())
            df = self.db.run_query(corrected_sql)
            logger.info("Self-correction succeeded.")
            return {
                "sql": corrected_sql,
                "dataframe": df,
                "row_count": len(df),
                "error": None,
                "answerable": True,
            }
        except Exception as e:
            return self._error_result(f"Self-correction also failed: {e}")

    def _error_result(self, msg: str) -> dict:
        return {
            "sql": "",
            "dataframe": None,
            "row_count": 0,
            "error": msg,
            "answerable": False,
        }
