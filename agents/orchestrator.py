"""
agents/orchestrator.py

The Orchestrator — Multi-Agent Pipeline Controller

This is the conductor. It sequences SQL Agent → Analyst Agent → Advisor Agent,
passes outputs between them, handles errors at each stage gracefully,
and assembles the final structured response that the UI consumes.

Design principle: each agent can fail independently without crashing the
entire pipeline. If the SQL agent can't run a query, the user gets a clear
message. If the analyst fails, the raw data is still shown.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from agents.sql_agent import SQLAgent
from agents.analyst_agent import AnalystAgent
from agents.advisor_agent import AdvisorAgent
from core.database import DatabaseConnection
from core.memory import PlatformMemory

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Unified response object returned by the orchestrator.
    Every field has a sensible default so partial failures don't crash the UI.
    """
    # Input
    user_question: str = ""
    role: str = "executive"

    # SQL Agent output
    sql_generated: str = ""
    dataframe: Optional[pd.DataFrame] = None
    row_count: int = 0
    sql_error: Optional[str] = None

    # Analyst Agent output
    analysis: str = ""
    headline: str = ""
    anomaly_detected: bool = False
    stats_summary: str = ""

    # Advisor Agent output
    recommendations: str = ""
    priority_level: str = "low"
    action_count: int = 0

    # Pipeline metadata
    execution_time_sec: float = 0.0
    memory_id: Optional[str] = None
    pipeline_stages_completed: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.sql_error is None and self.dataframe is not None

    @property
    def has_analysis(self) -> bool:
        return bool(self.analysis)

    @property
    def has_recommendations(self) -> bool:
        return bool(self.recommendations)


class AgentOrchestrator:
    """
    Coordinates the three-agent pipeline. Called by the Streamlit UI and
    the anomaly monitor alike.
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

        self.sql_agent = SQLAgent(db, memory, groq_api_key, model)
        self.analyst_agent = AnalystAgent(memory, groq_api_key, model)
        self.advisor_agent = AdvisorAgent(memory, groq_api_key, model)

    def run(self, question: str, role: str = "executive") -> PipelineResult:
        """
        Full three-agent pipeline. Returns a PipelineResult regardless of
        whether individual stages succeed or fail.
        """
        start_time = time.time()
        result = PipelineResult(user_question=question, role=role)

        logger.info(f"Pipeline started | role={role} | question={question[:80]}")

        # ── Stage 1: SQL Agent ──────────────────────────────────────────────
        try:
            sql_output = self.sql_agent.generate_and_run(question)
            result.sql_generated = sql_output["sql"]
            result.dataframe = sql_output["dataframe"]
            result.row_count = sql_output["row_count"]
            result.sql_error = sql_output["error"]
            result.pipeline_stages_completed.append("sql_agent")

            if result.sql_error:
                logger.warning(f"SQL Agent failed: {result.sql_error}")
                result.execution_time_sec = time.time() - start_time
                return result

        except Exception as e:
            result.sql_error = f"SQL Agent encountered an unexpected error: {e}"
            result.execution_time_sec = time.time() - start_time
            logger.error(result.sql_error)
            return result

        # ── Stage 2: Analyst Agent ──────────────────────────────────────────
        try:
            analysis_output = self.analyst_agent.analyze(
                user_question=question,
                sql_used=result.sql_generated,
                df=result.dataframe,
            )
            result.analysis = analysis_output["analysis"]
            result.headline = analysis_output["headline"]
            result.anomaly_detected = analysis_output["anomaly_detected"]
            result.stats_summary = analysis_output["stats_summary"]
            result.pipeline_stages_completed.append("analyst_agent")

        except Exception as e:
            logger.error(f"Analyst Agent failed: {e}")
            result.analysis = f"Analysis could not be completed: {e}"
            result.headline = "Analysis unavailable."

        # ── Stage 3: Advisor Agent ──────────────────────────────────────────
        try:
            advisor_output = self.advisor_agent.advise(
                user_question=question,
                analysis=result.analysis,
                headline=result.headline,
                role=role,
                anomaly_detected=result.anomaly_detected,
            )
            result.recommendations = advisor_output["recommendations"]
            result.priority_level = advisor_output["priority_level"]
            result.action_count = advisor_output["action_count"]
            result.pipeline_stages_completed.append("advisor_agent")

        except Exception as e:
            logger.error(f"Advisor Agent failed: {e}")
            result.recommendations = f"Recommendations could not be generated: {e}"

        # ── Persist to memory ───────────────────────────────────────────────
        try:
            mem_id = self.memory.store_interaction(
                user_query=question,
                sql_generated=result.sql_generated,
                analysis=result.analysis,
                recommendation=result.recommendations,
                role=role,
            )
            result.memory_id = mem_id
        except Exception as e:
            logger.warning(f"Memory storage failed (non-fatal): {e}")

        result.execution_time_sec = round(time.time() - start_time, 2)
        logger.info(
            f"Pipeline complete | stages={result.pipeline_stages_completed} | "
            f"time={result.execution_time_sec}s | anomaly={result.anomaly_detected}"
        )
        return result

    def run_sql_only(self, question: str) -> PipelineResult:
        """Lightweight version — just SQL + data, no analysis. Used for dashboard widgets."""
        result = PipelineResult(user_question=question)
        try:
            sql_output = self.sql_agent.generate_and_run(question)
            result.sql_generated = sql_output["sql"]
            result.dataframe = sql_output["dataframe"]
            result.row_count = sql_output["row_count"]
            result.sql_error = sql_output["error"]
        except Exception as e:
            result.sql_error = str(e)
        return result
