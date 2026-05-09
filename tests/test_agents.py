"""
tests/test_agents.py

Unit tests for the agent pipeline. Uses mocked database and LLM calls
so they run without real credentials — safe for CI environments.

Run with: pytest tests/ -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Realistic HR dataset sample for testing."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "employee_number": range(1, n + 1),
        "department": rng.choice(["Sales", "HR", "R&D", "Finance"], n),
        "job_role": rng.choice(["Manager", "Analyst", "Engineer", "Executive"], n),
        "monthly_income": rng.integers(3000, 20000, n),
        "job_satisfaction": rng.integers(1, 5, n),
        "performance_rating": rng.integers(1, 4, n),
        "years_at_company": rng.integers(0, 20, n),
        "attrition": rng.choice([True, False], n, p=[0.16, 0.84]),
    })


@pytest.fixture
def mock_db(sample_df):
    db = MagicMock()
    db.run_query.return_value = sample_df
    db.get_schema_as_text.return_value = """
TABLE: employees
  Columns:
    - employee_number (INTEGER)
    - department (VARCHAR)
    - monthly_income (NUMERIC)
    - job_satisfaction (INTEGER)
    - performance_rating (INTEGER)
    - years_at_company (INTEGER)
    - attrition (BOOLEAN)
"""
    db.test_connection.return_value = True
    return db


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory.get_context_summary.return_value = "No prior context available."
    memory.get_relevant_history.return_value = []
    memory.store_interaction.return_value = "mock-interaction-id"
    memory.get_stats.return_value = {
        "total_interactions": 5,
        "total_anomalies": 2,
        "total_insights": 1,
    }
    return memory


# ── SQL Agent Tests ────────────────────────────────────────────────────────────

class TestSQLAgent:

    @patch("agents.sql_agent.ChatGroq")
    def test_successful_query(self, mock_groq_cls, mock_db, mock_memory, sample_df):
        from agents.sql_agent import SQLAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT department, AVG(monthly_income) AS avg_salary FROM employees GROUP BY department"
        )
        mock_groq_cls.return_value = mock_llm

        agent = SQLAgent(mock_db, mock_memory, groq_api_key="test-key")
        result = agent.generate_and_run("What is average salary by department?")

        assert result["error"] is None
        assert result["answerable"] is True
        assert result["sql"] != ""
        assert isinstance(result["dataframe"], pd.DataFrame)
        assert result["row_count"] == len(sample_df)

    @patch("agents.sql_agent.ChatGroq")
    def test_cannot_answer_response(self, mock_groq_cls, mock_db, mock_memory):
        from agents.sql_agent import SQLAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="CANNOT_ANSWER")
        mock_groq_cls.return_value = mock_llm

        agent = SQLAgent(mock_db, mock_memory, groq_api_key="test-key")
        result = agent.generate_and_run("What is the CEO's mood today?")

        assert result["answerable"] is False
        assert result["dataframe"] is None
        assert result["error"] is not None

    @patch("agents.sql_agent.ChatGroq")
    def test_sql_cleaning_strips_fences(self, mock_groq_cls, mock_db, mock_memory, sample_df):
        from agents.sql_agent import SQLAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="```sql\nSELECT * FROM employees LIMIT 10\n```"
        )
        mock_groq_cls.return_value = mock_llm

        agent = SQLAgent(mock_db, mock_memory, groq_api_key="test-key")
        result = agent.generate_and_run("Show me some employees")

        assert "```" not in result["sql"]
        assert result["sql"].strip().upper().startswith("SELECT")

    @patch("agents.sql_agent.ChatGroq")
    def test_self_correction_on_db_error(self, mock_groq_cls, mock_db, mock_memory, sample_df):
        from agents.sql_agent import SQLAgent

        mock_llm = MagicMock()
        # First call returns bad SQL, second call returns corrected
        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT bad_column FROM employees"),
            MagicMock(content="SELECT employee_number FROM employees LIMIT 10"),
        ]
        mock_groq_cls.return_value = mock_llm

        # First DB call fails, second succeeds
        mock_db.run_query.side_effect = [
            Exception("column bad_column does not exist"),
            sample_df.head(10),
        ]

        agent = SQLAgent(mock_db, mock_memory, groq_api_key="test-key")
        result = agent.generate_and_run("Show employee numbers")

        assert result["error"] is None
        assert result["dataframe"] is not None


# ── Analyst Agent Tests ────────────────────────────────────────────────────────

class TestAnalystAgent:

    @patch("agents.analyst_agent.ChatGroq")
    def test_empty_dataframe_handling(self, mock_groq_cls, mock_memory):
        from agents.analyst_agent import AnalystAgent

        agent = AnalystAgent(mock_memory, groq_api_key="test-key")
        result = agent.analyze("test question", "SELECT 1", pd.DataFrame())

        assert result["anomaly_detected"] is False
        assert "no data" in result["analysis"].lower()

    @patch("agents.analyst_agent.ChatGroq")
    def test_anomaly_detection_from_keywords(self, mock_groq_cls, mock_memory, sample_df):
        from agents.analyst_agent import AnalystAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="KEY FINDING: A significant anomaly was detected in the Sales department. "
                    "Revenue shows an unusual spike of 3.2 standard deviations above the mean. "
                    "This is concerning and requires immediate attention."
        )
        mock_groq_cls.return_value = mock_llm

        agent = AnalystAgent(mock_memory, groq_api_key="test-key")
        result = agent.analyze("Check sales anomalies", "SELECT ...", sample_df)

        assert result["anomaly_detected"] is True

    @patch("agents.analyst_agent.ChatGroq")
    def test_stats_computation(self, mock_groq_cls, mock_memory, sample_df):
        from agents.analyst_agent import AnalystAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Normal distribution across all metrics.")
        mock_groq_cls.return_value = mock_llm

        agent = AnalystAgent(mock_memory, groq_api_key="test-key")
        stats = agent.quick_stats(sample_df)

        assert "monthly_income" in stats
        assert stats["monthly_income"]["mean"] > 0
        assert "outlier_count" in stats["monthly_income"]

    def test_compute_stats_empty(self, mock_memory):
        from agents.analyst_agent import AnalystAgent
        with patch("agents.analyst_agent.ChatGroq"):
            agent = AnalystAgent(mock_memory, groq_api_key="test-key")
            result = agent._compute_stats(pd.DataFrame())
            assert "No data" in result


# ── Advisor Agent Tests ────────────────────────────────────────────────────────

class TestAdvisorAgent:

    @patch("agents.advisor_agent.ChatGroq")
    def test_executive_advisory(self, mock_groq_cls, mock_memory):
        from agents.advisor_agent import AdvisorAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="1. STRATEGIC SUMMARY\nImmediate action required on attrition.\n"
                    "2. PRIORITY ACTIONS\nConduct exit interviews this week.\n"
                    "3. RISK FACTORS\nTalent pipeline at risk.\n"
                    "4. WATCH LIST\nMonitor weekly attrition rate."
        )
        mock_groq_cls.return_value = mock_llm

        agent = AdvisorAgent(mock_memory, groq_api_key="test-key")
        result = agent.advise(
            user_question="What is attrition rate?",
            analysis="Attrition is 18%, above industry average.",
            headline="Attrition at 18% — above benchmark.",
            role="executive",
            anomaly_detected=False,
        )

        assert result["recommendations"] != ""
        assert result["priority_level"] in ("high", "medium", "low")
        assert result["action_count"] >= 1

    @patch("agents.advisor_agent.ChatGroq")
    def test_high_priority_on_anomaly(self, mock_groq_cls, mock_memory):
        from agents.advisor_agent import AdvisorAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="URGENT: Critical decline detected. Immediate escalation required."
        )
        mock_groq_cls.return_value = mock_llm

        agent = AdvisorAgent(mock_memory, groq_api_key="test-key")
        result = agent.advise(
            user_question="Revenue trend?",
            analysis="Revenue dropped sharply.",
            headline="34% revenue decline in Q3.",
            role="executive",
            anomaly_detected=True,
        )

        assert result["priority_level"] == "high"

    @patch("agents.advisor_agent.ChatGroq")
    def test_manager_vs_executive_prompts(self, mock_groq_cls, mock_memory):
        from agents.advisor_agent import AdvisorAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Some advice.")
        mock_groq_cls.return_value = mock_llm

        agent = AdvisorAgent(mock_memory, groq_api_key="test-key")

        agent.advise("q", "analysis", "headline", role="executive")
        exec_call_args = mock_llm.invoke.call_args
        exec_system = exec_call_args[0][0][0].content
        assert "C-suite" in exec_system or "strategic" in exec_system.lower()

        agent.advise("q", "analysis", "headline", role="manager")
        mgr_call_args = mock_llm.invoke.call_args
        mgr_system = mgr_call_args[0][0][0].content
        assert "team" in mgr_system.lower() or "tactical" in mgr_system.lower()


# ── Anomaly Detector Tests ─────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_severity_scoring(self, mock_db, mock_memory):
        from core.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(mock_db, mock_memory, threshold=2.0)
        assert detector._severity_from_z(3.6) == "critical"
        assert detector._severity_from_z(3.1) == "high"
        assert detector._severity_from_z(2.6) == "medium"
        assert detector._severity_from_z(2.1) == "low"

    def test_column_anomaly_detection_with_outliers(self, mock_db, mock_memory):
        from core.anomaly_detector import AnomalyDetector

        # Create a series with a clear outlier
        rng = np.random.default_rng(0)
        values = list(rng.normal(5000, 500, 100))
        values.append(50000)   # massive outlier

        df = pd.DataFrame({"monthly_income": values})
        detector = AnomalyDetector(mock_db, mock_memory, threshold=2.0)
        anomalies = detector._detect_column_anomalies(df, "monthly_income", "employees")

        assert len(anomalies) >= 1
        assert anomalies[0].z_score > 2.0

    def test_group_anomaly_detection(self, mock_db, mock_memory):
        from core.anomaly_detector import AnomalyDetector

        # Create department data where one dept has wildly different salary
        df = pd.DataFrame({
            "department": ["Sales"] * 30 + ["HR"] * 30 + ["Ghost_Dept"] * 30,
            "monthly_income": (
                list(np.random.default_rng(1).normal(8000, 500, 30)) +
                list(np.random.default_rng(2).normal(7800, 500, 30)) +
                list(np.random.default_rng(3).normal(50000, 500, 30))  # extreme outlier dept
            ),
        })

        detector = AnomalyDetector(mock_db, mock_memory, threshold=1.5)
        anomalies = detector._detect_column_anomalies(
            df, "monthly_income", "employees", group_col="department"
        )

        assert any(a.metric.endswith("Ghost_Dept") for a in anomalies)


# ── Orchestrator Integration Test ──────────────────────────────────────────────

class TestOrchestrator:

    @patch("agents.sql_agent.ChatGroq")
    @patch("agents.analyst_agent.ChatGroq")
    @patch("agents.advisor_agent.ChatGroq")
    def test_full_pipeline_success(
        self,
        mock_advisor_groq,
        mock_analyst_groq,
        mock_sql_groq,
        mock_db,
        mock_memory,
        sample_df,
    ):
        from agents.orchestrator import AgentOrchestrator

        for mock_cls in [mock_sql_groq, mock_analyst_groq, mock_advisor_groq]:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="SELECT department, AVG(monthly_income) AS avg FROM employees GROUP BY department"
            )
            mock_cls.return_value = mock_llm

        mock_analyst_groq.return_value.invoke.return_value = MagicMock(
            content="Key Finding: Sales has the highest average income. Patterns look stable."
        )
        mock_advisor_groq.return_value.invoke.return_value = MagicMock(
            content="1. STRATEGIC SUMMARY\nPerformance is strong.\n2. PRIORITY ACTIONS\nMaintain current trajectory."
        )

        orch = AgentOrchestrator(mock_db, mock_memory, groq_api_key="test-key")
        result = orch.run("Average salary by department?", role="executive")

        assert result.success
        assert "sql_agent" in result.pipeline_stages_completed
        assert result.user_question == "Average salary by department?"

    @patch("agents.sql_agent.ChatGroq")
    @patch("agents.analyst_agent.ChatGroq")
    @patch("agents.advisor_agent.ChatGroq")
    def test_pipeline_partial_failure_graceful(
        self,
        mock_advisor_groq,
        mock_analyst_groq,
        mock_sql_groq,
        mock_db,
        mock_memory,
    ):
        from agents.orchestrator import AgentOrchestrator

        mock_sql_groq.return_value.invoke.return_value = MagicMock(content="CANNOT_ANSWER")

        orch = AgentOrchestrator(mock_db, mock_memory, groq_api_key="test-key")
        result = orch.run("Unanswerable question about aliens?")

        assert not result.success
        assert result.sql_error is not None
        # Should not crash — graceful degradation
        assert result.pipeline_stages_completed == ["sql_agent"]


# ── Memory Tests ───────────────────────────────────────────────────────────────

class TestPlatformMemory:

    def test_store_and_retrieve_interaction(self, tmp_path):
        from core.memory import PlatformMemory

        memory = PlatformMemory(persist_dir=str(tmp_path / "chroma"))
        mem_id = memory.store_interaction(
            user_query="What is attrition?",
            sql_generated="SELECT COUNT(*) FROM employees WHERE attrition = true",
            analysis="Attrition is 16%.",
            recommendation="Review exit interview data.",
            role="executive",
        )
        assert mem_id is not None

        history = memory.get_relevant_history("attrition rate employees", n=2)
        assert isinstance(history, list)

    def test_store_anomaly(self, tmp_path):
        from core.memory import PlatformMemory

        memory = PlatformMemory(persist_dir=str(tmp_path / "chroma2"))
        anomaly_id = memory.store_anomaly(
            metric="monthly_income",
            description="Sales dept income is 2.8 std above average.",
            severity="high",
            z_score=2.8,
            table="employees",
            column="monthly_income",
        )
        assert anomaly_id is not None

        anomalies = memory.get_recent_anomalies()
        assert len(anomalies) >= 1

    def test_stats_empty_store(self, tmp_path):
        from core.memory import PlatformMemory

        memory = PlatformMemory(persist_dir=str(tmp_path / "chroma3"))
        stats = memory.get_stats()
        assert stats["total_interactions"] == 0
        assert stats["total_anomalies"] == 0
