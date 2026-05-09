"""
agents/analyst_agent.py

Agent 2 — The Analyst Agent

Responsibility: take raw query results (a DataFrame) and produce a
structured analytical interpretation. This is where pattern detection,
trend identification, and anomaly flagging happen.

The agent receives the data, computes statistical context, and asks
the LLM to interpret findings the way a senior business analyst would —
not just describing what the numbers are, but explaining what they mean.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from core.memory import PlatformMemory

logger = logging.getLogger(__name__)


ANALYST_SYSTEM_PROMPT = """You are a senior business analyst at a top-tier management consulting firm.
You have been called in to interpret data results for a GCC enterprise client.

Your analytical approach:
1. Lead with the most important finding — the headline insight.
2. Identify patterns: trends over time, outliers, comparisons between groups.
3. Flag anything that looks anomalous — significant deviations from expectations.
4. Quantify everything — use percentages, ratios, and comparisons to give findings weight.
5. Be concise but complete. Executives want depth without noise.
6. Structure your response as: Key Finding → Supporting Evidence → Patterns Detected → Anomalies/Risks.

Never just list data back. Interpret it. If you see a 34% drop, say so AND explain what it likely means.

{context}
"""


class AnalystAgent:
    """
    Interprets DataFrames returned by the SQL Agent. Augments LLM prompts
    with computed statistical summaries (mean, std, outliers) so the model
    has the numbers it needs to reason accurately rather than hallucinating.
    """

    def __init__(
        self,
        memory: PlatformMemory,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ):
        self.memory = memory
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=0.2,
            max_tokens=1500,
        )

    def _compute_stats(self, df: pd.DataFrame) -> str:
        """
        Compute descriptive statistics and inject them into the prompt.
        This grounds the LLM's reasoning in actual numbers.
        """
        if df is None or df.empty:
            return "No data available for statistical analysis."

        lines = [f"Dataset: {len(df)} rows × {len(df.columns)} columns\n"]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            lines.append("Numeric column statistics:")
            for col in numeric_cols[:8]:   # cap to avoid token bloat
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                mean_val = series.mean()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
                outliers = series[np.abs(series - mean_val) > 2 * std_val]

                lines.append(
                    f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}, "
                    f"min={min_val:.2f}, max={max_val:.2f}, "
                    f"outliers={len(outliers)}"
                )

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            lines.append("\nCategorical columns:")
            for col in cat_cols[:4]:
                top = df[col].value_counts().head(3)
                lines.append(f"  {col} top values: {dict(top)}")

        return "\n".join(lines)

    def _format_data_for_prompt(self, df: pd.DataFrame, max_rows: int = 30) -> str:
        """Convert DataFrame to a compact string for the prompt."""
        if df is None or df.empty:
            return "No data returned."
        sample = df.head(max_rows)
        return sample.to_string(index=False, max_cols=15)

    def analyze(
        self,
        user_question: str,
        sql_used: str,
        df: pd.DataFrame,
    ) -> dict:
        """
        Analyze query results in the context of the original question.

        Returns:
          - analysis    : full analytical narrative (string)
          - headline    : one-line key finding
          - anomaly_detected : bool
          - stats_summary   : computed statistics string
        """
        logger.info(f"Analyst Agent analyzing results for: {user_question}")

        if df is None or df.empty:
            return {
                "analysis": "The query returned no data. This could indicate the filters are too narrow, or the data doesn't exist in the current dataset.",
                "headline": "No data returned for this query.",
                "anomaly_detected": False,
                "stats_summary": "",
            }

        stats = self._compute_stats(df)
        data_str = self._format_data_for_prompt(df)
        context = self.memory.get_context_summary(user_question)

        prompt = f"""
Original business question: "{user_question}"

SQL query that was run:
{sql_used}

Query results ({len(df)} rows):
{data_str}

Statistical context:
{stats}

Please provide your analytical interpretation following the structure in your instructions.
Be specific about numbers. Flag any anomalies you detect.
"""

        system = ANALYST_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            analysis_text = response.content.strip()

            # Heuristic: check if the analysis mentions anomaly-indicating language
            anomaly_keywords = [
                "anomaly", "anomalous", "unusual", "unexpected", "spike",
                "drop", "significant deviation", "outlier", "concerning",
                "drastic", "sudden", "sharp decline", "sharp increase",
            ]
            anomaly_detected = any(
                kw in analysis_text.lower() for kw in anomaly_keywords
            )

            # Extract headline (first sentence or first 120 chars)
            headline = analysis_text.split(".")[0].strip()
            if len(headline) > 150:
                headline = headline[:147] + "..."

            return {
                "analysis": analysis_text,
                "headline": headline,
                "anomaly_detected": anomaly_detected,
                "stats_summary": stats,
            }

        except Exception as e:
            logger.error(f"Analyst agent failed: {e}")
            return {
                "analysis": f"Analysis could not be completed: {e}",
                "headline": "Analysis failed.",
                "anomaly_detected": False,
                "stats_summary": stats,
            }

    def quick_stats(self, df: pd.DataFrame) -> dict:
        """
        Lightweight stats for dashboard display — no LLM call needed.
        Returns a dict of computed metrics directly from the DataFrame.
        """
        if df is None or df.empty:
            return {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            z_scores = np.abs((series - series.mean()) / series.std())
            result[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "outlier_count": int((z_scores > 2.0).sum()),
                "null_count": int(df[col].isna().sum()),
            }
        return result
