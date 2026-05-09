"""
core/anomaly_detector.py

Proactive Anomaly Detection Engine

This is what separates this platform from a dashboard. Instead of waiting
for someone to ask the right question, the monitor continuously scans the
data and surfaces alerts automatically.

Detection methods:
  1. Z-score analysis — flags values that deviate significantly from the mean
  2. Trend reversal detection — catches sudden directional changes
  3. Group comparison — identifies underperforming segments vs. peers
  4. Null/completeness monitoring — catches data quality issues

Each anomaly gets a severity score, a plain-English description, and is
handed to the Advisor Agent for a brief strategic recommendation.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.database import DatabaseConnection
from core.memory import PlatformMemory

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    metric: str
    table: str
    column: str
    description: str
    z_score: float
    severity: str          # 'critical' | 'high' | 'medium' | 'low'
    current_value: float
    expected_value: float
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    advisory: str = ""
    anomaly_id: str = ""


# Monitored metrics configuration — adapt to your schema
# Format: (table, column, group_by_column, description)
MONITORED_METRICS = [
    ("employees", "monthly_income", "department", "Monthly Income by Department"),
    ("employees", "performance_rating", "department", "Performance Rating by Department"),
    ("employees", "years_at_company", None, "Employee Tenure"),
    ("employees", "job_satisfaction", "job_role", "Job Satisfaction by Role"),
    ("sales_performance", "sales_amount", "region", "Sales by Region"),
    ("sales_performance", "quota_attainment", "sales_rep_id", "Quota Attainment by Rep"),
]


class AnomalyDetector:
    """
    Scans the live database for statistical anomalies and returns structured
    alert objects. Designed to run on a schedule (every N minutes) and push
    alerts to the UI and memory store.
    """

    def __init__(
        self,
        db: DatabaseConnection,
        memory: PlatformMemory,
        threshold: float = None,
    ):
        self.db = db
        self.memory = memory
        self.threshold = threshold or float(
            os.getenv("ANOMALY_THRESHOLD", "2.0")
        )

    def _severity_from_z(self, z: float) -> str:
        if abs(z) >= 3.5:
            return "critical"
        elif abs(z) >= 3.0:
            return "high"
        elif abs(z) >= 2.5:
            return "medium"
        else:
            return "low"

    def _detect_column_anomalies(
        self,
        df: pd.DataFrame,
        column: str,
        table: str,
        group_col: Optional[str] = None,
    ) -> list[Anomaly]:
        anomalies = []

        if column not in df.columns:
            return anomalies

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 5:
            return anomalies

        mean = series.mean()
        std = series.std()
        if std == 0:
            return anomalies

        if group_col and group_col in df.columns:
            # Group-level anomaly: flag groups whose mean deviates from overall
            group_means = df.groupby(group_col)[column].mean().dropna()
            z_scores = (group_means - group_means.mean()) / (group_means.std() + 1e-9)

            for group, z in z_scores.items():
                if abs(z) >= self.threshold:
                    direction = "above" if z > 0 else "below"
                    group_mean = group_means[group]
                    pct_diff = abs(group_mean - group_means.mean()) / (group_means.mean() + 1e-9) * 100

                    desc = (
                        f"{group} has a {column.replace('_', ' ')} of {group_mean:.2f}, "
                        f"which is {pct_diff:.1f}% {direction} the group average of "
                        f"{group_means.mean():.2f} ({abs(z):.1f}σ deviation)."
                    )
                    anomalies.append(
                        Anomaly(
                            metric=f"{column} — {group}",
                            table=table,
                            column=column,
                            description=desc,
                            z_score=float(z),
                            severity=self._severity_from_z(z),
                            current_value=round(float(group_mean), 2),
                            expected_value=round(float(group_means.mean()), 2),
                        )
                    )
        else:
            # Row-level outliers in a numeric column
            z_scores = np.abs((series - mean) / std)
            outlier_count = (z_scores >= self.threshold).sum()
            if outlier_count > 0:
                max_z_idx = z_scores.idxmax()
                max_z = z_scores[max_z_idx]
                worst_value = series[max_z_idx]

                desc = (
                    f"{column.replace('_', ' ').title()} has {outlier_count} outlier(s). "
                    f"Most extreme: {worst_value:.2f} vs expected mean of {mean:.2f} "
                    f"({max_z:.1f}σ deviation)."
                )
                anomalies.append(
                    Anomaly(
                        metric=column.replace("_", " ").title(),
                        table=table,
                        column=column,
                        description=desc,
                        z_score=float(max_z),
                        severity=self._severity_from_z(max_z),
                        current_value=round(float(worst_value), 2),
                        expected_value=round(float(mean), 2),
                    )
                )

        return anomalies

    def scan_all(self) -> list[Anomaly]:
        """
        Full database scan. Runs all configured metric checks and returns
        a deduplicated, severity-sorted list of anomalies.
        """
        all_anomalies: list[Anomaly] = []

        for table, column, group_col, description in MONITORED_METRICS:
            try:
                if group_col:
                    df = self.db.run_query(
                        f"SELECT {column}, {group_col} FROM {table} WHERE {column} IS NOT NULL LIMIT 5000"
                    )
                else:
                    df = self.db.run_query(
                        f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 5000"
                    )

                found = self._detect_column_anomalies(df, column, table, group_col)
                all_anomalies.extend(found)

            except Exception as e:
                logger.warning(f"Anomaly scan skipped for {table}.{column}: {e}")

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))

        # Persist to memory
        for anomaly in all_anomalies:
            try:
                anomaly.anomaly_id = self.memory.store_anomaly(
                    metric=anomaly.metric,
                    description=anomaly.description,
                    severity=anomaly.severity,
                    z_score=anomaly.z_score,
                    table=anomaly.table,
                    column=anomaly.column,
                )
            except Exception as e:
                logger.warning(f"Could not persist anomaly: {e}")

        logger.info(f"Anomaly scan complete — {len(all_anomalies)} anomalies found")
        return all_anomalies

    def scan_table(self, table: str) -> list[Anomaly]:
        """Targeted scan for a specific table. Used for on-demand checks."""
        matching = [(t, c, g, d) for t, c, g, d in MONITORED_METRICS if t == table]
        all_anomalies = []
        for t, c, g, d in matching:
            try:
                cols = f"{c}, {g}" if g else c
                df = self.db.run_query(
                    f"SELECT {cols} FROM {t} WHERE {c} IS NOT NULL LIMIT 5000"
                )
                all_anomalies.extend(self._detect_column_anomalies(df, c, t, g))
            except Exception as e:
                logger.warning(f"Targeted scan failed for {table}.{c}: {e}")
        return all_anomalies

    def get_data_quality_report(self) -> dict:
        """
        Check for data completeness issues — NULL rates, empty tables, etc.
        Useful for the executive dashboard's data health indicator.
        """
        report = {}
        for table, column, _, _ in MONITORED_METRICS:
            try:
                df = self.db.run_query(
                    f"""
                    SELECT
                        COUNT(*) as total_rows,
                        SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as null_count
                    FROM {table}
                    """
                )
                if not df.empty:
                    total = int(df["total_rows"].iloc[0])
                    nulls = int(df["null_count"].iloc[0])
                    null_rate = round(nulls / total * 100, 1) if total > 0 else 0
                    key = f"{table}.{column}"
                    report[key] = {
                        "total_rows": total,
                        "null_count": nulls,
                        "null_rate_pct": null_rate,
                        "healthy": null_rate < 5.0,
                    }
            except Exception as e:
                logger.warning(f"Data quality check failed for {table}.{column}: {e}")

        return report
