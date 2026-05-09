"""
data/seed_database.py

Database seeder — loads the IBM HR Analytics dataset into PostgreSQL.

Run once to set up the database. Safe to re-run (drops and recreates tables).
The dataset is pulled directly from a public source so no manual download needed.

Usage:
    python data/seed_database.py
"""

import os
import sys
import logging

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# IBM HR Analytics dataset — Kaggle mirror (public URL)
DATASET_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/IBM-HR-Analytics-Employee-Attrition-Performance.csv"

# Fallback column names if dataset format changes
REQUIRED_COLUMNS = [
    "Age", "Department", "Education", "EmployeeNumber", "Gender",
    "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
    "NumCompaniesWorked", "OverTime", "PerformanceRating", "TotalWorkingYears",
    "YearsAtCompany", "YearsInCurrentRole", "Attrition",
]


def load_hr_dataset() -> pd.DataFrame:
    logger.info("Loading IBM HR Analytics dataset...")
    try:
        df = pd.read_csv(DATASET_URL)
        logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Could not load from URL: {e}")
        logger.info("Attempting to load from local file data/hr_dataset.csv ...")
        local_path = os.path.join(os.path.dirname(__file__), "hr_dataset.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        raise FileNotFoundError(
            "Dataset not found. Download from Kaggle (IBM HR Analytics Employee Attrition & Performance) "
            "and save as data/hr_dataset.csv"
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case for cleaner SQL."""
    import re
    df.columns = [
        re.sub(r"(?<!^)(?=[A-Z])", "_", col).lower().replace(" ", "_")
        for col in df.columns
    ]
    return df


def create_employees_table(df: pd.DataFrame, engine) -> None:
    logger.info("Creating and populating employees table...")
    df_clean = df.copy()

    # Boolean conversions
    df_clean["attrition"] = df_clean["attrition"].map({"Yes": True, "No": False}).fillna(False)
    df_clean["over_time"] = df_clean["over_time"].map({"Yes": True, "No": False}).fillna(False)

    df_clean.to_sql("employees", engine, if_exists="replace", index=False)
    logger.info(f"  ✓ employees: {len(df_clean)} rows")


def create_sales_performance_table(df: pd.DataFrame, engine) -> None:
    """
    Synthesize a sales performance table from HR data.
    In production you'd replace this with a real sales DB.
    """
    import numpy as np
    logger.info("Creating synthetic sales_performance table...")

    rng = np.random.default_rng(42)
    n = len(df)

    sales_df = pd.DataFrame({
        "sales_rep_id": df["employee_number"].values,
        "department": df["department"].values,
        "region": rng.choice(["UAE", "Saudi Arabia", "Qatar", "Kuwait", "Bahrain", "Oman"], n),
        "quarter": rng.choice(["Q1-2024", "Q2-2024", "Q3-2024", "Q4-2024"], n),
        "sales_amount": (df["monthly_income"].values * rng.uniform(0.5, 3.0, n)).round(2),
        "quota_amount": (df["monthly_income"].values * 2.0).round(2),
        "quota_attainment": rng.uniform(40, 140, n).round(1),
        "deals_closed": rng.integers(0, 25, n),
        "avg_deal_size": rng.uniform(5000, 150000, n).round(2),
        "pipeline_value": rng.uniform(50000, 500000, n).round(2),
        "customer_satisfaction": rng.uniform(1, 5, n).round(1),
        "days_to_close": rng.integers(7, 120, n),
    })

    # Inject some artificial anomalies for demo purposes
    anomaly_indices = rng.choice(n, size=15, replace=False)
    sales_df.loc[anomaly_indices[:5], "sales_amount"] *= 0.15   # sudden drops
    sales_df.loc[anomaly_indices[5:10], "quota_attainment"] = rng.uniform(5, 20, 5)
    sales_df.loc[anomaly_indices[10:], "days_to_close"] = rng.integers(180, 365, 5)

    sales_df.to_sql("sales_performance", engine, if_exists="replace", index=False)
    logger.info(f"  ✓ sales_performance: {len(sales_df)} rows")


def create_department_metrics_table(df: pd.DataFrame, engine) -> None:
    """Aggregated department KPI table for executive dashboard queries."""
    import numpy as np
    logger.info("Creating department_metrics table...")

    dept_df = df.groupby("department").agg(
        headcount=("employee_number", "count"),
        avg_salary=("monthly_income", "mean"),
        avg_satisfaction=("job_satisfaction", "mean"),
        avg_performance=("performance_rating", "mean"),
        attrition_rate=("attrition", "mean"),
        avg_tenure=("years_at_company", "mean"),
    ).reset_index()

    dept_df = dept_df.round(2)
    dept_df.to_sql("department_metrics", engine, if_exists="replace", index=False)
    logger.info(f"  ✓ department_metrics: {len(dept_df)} rows")


def create_indices(engine) -> None:
    logger.info("Creating indices for query performance...")
    with engine.connect() as conn:
        queries = [
            "CREATE INDEX IF NOT EXISTS idx_emp_dept ON employees(department)",
            "CREATE INDEX IF NOT EXISTS idx_emp_role ON employees(job_role)",
            "CREATE INDEX IF NOT EXISTS idx_sales_region ON sales_performance(region)",
            "CREATE INDEX IF NOT EXISTS idx_sales_quarter ON sales_performance(quarter)",
            "CREATE INDEX IF NOT EXISTS idx_sales_rep ON sales_performance(sales_rep_id)",
        ]
        for q in queries:
            try:
                conn.execute(text(q))
                conn.commit()
            except Exception as e:
                logger.warning(f"Index creation skipped: {e}")
    logger.info("  ✓ Indices created")


def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in .env")
        sys.exit(1)

    engine = create_engine(db_url, connect_args={"connect_timeout": 10})

    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✓ Database connection successful")
    except Exception as e:
        logger.error(f"Cannot connect to database: {e}")
        sys.exit(1)

    df_raw = load_hr_dataset()
    df = normalize_columns(df_raw)

    create_employees_table(df, engine)
    create_sales_performance_table(df, engine)
    create_department_metrics_table(df, engine)
    create_indices(engine)

    logger.info("\n✅ Database seeded successfully!")
    logger.info("Tables created: employees, sales_performance, department_metrics")
    logger.info("You can now launch the platform with: streamlit run app.py")


if __name__ == "__main__":
    main()
