#!/usr/bin/env python3
"""
setup_check.py

Run this before launching the platform for the first time.
Verifies that your environment is configured correctly:
  - Python version
  - All dependencies installed
  - .env file present and populated
  - Database connection reachable
  - Groq API key valid

Usage:
    python setup_check.py
"""

import sys
import os

print("\n" + "=" * 60)
print("  Autonomous BI Platform — Environment Check")
print("=" * 60)

errors = []
warnings = []


# ── Python version ─────────────────────────────────────────────
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 10):
    errors.append(f"Python 3.10+ required. Found: {major}.{minor}")
else:
    print(f"  ✓  Python {major}.{minor}")


# ── Required packages ──────────────────────────────────────────
required_packages = [
    ("streamlit", "streamlit"),
    ("langchain", "langchain"),
    ("langchain_groq", "langchain-groq"),
    ("groq", "groq"),
    ("sqlalchemy", "sqlalchemy"),
    ("chromadb", "chromadb"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("plotly", "plotly"),
    ("reportlab", "reportlab"),
    ("dotenv", "python-dotenv"),
    ("scipy", "scipy"),
    ("yaml", "PyYAML"),
    ("bcrypt", "bcrypt"),
]

missing = []
for import_name, pip_name in required_packages:
    try:
        __import__(import_name)
        print(f"  ✓  {pip_name}")
    except ImportError:
        missing.append(pip_name)
        print(f"  ✗  {pip_name}  ← MISSING")

if missing:
    errors.append(
        f"Missing packages: {', '.join(missing)}\n"
        f"    Fix: pip install {' '.join(missing)}"
    )


# ── .env file ──────────────────────────────────────────────────
from pathlib import Path
env_path = Path(".env")
if not env_path.exists():
    errors.append(".env file not found. Run: cp .env.example .env and fill it in.")
    print("  ✗  .env file — NOT FOUND")
else:
    print("  ✓  .env file found")
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    db_url = os.getenv("DATABASE_URL", "")

    if not groq_key or groq_key == "your_groq_api_key_here":
        errors.append("GROQ_API_KEY not set in .env")
        print("  ✗  GROQ_API_KEY — not configured")
    else:
        print(f"  ✓  GROQ_API_KEY — {'*' * 8}{groq_key[-4:]}")

    if not db_url or "your" in db_url:
        errors.append("DATABASE_URL not set in .env")
        print("  ✗  DATABASE_URL — not configured")
    else:
        print(f"  ✓  DATABASE_URL — {db_url.split('@')[-1] if '@' in db_url else 'set'}")


# ── Database connection ────────────────────────────────────────
db_url = os.getenv("DATABASE_URL", "")
if db_url and "your" not in db_url:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url, connect_args={"connect_timeout": 8})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  ✓  Database connection — OK")

        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if "employees" in tables:
            print(f"  ✓  Tables found: {', '.join(tables)}")
        else:
            warnings.append(
                "Tables not found. Run: python data/seed_database.py"
            )
            print("  ⚠  Tables not seeded — run: python data/seed_database.py")

    except Exception as e:
        errors.append(f"Database connection failed: {e}")
        print(f"  ✗  Database connection — FAILED: {e}")


# ── Groq API ───────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY", "")
if groq_key and groq_key != "your_groq_api_key_here":
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        # Minimal API call to verify key
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print("  ✓  Groq API — key valid, model reachable")
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "invalid" in err_str.lower():
            errors.append("Groq API key is invalid.")
            print("  ✗  Groq API — INVALID KEY")
        else:
            warnings.append(f"Groq API check inconclusive: {e}")
            print(f"  ⚠  Groq API — {e}")


# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)

if warnings:
    print("\n  Warnings:")
    for w in warnings:
        print(f"    ⚠  {w}")

if errors:
    print("\n  Errors (must fix before launching):")
    for e in errors:
        print(f"    ✗  {e}")
    print("\n  Fix the above issues, then run: streamlit run app.py\n")
    sys.exit(1)
else:
    print("\n  ✅  All checks passed! Launch with:\n")
    print("      streamlit run app.py\n")
