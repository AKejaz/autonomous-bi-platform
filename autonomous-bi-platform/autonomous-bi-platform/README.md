# Autonomous Business Intelligence Platform

A multi-agent AI system that connects directly to live business data, detects anomalies automatically, and generates executive-ready reports — without waiting to be asked.

Built with Python, LangChain, Groq's LLaMA 3.3 70b, and Streamlit. Deployed publicly on Streamlit Cloud.

**[Live Demo →](https://your-app.streamlit.app)** &nbsp;|&nbsp; **[Architecture Diagram](assets/architecture.svg)**

---

![Platform Screenshot](assets/screenshot_placeholder.png)

---

## What This Actually Does

Most BI tools are reactive. You open a dashboard, you look at numbers, you form a question, and maybe you find an answer. This platform flips that model.

Three specialized AI agents work in sequence every time you ask a question — or even when you don't. The SQL Agent translates plain English into a database query and runs it. The Analyst Agent takes the results and interprets them the way a senior analyst would, looking for patterns, outliers, and deviations from the norm. The Advisor Agent takes those findings and turns them into concrete recommendations tailored to whether you're a C-suite executive or a team manager.

On top of that, a proactive anomaly monitor scans the data continuously and fires alerts when something looks wrong — before anyone thinks to look.

The system also has memory. Ask it something today, and next week it will remember what you flagged, what recovered, and what's still open.

---

## Architecture

```
User Interface (Streamlit)
        │
        ▼
Agent Orchestrator
   ├── Agent 1: SQL Agent        → NL → SQL → execute → DataFrame
   ├── Agent 2: Analyst Agent    → DataFrame → patterns → insights
   └── Agent 3: Advisor Agent    → insights → recommendations (role-aware)
        │
        ├── Live Database (PostgreSQL / Supabase)
        ├── Memory Store (ChromaDB — persistent, semantic search)
        ├── Anomaly Monitor (z-score + group deviation detection)
        └── Groq API (LLaMA 3.3 70b Versatile)
```

Full architecture diagram: [`assets/architecture.svg`](assets/architecture.svg)

---

## Features

**Multi-Agent Pipeline**
Three agents collaborate on every query. The SQL Agent is stateless and deterministic (temperature=0). The Analyst Agent is interpretive (temp=0.2). The Advisor Agent is strategic and role-aware (temp=0.35). Each can fail independently without crashing the pipeline.

**Proactive Anomaly Detection**
The monitor uses z-score analysis and group comparison to flag deviations without being asked. It fires alerts like: *"Sales quota attainment in Qatar is 2.8 standard deviations below the regional average — 67% below peers."* Nobody queried that. The system found it.

**Persistent Memory**
Every interaction is stored in ChromaDB with semantic embeddings. When you ask a follow-up question next week, the agents retrieve relevant prior context and reason about what's changed.

**Role-Based Dashboards**
Executive view: company-wide KPIs, attrition trends, salary distributions, performance vs satisfaction scatterplots. Manager view: team-level metrics, flagged employees, coaching opportunities, retention risk scoring.

**One-Click PDF Reports**
ReportLab-generated PDFs with a proper cover page, executive summary callout, analytical findings, data tables, anomaly log, and prioritized recommendations. Boardroom-ready.

**Self-Correcting SQL**
If the generated query fails, the agent feeds the error back to itself and attempts a correction before surfacing the error to the user. Works in the majority of cases.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — LLaMA 3.3 70b Versatile |
| Agent Framework | LangChain |
| Database | PostgreSQL via Supabase (free tier) |
| ORM / Query | SQLAlchemy |
| Vector Memory | ChromaDB (persistent local store) |
| UI | Streamlit |
| Charts | Plotly Express |
| PDF Generation | ReportLab |
| Statistics | NumPy, SciPy |
| Testing | pytest |
| Deployment | Streamlit Cloud |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com) (free)
- A PostgreSQL database — [Supabase](https://supabase.com) free tier works perfectly

### 1. Clone and install

```bash
git clone https://github.com/AKejaz/autonomous-bi-platform.git
cd autonomous-bi-platform

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:password@host:5432/dbname
APP_SECRET_KEY=any_random_string_here
```

If you're using Supabase, your `DATABASE_URL` is under **Settings → Database → Connection String (URI)**.

### 3. Seed the database

This loads the IBM HR Analytics dataset and creates the required tables:

```bash
python data/seed_database.py
```

You should see:
```
INFO: ✓ Database connection successful
INFO: ✓ employees: 1470 rows
INFO: ✓ sales_performance: 1470 rows
INFO: ✓ department_metrics: 3 rows
INFO: ✅ Database seeded successfully!
```

### 4. Run the platform

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter your credentials in the sidebar, and start querying.

---

## Default Credentials

After seeding:

| Role | Username | Password |
|---|---|---|
| Executive | `executive` | `exec2024!` |
| Manager | `manager` | `manager2024!` |

Change these in `auth_config.yaml` before deploying publicly.

---

## Sample Queries to Try

These work out of the box with the seeded IBM HR dataset:

```
What is our overall attrition rate and which department is worst?

Show me the bottom 10 employees by performance rating and their satisfaction scores

Which job roles have the highest income-to-satisfaction ratio?

Compare average tenure across departments — where are people staying longest?

Flag employees with high performance but low satisfaction — who are we about to lose?

What percentage of our workforce is on overtime and how does that correlate with attrition?
```

---

## Deployment on Streamlit Cloud

1. Push your repo to GitHub (make sure `.env` and `auth_config.yaml` are in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo

3. Set secrets in the Streamlit Cloud dashboard (**Settings → Secrets**):

```toml
GROQ_API_KEY = "your_key_here"
DATABASE_URL = "your_db_url_here"
APP_SECRET_KEY = "your_secret_here"
```

4. Set the main file path to `app.py` and deploy.

ChromaDB will initialize a fresh memory store on first run in the cloud. For persistent memory across Streamlit Cloud restarts, point `CHROMA_PERSIST_DIR` to a mounted volume or switch to a hosted vector DB like Pinecone.

---

## Project Structure

```
autonomous-bi-platform/
│
├── app.py                        # Streamlit application entry point
│
├── agents/
│   ├── sql_agent.py              # Agent 1: NL → SQL → DataFrame
│   ├── analyst_agent.py          # Agent 2: DataFrame → insights
│   ├── advisor_agent.py          # Agent 3: insights → recommendations
│   └── orchestrator.py           # Pipeline controller
│
├── core/
│   ├── database.py               # SQLAlchemy connection + schema introspection
│   ├── memory.py                 # ChromaDB persistent memory
│   └── anomaly_detector.py       # Proactive anomaly detection engine
│
├── reports/
│   └── pdf_generator.py          # ReportLab executive PDF generator
│
├── utils/
│   ├── auth.py                   # Role-based authentication
│   └── charts.py                 # Plotly chart utilities
│
├── data/
│   └── seed_database.py          # IBM HR dataset loader + DB seeder
│
├── tests/
│   └── test_agents.py            # pytest unit tests (mock-based, no DB needed)
│
├── assets/
│   └── architecture.svg          # System architecture diagram
│
├── .streamlit/
│   ├── config.toml               # Theme and server config
│   └── secrets.toml.example      # Template for Streamlit Cloud secrets
│
├── .env.example                  # Environment variable template
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Running Tests

Tests are fully mocked — no database or API key required:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_agents.py::TestSQLAgent::test_successful_query PASSED
tests/test_agents.py::TestSQLAgent::test_cannot_answer_response PASSED
tests/test_agents.py::TestSQLAgent::test_sql_cleaning_strips_fences PASSED
tests/test_agents.py::TestSQLAgent::test_self_correction_on_db_error PASSED
tests/test_agents.py::TestAnalystAgent::test_empty_dataframe_handling PASSED
...
```

---

## Dataset

The platform ships with a seeder for the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset from Kaggle — 1,470 employee records across 35 attributes including salary, satisfaction scores, performance ratings, tenure, attrition status, and job role.

A synthetic `sales_performance` table is generated alongside it, with regional breakdowns and quota attainment data specifically designed to demonstrate the anomaly detection features.

---

## Extending the Platform

**Add a new data source**
Connect additional tables by updating `MONITORED_METRICS` in `core/anomaly_detector.py` and the schema will be auto-detected on the next connection.

**Swap the LLM**
Change the `model` parameter in `AgentOrchestrator.__init__()`. Any model available through LangChain's ChatGroq interface will work. Tested with `llama-3.3-70b-versatile` and `mixtral-8x7b-32768`.

**Add a new agent**
Subclass or extend the pattern in `agents/`. The orchestrator is designed to accept additional pipeline stages — add a new stage between analyst and advisor, or as a post-processing step.

**Connect to your own database**
Update `DATABASE_URL` to point at any PostgreSQL database. The SQL Agent introspects the schema dynamically — no hardcoded table names.

---

## Why This Project

Built to solve a problem I kept seeing in GCC enterprise environments: organizations paying significant sums for BI consultants to manually surface insights that a well-designed AI system could find automatically, continuously, and at a fraction of the cost.

The multi-agent architecture isn't a gimmick — it reflects how this work actually happens. The analyst who writes the query is not the same person who interprets the results, and neither of them is the same person who decides what to do about it. Separating those responsibilities into distinct agents with different prompting strategies produces meaningfully better output than a single LLM call.

---

## License

MIT — use it, fork it, build on it.

---

## Contact

Built by Ali Ejaz(https://www.linkedin.com/in/ali-ejaz-analytics/) — open to opportunities in AI engineering and data science across the GCC region.
