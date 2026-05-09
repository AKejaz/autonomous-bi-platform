# Changelog

All notable changes to this project are documented here.

---

## [1.0.0] — 2024-06-01

### Initial release

**Core platform**
- Three-agent pipeline: SQL Agent → Analyst Agent → Advisor Agent
- Agent orchestrator with independent failure handling per stage
- Self-correcting SQL generation (one retry with error feedback)

**Intelligence features**
- Proactive anomaly detection using z-score and group deviation analysis
- Persistent memory via ChromaDB with semantic retrieval
- Role-aware recommendations (executive vs. manager prompting)

**Interface**
- Streamlit UI with custom CSS (navy/teal enterprise palette)
- Executive dashboard with live KPI tiles and Plotly charts
- Manager dashboard with team metrics and retention risk scoring
- Anomaly feed with severity filtering and on-demand AI advisories
- Query history with session-level tracking

**Data**
- IBM HR Analytics dataset seeder (1,470 records)
- Synthetic sales performance table with injected anomalies for demo
- SQLAlchemy connection pooling with schema auto-introspection

**Reports**
- ReportLab PDF generator with cover page, findings, data tables, anomaly log, recommendations, and SQL appendix

**Developer experience**
- pytest suite with mocked LLM and database calls
- GitHub Actions CI (Python 3.10 + 3.11)
- Environment check script (`setup_check.py`)
- `.env.example` template with documented variables
