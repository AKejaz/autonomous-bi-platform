"""
app.py

Autonomous Business Intelligence Platform — Main Application Entry Point

Launch with: streamlit run app.py
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Autonomous BI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-username/autonomous-bi-platform",
        "Report a bug": None,
        "About": "Autonomous BI Platform — Multi-Agent AI for Enterprise Intelligence",
    },
)


# ── Lazy imports (avoid crashing before credentials are entered) ───────────────
def get_dependencies():
    from core.database import DatabaseConnection
    from core.memory import PlatformMemory
    from agents.orchestrator import AgentOrchestrator
    from core.anomaly_detector import AnomalyDetector
    from reports.pdf_generator import ExecutiveReportGenerator
    return DatabaseConnection, PlatformMemory, AgentOrchestrator, AnomalyDetector, ExecutiveReportGenerator


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main palette */
    :root {
        --navy: #0A2342;
        --teal: #00A8A8;
        --light-bg: #F5F7FA;
        --border: #D0D7E2;
    }

    /* Hide default Streamlit header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Top banner */
    .platform-header {
        background: linear-gradient(135deg, #0A2342 0%, #1A3A5C 100%);
        color: white;
        padding: 18px 28px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .platform-header h1 {
        margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.3px;
    }
    .platform-header p {
        margin: 4px 0 0; opacity: 0.75; font-size: 0.85rem;
    }

    /* KPI metric cards */
    .kpi-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(10,35,66,0.06);
        transition: box-shadow 0.2s;
    }
    .kpi-card:hover { box-shadow: 0 4px 16px rgba(10,35,66,0.12); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0A2342; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #8A8A8A; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-delta { font-size: 0.82rem; font-weight: 600; margin-top: 2px; }
    .delta-up { color: #27AE60; }
    .delta-down { color: #C0392B; }

    /* Agent pipeline status */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .agent-active { background: #E8F8F0; color: #27AE60; border: 1px solid #A9DFBF; }
    .agent-pending { background: #F5F7FA; color: #8A8A8A; border: 1px solid var(--border); }
    .agent-error { background: #FDECEA; color: #C0392B; border: 1px solid #F5A6A0; }

    /* Anomaly alert cards */
    .anomaly-card {
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .anomaly-critical { border-color: #C0392B; background: #FDECEA; }
    .anomaly-high     { border-color: #E67E22; background: #FEF0E6; }
    .anomaly-medium   { border-color: #F39C12; background: #FEF9E7; }
    .anomaly-low      { border-color: #27AE60; background: #E8F8F0; }

    /* Query response area */
    .response-panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .headline-callout {
        background: linear-gradient(135deg, #E8F4FD, #F0F8FF);
        border-left: 4px solid #00A8A8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-weight: 600;
        color: #0A2342;
        margin-bottom: 16px;
    }

    /* Sidebar */
    .css-1d391kg { background: #F5F7FA; }
    .sidebar-section {
        background: white;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--light-bg);
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
    }

    /* SQL code block */
    .sql-block {
        background: #F0F4F8;
        border: 1px solid #BDC3C7;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #1E3A5F;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    /* Priority badge */
    .priority-high   { color: #C0392B; font-weight: 700; }
    .priority-medium { color: #E67E22; font-weight: 700; }
    .priority-low    { color: #27AE60; font-weight: 700; }

    /* Login page */
    .login-wrapper {
        max-width: 420px;
        margin: 60px auto;
        background: white;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 8px 40px rgba(10,35,66,0.12);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
def init_session():
    defaults = {
        "authenticated": False,
        "username": None,
        "role": "executive",
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "db_url": os.getenv("DATABASE_URL", ""),
        "pipeline_result": None,
        "anomalies": [],
        "anomaly_scan_done": False,
        "chat_history": [],
        "db": None,
        "memory": None,
        "orchestrator": None,
        "detector": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Autonomous BI Platform")
        st.markdown("---")

        # Connection settings
        with st.expander("⚙️  Connection Settings", expanded=not st.session_state.db):
            api_key = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                type="password",
                help="Get yours at console.groq.com",
            )
            db_url = st.text_input(
                "Database URL",
                value=st.session_state.db_url,
                type="password",
                placeholder="postgresql://user:pass@host:5432/db",
            )

            if st.button("🔌  Connect", use_container_width=True, type="primary"):
                if not api_key or not db_url:
                    st.error("Both fields are required.")
                else:
                    connect_platform(api_key, db_url)

        # Connection status
        if st.session_state.db:
            st.success("✅  Database connected")
            st.info(f"👤  Role: **{st.session_state.role.capitalize()}**")

            st.markdown("---")

            # Anomaly scanner
            st.markdown("#### 🔍  Anomaly Monitor")
            if st.button("Run Full Scan", use_container_width=True):
                with st.spinner("Scanning for anomalies..."):
                    run_anomaly_scan()

            if st.session_state.anomalies:
                critical = sum(1 for a in st.session_state.anomalies if a.severity == "critical")
                high = sum(1 for a in st.session_state.anomalies if a.severity == "high")
                st.markdown(
                    f"Last scan: **{len(st.session_state.anomalies)}** anomalies "
                    f"({critical} critical, {high} high)"
                )

            st.markdown("---")

            # Memory stats
            if st.session_state.memory:
                stats = st.session_state.memory.get_stats()
                st.markdown("#### 💾  Memory Store")
                col1, col2 = st.columns(2)
                col1.metric("Interactions", stats["total_interactions"])
                col2.metric("Anomalies", stats["total_anomalies"])

            st.markdown("---")
            if st.button("🚪  Reset Session", use_container_width=True):
                for k in ["db", "memory", "orchestrator", "detector", "pipeline_result", "anomalies"]:
                    st.session_state[k] = None if k not in ["anomalies"] else []
                st.session_state.anomaly_scan_done = False
                st.rerun()


def connect_platform(api_key: str, db_url: str):
    """Initialise all platform components and cache in session state."""
    try:
        DatabaseConnection, PlatformMemory, AgentOrchestrator, AnomalyDetector, _ = get_dependencies()

        with st.spinner("Connecting to database..."):
            db = DatabaseConnection(db_url)
            if not db.test_connection():
                st.error("❌  Could not reach the database. Check your connection string.")
                return

        with st.spinner("Initialising memory store..."):
            memory = PlatformMemory()

        with st.spinner("Wiring up agents..."):
            orchestrator = AgentOrchestrator(db, memory, api_key)
            detector = AnomalyDetector(db, memory)

        st.session_state.db = db
        st.session_state.memory = memory
        st.session_state.orchestrator = orchestrator
        st.session_state.detector = detector
        st.session_state.groq_api_key = api_key
        st.session_state.db_url = db_url
        st.session_state.authenticated = True

        st.success("✅  Platform ready!")
        st.rerun()

    except Exception as e:
        st.error(f"Connection failed: {e}")
        logger.error(f"Platform init failed: {e}", exc_info=True)


def run_anomaly_scan():
    if not st.session_state.detector:
        return
    try:
        anomalies = st.session_state.detector.scan_all()
        st.session_state.anomalies = anomalies
        st.session_state.anomaly_scan_done = True
    except Exception as e:
        st.error(f"Anomaly scan failed: {e}")


# ── Landing / onboarding page ──────────────────────────────────────────────────
def render_landing():
    st.markdown("""
    <div class="platform-header">
        <div>
            <h1>🧠 Autonomous Business Intelligence Platform</h1>
            <p>Multi-agent AI that monitors your data, detects anomalies, and delivers executive-ready insights — proactively.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🤖 Three Specialized Agents**
        - SQL Agent — translates questions to queries
        - Analyst Agent — detects patterns & anomalies
        - Advisor Agent — generates strategic recommendations
        """)
    with col2:
        st.markdown("""
        **⚡ Proactive Intelligence**
        - Continuous anomaly monitoring
        - Alerts before you think to ask
        - Statistical deviation detection
        """)
    with col3:
        st.markdown("""
        **📊 Enterprise Ready**
        - Role-based dashboards (Exec / Manager)
        - One-click PDF executive reports
        - Persistent memory across sessions
        """)

    st.info("👈  Enter your Groq API key and database URL in the sidebar to get started.")

    with st.expander("🔑  Demo Credentials (if using seeded database)", expanded=False):
        st.markdown("""
        | Role | Username | Password |
        |------|----------|----------|
        | Executive | `executive` | `exec2024!` |
        | Manager | `manager` | `manager2024!` |

        Run `python data/seed_database.py` first to populate the database.
        """)


# ── Main query interface ───────────────────────────────────────────────────────
def render_query_interface():
    st.markdown("""
    <div class="platform-header">
        <div>
            <h1>🧠 Autonomous BI Platform</h1>
            <p>Ask anything about your business data. Three agents collaborate to give you the answer.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions based on role
    suggestions = {
        "executive": [
            "What is our company-wide attrition rate and which department is highest?",
            "Show me average salary by department ranked highest to lowest",
            "Which job roles have the lowest satisfaction scores this period?",
            "Compare performance ratings across all departments",
            "What percentage of our workforce is at risk of leaving?",
        ],
        "manager": [
            "Who are the bottom 5 performers by performance rating in my department?",
            "Show me overtime hours by employee and flag anyone working excessive hours",
            "Which employees have been in their current role the longest without promotion?",
            "What is the average tenure by job role in the Sales department?",
            "List employees with low job satisfaction and high performance — retention risks",
        ],
    }

    role = st.session_state.role

    # Quick-select suggestions
    st.markdown("#### 💡  Suggested Questions")
    cols = st.columns(min(3, len(suggestions[role])))
    selected_suggestion = None
    for i, suggestion in enumerate(suggestions[role][:3]):
        with cols[i]:
            if st.button(f"📌 {suggestion[:55]}...", key=f"sug_{i}", use_container_width=True):
                selected_suggestion = suggestion

    st.markdown("#### 🔍  Ask the Platform")
    query_value = selected_suggestion or ""

    with st.form("query_form", clear_on_submit=False):
        user_question = st.text_area(
            "Your question",
            value=query_value,
            height=90,
            placeholder="e.g. Who are my bottom 5 sales reps this quarter and why?",
            label_visibility="collapsed",
        )
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            submit = st.form_submit_button("🚀  Run Analysis", type="primary", use_container_width=True)
        with col_b:
            role_choice = st.selectbox("Role View", ["executive", "manager"],
                                       index=0 if role == "executive" else 1,
                                       label_visibility="collapsed")
        with col_c:
            sql_only = st.form_submit_button("⚡  SQL Only", use_container_width=True)

    if submit and user_question.strip():
        st.session_state.role = role_choice
        run_full_pipeline(user_question, role_choice)

    if sql_only and user_question.strip():
        run_sql_only(user_question)

    # Display results
    if st.session_state.pipeline_result:
        render_pipeline_results(st.session_state.pipeline_result)


def run_full_pipeline(question: str, role: str):
    orchestrator = st.session_state.orchestrator
    if not orchestrator:
        st.error("Platform not connected.")
        return

    progress = st.progress(0, text="Agent 1/3 — SQL Agent generating query...")

    try:
        # Run pipeline (no real-time progress since it's synchronous)
        with st.spinner("Three agents are collaborating on your question..."):
            result = orchestrator.run(question, role)

        progress.progress(100, text="✅  Analysis complete")
        st.session_state.pipeline_result = result

        # Add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "headline": result.headline,
            "role": role,
            "time": result.execution_time_sec,
            "anomaly": result.anomaly_detected,
        })

        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)


def run_sql_only(question: str):
    orchestrator = st.session_state.orchestrator
    if not orchestrator:
        return
    with st.spinner("SQL Agent running..."):
        result = orchestrator.run_sql_only(question)
    st.session_state.pipeline_result = result
    st.rerun()


def render_pipeline_results(result):
    from utils.charts import auto_chart
    from reports.pdf_generator import ExecutiveReportGenerator

    st.markdown("---")

    # Agent status badges
    stages = result.pipeline_stages_completed
    badge_html = ""
    for agent, label in [("sql_agent", "SQL Agent"), ("analyst_agent", "Analyst Agent"), ("advisor_agent", "Advisor Agent")]:
        css = "agent-active" if agent in stages else ("agent-error" if result.sql_error else "agent-pending")
        icon = "✓" if agent in stages else ("✗" if result.sql_error else "○")
        badge_html += f'<span class="agent-badge {css}">{icon} {label}</span>'

    st.markdown(
        f'<div style="margin-bottom:12px;">{badge_html}'
        f'<span style="color:#8A8A8A;font-size:0.78rem;margin-left:8px;">'
        f'Completed in {result.execution_time_sec}s</span></div>',
        unsafe_allow_html=True,
    )

    if result.sql_error:
        st.error(f"**Query Error:** {result.sql_error}")
        return

    # Anomaly alert banner
    if result.anomaly_detected:
        st.markdown("""
        <div class="anomaly-card anomaly-high">
            ⚠️  <strong>Anomaly Detected</strong> — The analyst identified unusual patterns in this data.
            See the Analysis tab for details.
        </div>
        """, unsafe_allow_html=True)

    # Priority indicator
    priority_colors = {"high": "#C0392B", "medium": "#E67E22", "low": "#27AE60"}
    p_color = priority_colors.get(result.priority_level, "#8A8A8A")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analysis", "🎯 Recommendations", "🗃️ Data", "🔧 SQL", "📄 Export"
    ])

    with tab1:
        if result.headline:
            st.markdown(
                f'<div class="headline-callout">💡 {result.headline}</div>',
                unsafe_allow_html=True,
            )

        if result.analysis:
            st.markdown(result.analysis)

        if result.dataframe is not None and not result.dataframe.empty:
            st.markdown("##### Data Visualization")
            fig = auto_chart(result.dataframe, title="Query Results")
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if result.recommendations:
            st.markdown(
                f'<p>Priority: <span style="color:{p_color};font-weight:700;">'
                f'{result.priority_level.upper()}</span></p>',
                unsafe_allow_html=True,
            )
            st.markdown(result.recommendations)
        else:
            st.info("Recommendations available after full pipeline run.")

    with tab3:
        if result.dataframe is not None and not result.dataframe.empty:
            st.markdown(f"**{result.row_count} rows returned**")

            # Search/filter
            search_term = st.text_input("🔍 Filter rows", placeholder="Type to filter...", key="table_filter")
            display_df = result.dataframe
            if search_term:
                mask = display_df.astype(str).apply(
                    lambda col: col.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                display_df = display_df[mask]

            st.dataframe(display_df, use_container_width=True, height=420)

            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download CSV",
                data=csv_data,
                file_name="query_results.csv",
                mime="text/csv",
            )
        else:
            st.info("No data to display.")

    with tab4:
        if result.sql_generated:
            st.markdown("**Generated SQL Query:**")
            st.markdown(
                f'<div class="sql-block">{result.sql_generated}</div>',
                unsafe_allow_html=True,
            )
            if result.stats_summary:
                with st.expander("📈 Statistical Summary"):
                    st.text(result.stats_summary)
        else:
            st.info("No SQL generated.")

    with tab5:
        st.markdown("#### 📄 Generate Executive Report")
        st.markdown(
            "Produce a boardroom-ready PDF with the full analysis, data snapshot, "
            "anomaly log, and prioritised recommendations."
        )
        company = st.text_input("Company Name (for report header)", "Enterprise Client")
        report_title = st.text_input("Report Title", f"Business Intelligence Report — {result.user_question[:40]}")

        if st.button("🖨️  Generate PDF Report", type="primary"):
            with st.spinner("Generating executive report..."):
                try:
                    generator = ExecutiveReportGenerator()
                    pdf_buffer = generator.generate(
                        title=report_title,
                        user_question=result.user_question,
                        headline=result.headline,
                        analysis=result.analysis,
                        recommendations=result.recommendations,
                        df=result.dataframe,
                        anomalies=st.session_state.anomalies or [],
                        sql_query=result.sql_generated,
                        role=result.role,
                        company_name=company,
                    )
                    st.download_button(
                        label="⬇️  Download Executive Report (PDF)",
                        data=pdf_buffer,
                        file_name=f"BI_Report_{result.user_question[:30].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅  Report ready for download!")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")


# ── Executive Dashboard ────────────────────────────────────────────────────────
def render_executive_dashboard():
    from utils.charts import bar_chart, pie_chart, scatter_chart, line_chart

    st.markdown("""
    <div class="platform-header">
        <div>
            <h1>📈 Executive Dashboard</h1>
            <p>Company-wide KPIs, trends, and strategic indicators — live from your database.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = st.session_state.db
    if not db:
        st.warning("Connect to database first.")
        return

    # KPI row
    try:
        kpi_df = db.run_query("""
            SELECT
                COUNT(*) as headcount,
                ROUND(AVG(monthly_income)::numeric, 0) as avg_salary,
                ROUND(AVG(job_satisfaction)::numeric, 2) as avg_satisfaction,
                ROUND(AVG(performance_rating)::numeric, 2) as avg_performance,
                ROUND(AVG(CASE WHEN attrition THEN 1.0 ELSE 0 END) * 100, 1) as attrition_pct,
                ROUND(AVG(years_at_company)::numeric, 1) as avg_tenure
            FROM employees
        """)

        if not kpi_df.empty:
            r = kpi_df.iloc[0]
            cols = st.columns(6)
            metrics = [
                ("Headcount", f"{int(r['headcount']):,}", None),
                ("Avg Salary", f"${float(r['avg_salary']):,.0f}", None),
                ("Satisfaction", f"{float(r['avg_satisfaction']):.1f}/5", None),
                ("Performance", f"{float(r['avg_performance']):.1f}/4", None),
                ("Attrition Rate", f"{float(r['attrition_pct']):.1f}%", "down" if r['attrition_pct'] > 15 else "up"),
                ("Avg Tenure", f"{float(r['avg_tenure']):.1f} yrs", None),
            ]
            for col, (label, value, trend) in zip(cols, metrics):
                delta_html = ""
                if trend == "down":
                    delta_html = '<div class="kpi-delta delta-down">▼ Monitor</div>'
                elif trend == "up":
                    delta_html = '<div class="kpi-delta delta-up">✓ Healthy</div>'
                col.markdown(
                    f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
                    f'<div class="kpi-label">{label}</div>{delta_html}</div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.warning(f"KPI query failed: {e}")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)
    with col1:
        try:
            dept_df = db.run_query("""
                SELECT department, ROUND(AVG(monthly_income)::numeric, 0) as avg_salary,
                COUNT(*) as headcount
                FROM employees GROUP BY department ORDER BY avg_salary DESC
            """)
            if not dept_df.empty:
                fig = bar_chart(dept_df, x="department", y="avg_salary",
                                title="Average Salary by Department", height=320)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Department chart: {e}")

    with col2:
        try:
            role_df = db.run_query("""
                SELECT job_role, ROUND(AVG(job_satisfaction)::numeric, 2) as satisfaction,
                COUNT(*) as count
                FROM employees GROUP BY job_role ORDER BY satisfaction ASC LIMIT 10
            """)
            if not role_df.empty:
                fig = bar_chart(role_df, x="satisfaction", y="job_role",
                                title="Satisfaction by Job Role (Bottom 10)",
                                orientation="h", height=320)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Role satisfaction chart: {e}")

    col3, col4 = st.columns(2)
    with col3:
        try:
            attr_df = db.run_query("""
                SELECT department,
                ROUND(AVG(CASE WHEN attrition THEN 1.0 ELSE 0 END) * 100, 1) as attrition_rate
                FROM employees GROUP BY department ORDER BY attrition_rate DESC
            """)
            if not attr_df.empty:
                fig = bar_chart(attr_df, x="department", y="attrition_rate",
                                title="Attrition Rate by Department (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Attrition chart: {e}")

    with col4:
        try:
            scatter_df = db.run_query("""
                SELECT monthly_income, job_satisfaction, department,
                performance_rating, years_at_company
                FROM employees
                LIMIT 400
            """)
            if not scatter_df.empty:
                fig = scatter_chart(scatter_df, x="monthly_income", y="job_satisfaction",
                                    color="department", size="performance_rating",
                                    title="Income vs Satisfaction (size = performance)", height=300)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Scatter chart: {e}")


# ── Manager Dashboard ──────────────────────────────────────────────────────────
def render_manager_dashboard():
    from utils.charts import bar_chart, scatter_chart

    st.markdown("""
    <div class="platform-header">
        <div>
            <h1>👥 Manager Dashboard</h1>
            <p>Team performance metrics, individual flags, and coaching opportunities.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = st.session_state.db
    if not db:
        st.warning("Connect to database first.")
        return

    # Department filter
    try:
        dept_options = db.run_query("SELECT DISTINCT department FROM employees ORDER BY department")
        departments = dept_options["department"].tolist() if not dept_options.empty else []
    except Exception:
        departments = []

    selected_dept = st.selectbox("Filter by Department", ["All"] + departments)
    dept_filter = f"WHERE department = '{selected_dept}'" if selected_dept != "All" else ""

    col1, col2, col3, col4 = st.columns(4)
    try:
        team_df = db.run_query(f"""
            SELECT COUNT(*) as team_size,
            ROUND(AVG(performance_rating)::numeric, 2) as avg_performance,
            ROUND(AVG(job_satisfaction)::numeric, 2) as avg_satisfaction,
            SUM(CASE WHEN over_time THEN 1 ELSE 0 END) as overtime_count
            FROM employees {dept_filter}
        """)
        if not team_df.empty:
            r = team_df.iloc[0]
            col1.metric("Team Size", int(r["team_size"]))
            col2.metric("Avg Performance", f"{float(r['avg_performance']):.2f}/4")
            col3.metric("Avg Satisfaction", f"{float(r['avg_satisfaction']):.2f}/5")
            col4.metric("On Overtime", int(r["overtime_count"]))
    except Exception as e:
        st.warning(f"Team KPIs: {e}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        try:
            bottom_df = db.run_query(f"""
                SELECT employee_number, job_role, performance_rating,
                job_satisfaction, years_at_company, monthly_income
                FROM employees {dept_filter}
                ORDER BY performance_rating ASC, job_satisfaction ASC
                LIMIT 15
            """)
            if not bottom_df.empty:
                st.markdown("**⚠️  Employees Flagged for Attention**")
                st.dataframe(bottom_df, use_container_width=True, height=320)
        except Exception as e:
            st.warning(f"Bottom performers: {e}")

    with col2:
        try:
            role_perf_df = db.run_query(f"""
                SELECT job_role,
                ROUND(AVG(performance_rating)::numeric, 2) as avg_perf,
                ROUND(AVG(job_satisfaction)::numeric, 2) as avg_sat,
                COUNT(*) as count
                FROM employees {dept_filter}
                GROUP BY job_role ORDER BY avg_perf DESC
            """)
            if not role_perf_df.empty:
                fig = bar_chart(role_perf_df, x="job_role", y="avg_perf",
                                title="Performance by Role", height=320)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Role performance: {e}")

    # High performers at risk of leaving
    try:
        st.markdown("#### 🌟  High Performers — Retention Risk")
        risk_df = db.run_query(f"""
            SELECT employee_number, job_role, performance_rating,
            job_satisfaction, years_at_company, monthly_income,
            CASE WHEN job_satisfaction <= 2 AND performance_rating >= 3 THEN 'HIGH RISK'
                 WHEN job_satisfaction = 3 AND performance_rating >= 3 THEN 'MEDIUM RISK'
                 ELSE 'LOW RISK' END as retention_risk
            FROM employees {dept_filter}
            WHERE performance_rating >= 3
            ORDER BY job_satisfaction ASC, performance_rating DESC
            LIMIT 20
        """)
        if not risk_df.empty:
            st.dataframe(risk_df, use_container_width=True, height=300)
    except Exception as e:
        st.warning(f"Retention risk: {e}")


# ── Anomaly Feed ───────────────────────────────────────────────────────────────
def render_anomaly_feed():
    st.markdown("""
    <div class="platform-header">
        <div>
            <h1>⚠️ Proactive Anomaly Monitor</h1>
            <p>The system scans your data continuously and surfaces issues before you think to look.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄  Run New Scan", type="primary"):
            with st.spinner("Scanning all monitored metrics..."):
                run_anomaly_scan()
            st.rerun()
    with col2:
        if st.session_state.anomalies:
            st.markdown(f"**{len(st.session_state.anomalies)} anomalies** detected in last scan")

    if not st.session_state.anomaly_scan_done:
        st.info("👆  Run a scan to start proactive monitoring.")
        st.markdown("""
        The anomaly monitor checks:
        - **Z-score deviations** in salary, performance, and satisfaction
        - **Group comparisons** — departments or regions vs. peers
        - **Outlier detection** across all numeric columns
        - **Sales quota attainment** anomalies by region and rep
        """)
        return

    anomalies = st.session_state.anomalies
    if not anomalies:
        st.success("✅  No anomalies detected. All monitored metrics are within normal range.")
        return

    # Severity filter
    severity_filter = st.selectbox(
        "Filter by severity",
        ["All", "critical", "high", "medium", "low"],
    )
    filtered = [a for a in anomalies if severity_filter == "All" or a.severity == severity_filter]

    for anomaly in filtered:
        css_class = f"anomaly-{anomaly.severity}"
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(anomaly.severity, "⚪")

        st.markdown(
            f"""
            <div class="anomaly-card {css_class}">
                {icon} <strong>[{anomaly.severity.upper()}]</strong> {anomaly.metric}<br>
                <span style="font-size:0.88rem;">{anomaly.description}</span><br>
                <span style="font-size:0.78rem;color:#8A8A8A;">
                    Z-score: {anomaly.z_score:.2f} σ &nbsp;|&nbsp;
                    Current: {anomaly.current_value} &nbsp;|&nbsp;
                    Expected: {anomaly.expected_value} &nbsp;|&nbsp;
                    Table: {anomaly.table}.{anomaly.column}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick advisory from Advisor Agent
        if anomaly.severity in ("critical", "high") and st.session_state.groq_api_key:
            with st.expander(f"📋  Get AI Advisory for this anomaly"):
                if st.button(f"Generate Advisory", key=f"adv_{anomaly.anomaly_id or anomaly.metric}"):
                    from agents.advisor_agent import AdvisorAgent
                    from core.memory import PlatformMemory
                    advisor = AdvisorAgent(
                        st.session_state.memory or PlatformMemory(),
                        st.session_state.groq_api_key,
                    )
                    with st.spinner("Advisor Agent thinking..."):
                        advisory = advisor.generate_anomaly_advisory(
                            metric=anomaly.metric,
                            description=anomaly.description,
                            z_score=anomaly.z_score,
                            role=st.session_state.role,
                        )
                    st.markdown(advisory)


# ── Chat History ───────────────────────────────────────────────────────────────
def render_history():
    st.markdown("#### 💬  Query History")
    history = st.session_state.chat_history
    if not history:
        st.info("No queries run yet this session.")
        return

    for i, item in enumerate(reversed(history)):
        with st.expander(f"[{item['role'].upper()}] {item['question'][:70]}...", expanded=i == 0):
            st.markdown(f"**Finding:** {item['headline']}")
            cols = st.columns(3)
            cols[0].markdown(f"Role: `{item['role']}`")
            cols[1].markdown(f"Time: `{item['time']}s`")
            cols[2].markdown(f"Anomaly: {'⚠️ Yes' if item['anomaly'] else '✓ No'}")


# ── Schema explorer ────────────────────────────────────────────────────────────
def render_schema():
    st.markdown("#### 🗺️  Database Schema Explorer")
    db = st.session_state.db
    if not db:
        st.warning("Connect first.")
        return

    try:
        schema = db.get_schema_info()
        for table, meta in schema.items():
            with st.expander(f"📋  {table}  ({len(meta['columns'])} columns)"):
                import pandas as pd
                cols_df = pd.DataFrame(meta["columns"])
                st.dataframe(cols_df, use_container_width=True)
                if meta["primary_keys"]:
                    st.markdown(f"**Primary Key:** {', '.join(meta['primary_keys'])}")

        if st.button("📊  Show Sample Data"):
            for table in list(schema.keys())[:3]:
                st.markdown(f"**{table}** (first 5 rows):")
                st.dataframe(db.get_table_sample(table), use_container_width=True)
    except Exception as e:
        st.error(f"Schema retrieval failed: {e}")


# ── Router ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    if not st.session_state.db:
        render_landing()
        return

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Ask the Platform", "📈 Executive Dashboard", "👥 Manager Dashboard",
         "⚠️ Anomaly Monitor", "💬 History", "🗺️ Schema"],
        label_visibility="collapsed",
    )

    pages = {
        "🔍 Ask the Platform": render_query_interface,
        "📈 Executive Dashboard": render_executive_dashboard,
        "👥 Manager Dashboard": render_manager_dashboard,
        "⚠️ Anomaly Monitor": render_anomaly_feed,
        "💬 History": render_history,
        "🗺️ Schema": render_schema,
    }

    pages.get(page, render_query_interface)()


if __name__ == "__main__":
    main()
