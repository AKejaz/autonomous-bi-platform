"""
agents/advisor_agent.py

Agent 3 — The Advisor Agent

Responsibility: take the analyst's findings and convert them into
concrete, prioritized strategic recommendations tailored to the user's
role (executive vs. manager). This is the layer that bridges data
to decisions.

The advisor operates at the intersection of data science and business
strategy. It knows the GCC enterprise context and frames recommendations
accordingly — linking findings to bottom-line impact, workforce risk,
revenue velocity, and operational efficiency.
"""

import logging

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from core.memory import PlatformMemory

logger = logging.getLogger(__name__)


ADVISOR_SYSTEM_PROMPT_EXECUTIVE = """You are a strategic advisor to the C-suite of a leading GCC enterprise.
Your recommendations carry significant operational and financial weight.

Your advisory style:
- Think in terms of business impact: revenue, risk, cost, and competitive positioning.
- Every recommendation must have a clear rationale tied to the data.
- Prioritize by urgency (immediate action) and impact (high/medium/low).
- Be direct. Executives don't have time for hedging.
- Frame recommendations in terms of KPIs, ROI, and strategic objectives.
- Reference GCC market context where relevant (Vision 2030, UAE digital economy, etc.).
- Close with one "Watch List" item — the single thing leadership should monitor most closely.

Format your response as:
1. STRATEGIC SUMMARY (2-3 sentences max)
2. PRIORITY ACTIONS (numbered, specific, time-bound where possible)
3. RISK FACTORS (what could get worse if unaddressed)
4. WATCH LIST (one metric to track weekly)
"""

ADVISOR_SYSTEM_PROMPT_MANAGER = """You are an experienced operations manager providing tactical guidance to a team lead.
Your recommendations must be immediately actionable at the team level.

Your advisory style:
- Focus on what can be done this week and this month.
- Be specific about which team members, processes, or accounts need attention.
- Acknowledge team dynamics — people management is as important as metrics.
- Suggest specific interventions: coaching conversations, process changes, resource reallocation.
- Quantify expected impact where possible.

Format your response as:
1. TEAM SITUATION SUMMARY (2-3 sentences)
2. IMMEDIATE ACTIONS (this week — specific and assignable)
3. 30-DAY PLAN (medium-term tactical moves)
4. COACHING OPPORTUNITIES (where to invest in your people)
"""


class AdvisorAgent:
    """
    Synthesizes analyst findings into role-aware strategic recommendations.
    The same data produces different advice for an executive vs. a manager —
    because the decisions they make are fundamentally different.
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
            temperature=0.35,    # slight creativity for strategic framing
            max_tokens=1500,
        )

    def advise(
        self,
        user_question: str,
        analysis: str,
        headline: str,
        role: str = "executive",
        anomaly_detected: bool = False,
    ) -> dict:
        """
        Generate strategic recommendations from analyst findings.

        Args:
            user_question : original user query
            analysis      : full analysis from AnalystAgent
            headline      : one-line key finding
            role          : 'executive' or 'manager'
            anomaly_detected: whether the analyst flagged anomalies

        Returns:
            - recommendations : full advisory text
            - priority_level  : 'high' | 'medium' | 'low'
            - action_count    : estimated number of recommended actions
        """
        logger.info(f"Advisor Agent generating recommendations (role: {role})")

        system_prompt = (
            ADVISOR_SYSTEM_PROMPT_EXECUTIVE
            if role == "executive"
            else ADVISOR_SYSTEM_PROMPT_MANAGER
        )

        context = self.memory.get_context_summary(user_question)
        anomaly_note = (
            "\n⚠️  ALERT: The analysis has flagged one or more anomalies. "
            "Treat this as elevated priority.\n"
            if anomaly_detected
            else ""
        )

        prompt = f"""
{anomaly_note}
Business question that was asked: "{user_question}"

Key finding: {headline}

Full analysis:
{analysis}

Prior context:
{context}

Based on this analysis, provide your strategic recommendations in the format specified.
Be specific, actionable, and tie every recommendation back to the data.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            recommendations = response.content.strip()

            # Assess priority level from content signals
            high_signals = [
                "immediate", "urgent", "critical", "risk", "decline",
                "anomaly", "alert", "significant drop", "escalate",
            ]
            medium_signals = [
                "monitor", "review", "consider", "opportunity", "watch",
            ]

            text_lower = recommendations.lower()
            if anomaly_detected or any(s in text_lower for s in high_signals):
                priority = "high"
            elif any(s in text_lower for s in medium_signals):
                priority = "medium"
            else:
                priority = "low"

            # Count actionable items heuristically
            action_count = recommendations.count("\n") // 3 + 1

            return {
                "recommendations": recommendations,
                "priority_level": priority,
                "action_count": action_count,
            }

        except Exception as e:
            logger.error(f"Advisor agent failed: {e}")
            return {
                "recommendations": f"Could not generate recommendations: {e}",
                "priority_level": "low",
                "action_count": 0,
            }

    def generate_anomaly_advisory(
        self,
        metric: str,
        description: str,
        z_score: float,
        role: str = "executive",
    ) -> str:
        """
        Lightweight advisory specifically for proactive anomaly alerts.
        Called by the monitor without a full multi-agent pipeline.
        """
        prompt = f"""
An automated monitor has detected a data anomaly. Generate a brief advisory (3-5 sentences).

Metric affected: {metric}
Anomaly description: {description}
Statistical severity: {z_score:.1f} standard deviations from normal

The advisory should:
1. State clearly what was detected and how significant it is.
2. Name the most likely business causes (2-3 hypotheses).
3. Recommend one immediate action.

Keep it under 150 words. Be direct.
"""
        system = (
            ADVISOR_SYSTEM_PROMPT_EXECUTIVE
            if role == "executive"
            else ADVISOR_SYSTEM_PROMPT_MANAGER
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Advisory generation failed: {e}"
