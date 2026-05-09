"""
core/memory.py

Persistent memory layer built on ChromaDB. Every agent interaction —
queries asked, anomalies flagged, insights generated — gets stored here
with metadata. Future sessions can retrieve relevant history, giving the
platform the feel of a seasoned analyst who remembers what you care about.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PlatformMemory:
    """
    Wraps ChromaDB collections to give the platform persistent, searchable
    memory across sessions. Three collections:
      - interactions  : user queries and full agent responses
      - anomalies     : every anomaly flagged, with resolution status
      - insights      : recurring patterns and strategic notes
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_store"
        )
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.interactions = self.client.get_or_create_collection(
            name="interactions",
            metadata={"description": "User queries and agent responses"},
        )
        self.anomalies_col = self.client.get_or_create_collection(
            name="anomalies",
            metadata={"description": "Detected anomalies and their status"},
        )
        self.insights_col = self.client.get_or_create_collection(
            name="insights",
            metadata={"description": "Strategic insights and recurring patterns"},
        )

    # ─── Interactions ──────────────────────────────────────────────────────────

    def store_interaction(
        self,
        user_query: str,
        sql_generated: str,
        analysis: str,
        recommendation: str,
        role: str = "executive",
    ) -> str:
        doc_id = str(uuid.uuid4())
        full_text = f"Query: {user_query}\n\nAnalysis: {analysis}\n\nRecommendation: {recommendation}"

        self.interactions.add(
            documents=[full_text],
            metadatas=[
                {
                    "user_query": user_query[:500],
                    "sql_generated": sql_generated[:1000],
                    "role": role,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            ids=[doc_id],
        )
        logger.debug(f"Stored interaction {doc_id}")
        return doc_id

    def get_relevant_history(self, query: str, n: int = 4) -> list[dict]:
        """
        Semantic search over past interactions. Used to give agents context
        about what the user has asked before and what was found.
        """
        try:
            results = self.interactions.query(
                query_texts=[query], n_results=min(n, self.interactions.count())
            )
            if not results["documents"][0]:
                return []

            history = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                history.append(
                    {
                        "summary": doc[:400],
                        "timestamp": meta.get("timestamp", ""),
                        "original_query": meta.get("user_query", ""),
                        "role": meta.get("role", ""),
                    }
                )
            return history
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return []

    # ─── Anomalies ─────────────────────────────────────────────────────────────

    def store_anomaly(
        self,
        metric: str,
        description: str,
        severity: str,
        z_score: float,
        table: str,
        column: str,
    ) -> str:
        doc_id = str(uuid.uuid4())
        text = f"Anomaly in {metric}: {description}"

        self.anomalies_col.add(
            documents=[text],
            metadatas=[
                {
                    "metric": metric,
                    "severity": severity,
                    "z_score": round(z_score, 3),
                    "table": table,
                    "column": column,
                    "resolved": "false",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            ids=[doc_id],
        )
        return doc_id

    def get_recent_anomalies(self, limit: int = 10) -> list[dict]:
        try:
            count = self.anomalies_col.count()
            if count == 0:
                return []
            results = self.anomalies_col.get(
                limit=min(limit, count),
                include=["documents", "metadatas"],
            )
            anomalies = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                anomalies.append({"description": doc, **meta})
            # sort newest first
            anomalies.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return anomalies
        except Exception as e:
            logger.warning(f"Anomaly retrieval failed: {e}")
            return []

    def resolve_anomaly(self, anomaly_id: str):
        try:
            self.anomalies_col.update(
                ids=[anomaly_id], metadatas=[{"resolved": "true"}]
            )
        except Exception as e:
            logger.warning(f"Could not resolve anomaly {anomaly_id}: {e}")

    # ─── Insights ──────────────────────────────────────────────────────────────

    def store_insight(self, title: str, detail: str, category: str = "general"):
        doc_id = str(uuid.uuid4())
        self.insights_col.add(
            documents=[f"{title}: {detail}"],
            metadatas=[
                {
                    "title": title,
                    "category": category,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            ids=[doc_id],
        )

    def get_context_summary(self, query: str) -> str:
        """
        Build a concise context string injected into agent prompts so they
        can reference prior findings without re-running analysis.
        """
        history = self.get_relevant_history(query, n=3)
        anomalies = self.get_recent_anomalies(limit=3)

        parts = []
        if history:
            parts.append("PREVIOUS RELATED FINDINGS:")
            for h in history:
                ts = h["timestamp"][:10] if h["timestamp"] else "unknown"
                parts.append(f"  [{ts}] {h['original_query']}: {h['summary'][:200]}")

        if anomalies:
            parts.append("\nRECENT ANOMALIES:")
            for a in anomalies:
                if a.get("resolved") != "true":
                    ts = a.get("timestamp", "")[:10]
                    parts.append(
                        f"  [{ts}] {a['description'][:200]} (severity: {a.get('severity', 'unknown')})"
                    )

        return "\n".join(parts) if parts else "No prior context available."

    def get_stats(self) -> dict:
        return {
            "total_interactions": self.interactions.count(),
            "total_anomalies": self.anomalies_col.count(),
            "total_insights": self.insights_col.count(),
        }
