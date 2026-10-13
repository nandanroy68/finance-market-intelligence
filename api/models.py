"""
SQLAlchemy ORM models for logging RAG queries and user feedback.
Uses SQLite by default for zero-configuration local deployment.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RAGLog(Base):
    """Stores query execution logs, retrieved metadata, and latency."""
    __tablename__ = "rag_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    intent = Column(String(50), nullable=True)
    agents_invoked = Column(Text, nullable=True)  # JSON or comma-separated list
    confidence = Column(Float, default=0.0)
    latency = Column(Float, default=0.0)
    sources = Column(Text, nullable=True)  # JSON string of sources
    evaluator_verdict = Column(String(50), nullable=True)
    evaluator_score = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackLog(Base):
    """Stores user thumbs-up/down ratings and feedback/corrections."""
    __tablename__ = "feedback_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    query = Column(Text, nullable=True)
    rating = Column(Integer, nullable=False)  # 1 for up, -1 for down
    feedback_text = Column(Text, nullable=True)
    correction = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
