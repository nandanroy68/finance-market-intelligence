"""
Pydantic schemas for the Finance Market Intelligence API.
Defines request and response models for /query, /feedback, and /health endpoints.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: str = Field(default="fin-session-001", description="Unique session identifier for multi-turn memory")
    query: str = Field(..., min_length=2, description="User's financial market or portfolio question")
    risk_appetite: Optional[str] = Field(default="moderate", description="User risk appetite: conservative, moderate, aggressive")


class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_context: List[str] = Field(default_factory=list)
    agents_invoked: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    intent: Optional[str] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    portfolio_suggestion: Optional[str] = None
    latency: float = 0.0
    evaluator_verdict: Optional[str] = None
    evaluator_score: Optional[int] = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier related to the query")
    query: Optional[str] = Field(default="", description="The query being rated")
    rating: int = Field(..., ge=-1, le=5, description="1 for thumbs up, -1 for thumbs down, or 1-5 scale")
    feedback_text: Optional[str] = Field(default=None, description="Optional textual user feedback")
    correction: Optional[str] = Field(default=None, description="Optional factual correction from user")


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "Finance Market Intelligence API"
    version: str = "1.0.0"
    vector_store_loaded: bool
    database_connected: bool
    llm_configured: bool
    available_data_sources: List[str] = Field(default_factory=list)
