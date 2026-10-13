"""
Agent state definition for the Finance Market Intelligence multi-agent system.
Uses TypedDict for LangGraph state management.
"""
from typing import TypedDict


class AgentState(TypedDict):
    """
    Shared state across all agents in the LangGraph workflow.
    Each agent reads and writes to this state.
    """
    # ── User Input ──
    query: str                       # Original user query
    session_id: str                  # Session identifier for tracking
    risk_appetite: str               # User's risk appetite: conservative, moderate, aggressive

    # ── Routing ──
    intent: str                      # Classified intent: market_info, analysis, portfolio, risk

    # ── RAG Results ──
    retrieved_docs: list[str]        # Retrieved document contents
    sources: list[dict]              # Source metadata for retrieved docs

    # ── Agent Outputs ──
    retriever_output: str            # Output from the Retriever Agent
    analysis_output: str             # Output from the Analysis Agent
    portfolio_output: str            # Output from the Portfolio Agent
    risk_output: str                 # Output from the Risk Assessment Agent
    agent_answer: str                # Current agent's response
    final_answer: str                # Final response to return to user

    # ── Validation & Feedback ──
    evaluator_verdict: str           # approved / needs_retry / failed
    evaluator_score: int             # Quality score (1-5)
    evaluator_feedback: str          # Feedback from the validator
    confidence: float                # Confidence score (0.0 - 1.0)

    # ── Control Flow ──
    attempts: int                    # Number of retry attempts
    agents_invoked: list[str]        # List of agents that were invoked
    latency: float                   # Total processing time in seconds
    error: str                       # Error message if any
