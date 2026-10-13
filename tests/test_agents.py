"""
Unit tests for agent routing, agent state initialization, and evaluation metrics.
"""
from src.graph.workflow import (
    get_initial_state,
    route_after_planner,
    route_after_retriever,
    route_after_validator,
)
from src.evaluation.metrics import compute_bleu, compute_rouge_l, compute_relevance_score


def test_get_initial_state():
    state = get_initial_state("What is the repo rate?", session_id="test_123", risk_appetite="aggressive")
    assert state["query"] == "What is the repo rate?"
    assert state["session_id"] == "test_123"
    assert state["risk_appetite"] == "aggressive"
    assert state["attempts"] == 0
    assert state["retrieved_docs"] == []


def test_routing_logic():
    # route_after_planner always routes to retriever
    state = get_initial_state("test")
    assert route_after_planner(state) == "retriever"

    # route_after_retriever routes based on intent
    state["intent"] = "portfolio"
    assert route_after_retriever(state) == "portfolio_agent"

    state["intent"] = "risk"
    assert route_after_retriever(state) == "risk_agent"

    state["intent"] = "analysis"
    assert route_after_retriever(state) == "analysis_agent"

    state["intent"] = "market_info"
    assert route_after_retriever(state) == "validator"

    # route_after_validator checks evaluator_verdict
    state["evaluator_verdict"] = "approved"
    assert route_after_validator(state) == "end"

    state["evaluator_verdict"] = "needs_retry"
    assert route_after_validator(state) == "retriever"


def test_evaluation_metrics():
    ref = "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent."
    cand = "Reserve Bank of India maintained the repo rate at 6.5 percent."

    bleu = compute_bleu(ref, cand)
    assert isinstance(bleu, float)
    assert bleu >= 0.0

    rouge = compute_rouge_l(ref, cand)
    assert "f1" in rouge
    assert rouge["f1"] > 0.5

    relevance = compute_relevance_score("repo rate", cand, [ref])
    assert isinstance(relevance, float)
    assert relevance > 0.0
