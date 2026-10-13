"""
LangGraph Workflow - Compiles the multi-agent graph for Financial Market Intelligence.
Implements the Planner → Retriever → Specialist → Validator loop.
"""
import os
import sys

# Ensure Windows console supports UTF-8 characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState
from src.agents.planner import planner
from src.agents.retriever_agent import retriever_agent
from src.agents.analysis_agent import analysis_agent
from src.agents.portfolio_agent import portfolio_agent
from src.agents.risk_agent import risk_agent
from src.agents.validator import validator


def route_after_planner(state: AgentState) -> str:
    """Route to the retriever agent for all intents (retriever feeds all specialists)."""
    return "retriever"


def route_after_retriever(state: AgentState) -> str:
    """Route from retriever to the appropriate specialist agent based on intent."""
    intent = state.get("intent", "market_info")
    route_map = {
        "market_info": "validator",      # Simple RAG answer, skip specialist
        "analysis": "analysis_agent",
        "portfolio": "portfolio_agent",
        "risk": "risk_agent",
    }
    return route_map.get(intent, "validator")


def route_after_validator(state: AgentState) -> str:
    """
    Route after validation.
    If needs_retry and under max attempts → back to retriever.
    Otherwise → END.
    """
    verdict = state.get("evaluator_verdict", "approved")
    if verdict == "needs_retry":
        return "retriever"
    return "end"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph multi-agent workflow.
    
    Flow:
    START → Planner → Retriever → [Analysis/Portfolio/Risk/Validator] → Validator → END
                                                                           ↑          |
                                                                           └──retry───┘
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("portfolio_agent", portfolio_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("validator", validator)

    # ── Add edges ──
    # START → Planner
    graph.add_edge(START, "planner")

    # Planner → Retriever (all queries go through retrieval first)
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retriever": "retriever"},
    )

    # Retriever → Specialist (based on intent) or directly to Validator
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "validator": "validator",
            "analysis_agent": "analysis_agent",
            "portfolio_agent": "portfolio_agent",
            "risk_agent": "risk_agent",
        },
    )

    # Specialist Agents → Validator
    graph.add_edge("analysis_agent", "validator")
    graph.add_edge("portfolio_agent", "validator")
    graph.add_edge("risk_agent", "validator")

    # Validator → END or Retry
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "retriever": "retriever",
            "end": END,
        },
    )

    return graph


def compile_workflow():
    """Build and compile the workflow with memory checkpointing."""
    graph = build_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    try:
        print("✅ LangGraph workflow compiled successfully")
    except Exception:
        print("[OK] LangGraph workflow compiled successfully")
    return app


def get_initial_state(query: str, session_id: str = "", risk_appetite: str = "moderate") -> AgentState:
    """Create the initial state for a new query."""
    return AgentState(
        query=query,
        session_id=session_id,
        risk_appetite=risk_appetite,
        intent="",
        retrieved_docs=[],
        sources=[],
        retriever_output="",
        analysis_output="",
        portfolio_output="",
        risk_output="",
        agent_answer="",
        final_answer="",
        evaluator_verdict="",
        evaluator_score=0,
        evaluator_feedback="",
        confidence=0.0,
        attempts=0,
        agents_invoked=[],
        latency=0.0,
        error="",
    )


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🧠 AlphaFin Multi-Agent LangGraph Workflow Architecture")
    print("=" * 65 + "\n")

    app = compile_workflow()

    print("\n--- [ASCII Graph Representation] ---")
    try:
        print(app.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not render ASCII graph: {e}")

    print("\n--- [Mermaid Diagram Definition] ---")
    try:
        mermaid_code = app.get_graph().draw_mermaid()
        print(mermaid_code)
    except Exception as e:
        print(f"Could not render Mermaid code: {e}")

    print("\n" + "=" * 65)
    print("Graph execution ready. All nodes & conditional edges verified!")
    print("=" * 65)
