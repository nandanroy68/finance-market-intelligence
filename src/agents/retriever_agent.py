"""
Retriever Agent - Fetches relevant financial documents using the RAG tool.
This is the primary RAG consumer that feeds context to downstream agents.
"""
from src.agents.state import AgentState
from src.tools.rag_tool import rag_query


def retriever_agent(state: AgentState) -> AgentState:
    """
    Invoke the RAG tool to retrieve relevant financial documents.
    Stores retrieved docs, sources, and a preliminary answer in state.
    """
    query = state["query"]
    agents_invoked = list(state.get("agents_invoked", []) or []) + ["retriever"]

    try:
        result = rag_query.invoke(query)

        if "error" in result:
            return {
                **state,
                "retriever_output": f"Retrieval error: {result['error']}",
                "retrieved_docs": [],
                "sources": [],
                "agents_invoked": agents_invoked,
                "error": result["error"],
            }

        retrieved_docs = result.get("retrieved_docs", [])
        sources = result.get("sources", [])
        answer = result.get("answer", "")
        latency = result.get("latency", 0)

        print(f"📥 Retriever: Retrieved {len(retrieved_docs)} documents in {latency}s")

        return {
            **state,
            "retriever_output": answer,
            "retrieved_docs": retrieved_docs,
            "sources": sources,
            "agent_answer": answer,
            "latency": latency,
            "agents_invoked": agents_invoked,
        }

    except Exception as e:
        print(f"❌ Retriever error: {e}")
        return {
            **state,
            "retriever_output": f"Error during retrieval: {str(e)}",
            "retrieved_docs": [],
            "sources": [],
            "agents_invoked": agents_invoked,
            "error": str(e),
        }
