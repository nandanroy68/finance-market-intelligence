"""
Analysis Agent - Produces structured financial insights grounded in retrieved context.
Uses RAG output to generate data-driven analysis.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE


def _invoke_llm(prompt: str) -> str:
    """Invoke the LLM and return content as a string."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=2048,
        google_api_key=GEMINI_API_KEY,
    )
    result = llm.invoke(prompt)
    content = result.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def analysis_agent(state: AgentState) -> AgentState:
    """
    Produce structured financial insights grounded in the retrieved context.
    Takes the retriever's output and generates a comprehensive analysis.
    """
    query = state["query"]
    retrieved_context = "\n---\n".join(state.get("retrieved_docs", []))
    retriever_answer = state.get("retriever_output", "")
    agents_invoked = list(state.get("agents_invoked", []) or []) + ["analysis"]

    prompt = f"""
    You are a Senior Financial Analyst specializing in Indian and global markets.
    Based on the retrieved context and initial findings below, produce a structured financial analysis.

    User Query: {query}

    Retrieved Context:
    {retrieved_context}

    Initial Retriever Findings:
    {retriever_answer}

    Provide your analysis in this structure:
    1. **Key Findings**: Main insights from the data (2-3 bullet points)
    2. **Market Context**: Broader market conditions relevant to the query
    3. **Data-Driven Assessment**: Specific numbers, ratios, or metrics that support your analysis
    4. **Outlook**: Short-term and medium-term outlook based on available data
    5. **Confidence Level**: How confident you are in this analysis (low/medium/high) and why

    Base your analysis ONLY on the provided context. Do not fabricate data.
    If the context is insufficient, clearly state what additional data would be needed.
    """

    try:
        analysis = _invoke_llm(prompt)
        print(f"📊 Analysis Agent: Generated analysis ({len(analysis)} chars)")

        return {
            **state,
            "analysis_output": analysis,
            "agent_answer": analysis,
            "final_answer": analysis,
            "agents_invoked": agents_invoked,
        }

    except Exception as e:
        print(f"❌ Analysis Agent error: {e}")
        fallback = retriever_answer or "Unable to generate analysis at this time."
        return {
            **state,
            "analysis_output": fallback,
            "agent_answer": fallback,
            "final_answer": fallback,
            "agents_invoked": agents_invoked,
            "error": str(e),
        }
