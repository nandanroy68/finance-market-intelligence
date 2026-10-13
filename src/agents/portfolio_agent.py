"""
Portfolio Agent - Suggests asset allocations based on user's risk appetite and retrieved context.
Invokes the Risk Assessment Agent as a tool (Agents-as-Tools pattern).
"""
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.tools.market_data_tool import suggest_allocation
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


def portfolio_agent(state: AgentState) -> AgentState:
    """
    Suggest portfolio allocations based on user's risk appetite and market context.
    Uses the suggest_allocation tool and enriches with LLM-generated advice.
    """
    query = state["query"]
    risk_appetite = state.get("risk_appetite", "moderate") or "moderate"
    analysis_output = state.get("analysis_output", "")
    retrieved_context = "\n---\n".join(state.get("retrieved_docs", []))
    agents_invoked = list(state.get("agents_invoked", []) or []) + ["portfolio"]

    # Use the allocation tool
    allocation_result = suggest_allocation.invoke(risk_appetite)

    # Generate personalized portfolio advice
    prompt = f"""
    You are a Portfolio Manager with expertise in Indian financial markets.
    Based on the user's risk appetite and the available market data, provide portfolio advice.

    User Query: {query}
    Risk Appetite: {risk_appetite}

    Suggested Base Allocation:
    {json.dumps(allocation_result, indent=2)}

    Market Analysis Context:
    {analysis_output}

    Retrieved Market Data:
    {retrieved_context}

    Provide:
    1. **Recommended Allocation**: The suggested percentage allocation across asset classes
    2. **Specific Stock/Fund Suggestions**: Based on available data, suggest specific instruments
    3. **Rationale**: Why this allocation suits the user's risk appetite
    4. **Rebalancing Advice**: When and how to rebalance
    5. **Key Risks**: Main risks to watch for this allocation

    Be specific and data-driven. Reference actual stocks, sectors, or metrics from the context.
    Include a disclaimer that this is for educational purposes only, not financial advice.
    """

    try:
        portfolio_advice = _invoke_llm(prompt)
        print(f"💼 Portfolio Agent: Generated allocation advice ({len(portfolio_advice)} chars)")

        return {
            **state,
            "portfolio_output": portfolio_advice,
            "agent_answer": portfolio_advice,
            "final_answer": portfolio_advice,
            "agents_invoked": agents_invoked,
        }

    except Exception as e:
        print(f"❌ Portfolio Agent error: {e}")
        fallback = f"Base allocation for {risk_appetite} risk appetite:\n{json.dumps(allocation_result, indent=2)}"
        return {
            **state,
            "portfolio_output": fallback,
            "agent_answer": fallback,
            "final_answer": fallback,
            "agents_invoked": agents_invoked,
            "error": str(e),
        }
