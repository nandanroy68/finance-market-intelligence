"""
Risk Assessment Agent - Evaluates market volatility and portfolio exposure.
Can be invoked as a tool by the Portfolio Agent (Agents-as-Tools pattern).
"""
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.tools.market_data_tool import calculate_portfolio_risk
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


def risk_agent(state: AgentState) -> AgentState:
    """
    Evaluate market volatility and portfolio risk exposure.
    Uses retrieved market data to assess risk factors.
    """
    query = state["query"]
    retrieved_context = "\n---\n".join(state.get("retrieved_docs", []))
    analysis_output = state.get("analysis_output", "")
    agents_invoked = list(state.get("agents_invoked", []) or []) + ["risk_assessment"]

    prompt = f"""
    You are a Risk Management Specialist with expertise in financial markets.
    Evaluate the risk factors based on the user's query and available market data.

    User Query: {query}

    Market Analysis:
    {analysis_output}

    Retrieved Context:
    {retrieved_context}

    Provide a comprehensive risk assessment:
    1. **Market Risk**: Current market volatility indicators (VIX levels, beta analysis)
    2. **Sector Risk**: Concentration risks and sector-specific concerns
    3. **Macroeconomic Risk**: GDP, inflation, interest rate, and currency risks
    4. **Liquidity Risk**: Market depth and liquidity considerations
    5. **Risk Mitigation Strategies**: Specific actions to reduce identified risks
    6. **Risk Score**: Overall risk rating (1-10, where 10 is highest risk)

    Base your assessment ONLY on provided context. Be specific with data points.
    Include VaR estimation methodology where applicable.
    """

    try:
        risk_assessment = _invoke_llm(prompt)
        print(f"⚠️ Risk Agent: Generated risk assessment ({len(risk_assessment)} chars)")

        return {
            **state,
            "risk_output": risk_assessment,
            "agent_answer": risk_assessment,
            "final_answer": risk_assessment,
            "agents_invoked": agents_invoked,
        }

    except Exception as e:
        print(f"❌ Risk Agent error: {e}")
        fallback = "Unable to complete risk assessment. Please provide more specific query details."
        return {
            **state,
            "risk_output": fallback,
            "agent_answer": fallback,
            "final_answer": fallback,
            "agents_invoked": agents_invoked,
            "error": str(e),
        }
