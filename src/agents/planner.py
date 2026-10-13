"""
Planner Agent - Classifies user intent and routes to the appropriate specialist agent.
"""
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.config import GEMINI_API_KEY, GEMINI_MODEL


def _invoke_llm(prompt: str) -> str:
    """Invoke the LLM and return content as a string."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        max_output_tokens=256,
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


def planner(state: AgentState) -> AgentState:
    """
    Classify the user's query into one of the financial intents.
    Routes to: market_info, analysis, portfolio, or risk.
    """
    prompt = f"""
    Classify the user's financial query into exactly one intent:
    - market_info: Questions about stock market basics, regulations, banking policies, FAQs,
      financial terms, sector information, or general financial knowledge.
    - analysis: Requests for financial analysis, insights, market trends, sector outlook,
      company performance evaluation, or data-driven assessments.
    - portfolio: Questions about asset allocation, investment planning, portfolio construction,
      SIP strategies, or investment recommendations based on risk appetite.
    - risk: Questions about risk assessment, volatility, VaR, portfolio beta, stress testing,
      credit risk, or market risk evaluation.

    User query:
    {state['query']}

    Return ONLY one word: market_info, analysis, portfolio, or risk.
    Do NOT include any explanation, markdown, or extra text.
    """

    raw_response = _invoke_llm(prompt).strip().lower()

    # Clean up any unexpected formatting
    raw_response = re.sub(r'[^a-z_]', '', raw_response)

    valid_intents = {"market_info", "analysis", "portfolio", "risk"}
    intent = raw_response if raw_response in valid_intents else "market_info"

    agents_invoked = state.get("agents_invoked", []) or []
    agents_invoked = list(agents_invoked) + ["planner"]

    print(f"🧠 Planner: query='{state['query'][:60]}...' → intent={intent}")

    return {
        **state,
        "intent": intent,
        "agents_invoked": agents_invoked,
    }
