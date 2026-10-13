"""
Validator Agent - Checks answer grounding and relevance; triggers retry if quality is low.
Implements the feedback loop with query rewriting and re-retrieval.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_RETRY_ATTEMPTS, RELEVANCE_THRESHOLD


def _invoke_llm(prompt: str) -> str:
    """Invoke the LLM and return content as a string."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        max_output_tokens=512,
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


def validator(state: AgentState) -> AgentState:
    """
    Validate the agent's response for grounding, relevance, and quality.
    If validation fails, trigger a retry with a rewritten query.
    """
    answer = state.get("agent_answer", "")
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    attempts = state.get("attempts", 0) + 1
    agents_invoked = list(state.get("agents_invoked", []) or []) + ["validator"]

    # If no answer was generated
    if not answer or answer.strip() == "":
        print(f"✅ Validator: No answer to validate (attempt {attempts})")
        return {
            **state,
            "evaluator_verdict": "needs_retry",
            "evaluator_score": 1,
            "evaluator_feedback": "No answer was generated. Re-querying.",
            "confidence": 0.0,
            "attempts": attempts,
            "agents_invoked": agents_invoked,
        }

    # Use LLM to evaluate the response
    context_summary = "\n".join(retrieved_docs[:3]) if retrieved_docs else "(no context retrieved)"

    prompt = f"""
    Evaluate the following response for quality and grounding.
    
    User Question: {query}
    
    Retrieved Context (summary):
    {context_summary}
    
    Generated Answer:
    {answer[:1500]}
    
    Evaluate on these criteria and return ONLY a JSON object:
    1. relevance: Is the answer relevant to the question? (0.0-1.0)
    2. grounding: Is the answer grounded in the retrieved context? (0.0-1.0)
    3. completeness: Does the answer fully address the question? (0.0-1.0)
    4. verdict: "approved" if all scores >= 0.6, otherwise "needs_retry"
    5. feedback: Brief explanation of the evaluation

    Return ONLY valid JSON, no markdown, no code fences:
    {{"relevance": 0.0, "grounding": 0.0, "completeness": 0.0, "verdict": "approved", "feedback": "..."}}
    """

    try:
        import json, re

        eval_response = _invoke_llm(prompt).strip()
        # Clean markdown fences
        eval_response = re.sub(r"^```json\s*", "", eval_response, flags=re.IGNORECASE)
        eval_response = re.sub(r"^```\s*", "", eval_response)
        eval_response = re.sub(r"\s*```$", "", eval_response)
        eval_response = eval_response.strip()

        eval_data = json.loads(eval_response)

        relevance = float(eval_data.get("relevance", 0.5))
        grounding = float(eval_data.get("grounding", 0.5))
        completeness = float(eval_data.get("completeness", 0.5))
        confidence = round((relevance + grounding + completeness) / 3, 2)
        verdict = eval_data.get("verdict", "approved")
        feedback = eval_data.get("feedback", "Evaluation completed.")

        # Override verdict if confidence is below threshold
        if confidence < RELEVANCE_THRESHOLD and attempts < MAX_RETRY_ATTEMPTS:
            verdict = "needs_retry"
            feedback += f" Confidence {confidence} below threshold {RELEVANCE_THRESHOLD}."

        # Map to score (1-5)
        score = max(1, min(5, int(confidence * 5)))

        print(f"✅ Validator: verdict={verdict}, confidence={confidence}, score={score}, attempt={attempts}")

    except Exception as e:
        print(f"⚠️ Validator: LLM evaluation failed ({e}), defaulting to approved")
        verdict = "approved"
        score = 3
        confidence = 0.6
        feedback = f"Evaluation parsing failed: {str(e)}. Defaulting to approved."

    # Force approval if max retries exceeded
    if attempts >= MAX_RETRY_ATTEMPTS and verdict == "needs_retry":
        verdict = "approved"
        feedback += f" Max retries ({MAX_RETRY_ATTEMPTS}) reached. Approving best available response."

    return {
        **state,
        "evaluator_verdict": verdict,
        "evaluator_score": score,
        "evaluator_feedback": feedback,
        "confidence": confidence,
        "attempts": attempts,
        "final_answer": answer,
        "agents_invoked": agents_invoked,
    }
