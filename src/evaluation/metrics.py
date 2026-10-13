"""
Evaluation module - Measures RAG pipeline quality using BLEU, ROUGE, and relevance scores.
Compares baseline RAG vs. agentic RAG performance.
"""
import time
from collections import Counter


def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """
    Compute BLEU score between reference and candidate texts.
    Uses a simple n-gram precision approach.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        cand_ngrams = Counter(
            tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)
        )

        matches = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())

        if total == 0:
            scores.append(0.0)
        else:
            scores.append(matches / total)

    # Geometric mean of precisions
    if any(s == 0 for s in scores):
        return 0.0

    import math
    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = min(1.0, len(cand_tokens) / max(len(ref_tokens), 1))

    return round(bp * math.exp(log_avg), 4)


def compute_rouge_l(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-L score using Longest Common Subsequence.
    Returns precision, recall, and F1.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute LCS length
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    precision = lcs_length / n if n > 0 else 0.0
    recall = lcs_length / m if m > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_relevance_score(query: str, answer: str, context: list[str]) -> float:
    """
    Compute a simple term-overlap relevance score between query+context and the answer.
    Returns a score between 0.0 and 1.0.
    """
    query_terms = set(query.lower().split())
    context_terms = set()
    for c in context:
        context_terms.update(c.lower().split())

    combined_terms = query_terms | context_terms
    answer_terms = set(answer.lower().split())

    if not combined_terms:
        return 0.0

    overlap = combined_terms & answer_terms
    return round(len(overlap) / len(combined_terms), 4)


def evaluate_response(query: str, answer: str, reference: str = "",
                      context: list[str] = None) -> dict:
    """
    Run all evaluation metrics on a response.
    
    Args:
        query: The original user query.
        answer: The generated answer.
        reference: Optional reference/gold answer for BLEU/ROUGE.
        context: Optional list of retrieved document texts.
    
    Returns:
        Dictionary with all metric scores.
    """
    context = context or []
    results = {}

    if reference:
        results["bleu"] = compute_bleu(reference, answer)
        results["rouge_l"] = compute_rouge_l(reference, answer)
    else:
        results["bleu"] = None
        results["rouge_l"] = None

    results["relevance"] = compute_relevance_score(query, answer, context)

    return results


def run_baseline_vs_agentic_comparison(test_queries: list[dict],
                                        baseline_fn, agentic_fn) -> list[dict]:
    """
    Compare baseline RAG vs. agentic RAG on a set of test queries.
    
    Args:
        test_queries: List of dicts with 'query' and optional 'reference'.
        baseline_fn: Function(query) -> str for baseline RAG.
        agentic_fn: Function(query) -> dict for agentic RAG.
    
    Returns:
        List of comparison results.
    """
    results = []

    for item in test_queries:
        query = item["query"]
        reference = item.get("reference", "")

        # Baseline
        start = time.time()
        baseline_answer = baseline_fn(query)
        baseline_latency = time.time() - start
        baseline_metrics = evaluate_response(query, baseline_answer, reference)

        # Agentic
        start = time.time()
        agentic_result = agentic_fn(query)
        agentic_latency = time.time() - start
        agentic_answer = agentic_result.get("final_answer", "")
        agentic_context = agentic_result.get("retrieved_docs", [])
        agentic_metrics = evaluate_response(query, agentic_answer, reference, agentic_context)

        results.append({
            "query": query,
            "baseline": {
                "answer": baseline_answer[:200],
                "metrics": baseline_metrics,
                "latency": round(baseline_latency, 2),
            },
            "agentic": {
                "answer": agentic_answer[:200],
                "metrics": agentic_metrics,
                "latency": round(agentic_latency, 2),
                "agents_invoked": agentic_result.get("agents_invoked", []),
                "confidence": agentic_result.get("confidence", 0.0),
            },
        })

    return results
