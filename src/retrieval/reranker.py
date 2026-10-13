"""
Re-ranking module using Cohere Rerank for improved retrieval precision.
Wraps the hybrid retriever with ContextualCompressionRetriever.
"""
from langchain_core.retrievers import BaseRetriever
from src.config import COHERE_API_KEY, RERANK_TOP_N

try:
    from langchain_cohere import CohereRerank
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


def get_reranked_retriever(base_retriever: BaseRetriever,
                           top_n: int = None,
                           cohere_api_key: str = None) -> BaseRetriever:
    """
    Wrap a base retriever with Cohere Rerank for improved precision if available.
    Falls back gracefully to the base retriever if Cohere is not configured or installed.
    """
    top_n = top_n or RERANK_TOP_N
    api_key = cohere_api_key or COHERE_API_KEY

    if not COHERE_AVAILABLE or not api_key:
        print("ℹ️ Cohere Rerank not available or API key missing; using base retriever.")
        return base_retriever

    try:
        reranker = CohereRerank(
            cohere_api_key=api_key,
            model="rerank-v3.5",
            top_n=top_n
        )
        reranked_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
        print(f"✅ Reranked retriever initialized (Cohere rerank-v3.5, top_n={top_n})")
        return reranked_retriever
    except Exception as e:
        print(f"⚠️ Cohere reranker init failed ({e}), using base retriever.")
        return base_retriever
