"""
Semantic search module using FAISS with MMR (Maximal Marginal Relevance).
"""
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from src.config import SEMANTIC_SEARCH_K


def get_semantic_retriever(vectorstore: FAISS,
                           search_k: int = None) -> BaseRetriever:
    """
    Create a semantic retriever using FAISS with MMR search.
    MMR balances relevance with diversity in retrieved documents.
    
    Args:
        vectorstore: FAISS vector store instance.
        search_k: Number of documents to retrieve (defaults to config).
    
    Returns:
        A LangChain retriever instance.
    """
    k = search_k or SEMANTIC_SEARCH_K
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    )
    print(f"✅ Semantic retriever initialized (MMR, k={k})")
    return retriever
