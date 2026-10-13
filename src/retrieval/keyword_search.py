"""
Keyword search module using BM25 retriever for sparse retrieval.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import BM25_SEARCH_K


def get_bm25_retriever(documents: list[Document],
                       search_k: int = None,
                       k: int = None) -> BM25Retriever:
    """
    Create a BM25 keyword-based retriever from documents.
    BM25 excels at exact keyword matching, complementing semantic search.
    
    Args:
        documents: List of Document chunks for BM25 indexing.
        search_k: Number of documents to retrieve (defaults to config).
        k: Optional alias for search_k.
    
    Returns:
        A BM25Retriever instance.
    """
    final_k = search_k or k or BM25_SEARCH_K
    retriever = BM25Retriever.from_documents(documents=documents, k=final_k)
    print(f"✅ BM25 retriever initialized (k={final_k})")
    return retriever
