"""
Hybrid search module combining semantic (FAISS MMR) and keyword (BM25) retrieval
using LangChain's EnsembleRetriever with configurable weights.
"""
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.config import HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT


class EnsembleRetriever(BaseRetriever):
    """
    Hybrid Ensemble Retriever that merges results from multiple retrievers
    using Weighted Reciprocal Rank Fusion (RRF).
    """
    retrievers: List[BaseRetriever]
    weights: List[float] = [0.6, 0.4]
    c: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        doc_scores = {}
        doc_map = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query)
            except Exception:
                docs = []

            for rank, doc in enumerate(docs):
                doc_key = doc.page_content.strip()
                doc_map[doc_key] = doc
                score = weight * (1.0 / (rank + self.c))
                doc_scores[doc_key] = doc_scores.get(doc_key, 0.0) + score

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]


def get_hybrid_retriever(semantic_retriever: BaseRetriever,
                         bm25_retriever: BaseRetriever,
                         semantic_weight: float = None,
                         bm25_weight: float = None) -> EnsembleRetriever:
    """
    Create a hybrid retriever combining semantic and BM25 retrievers.
    Uses EnsembleRetriever to merge results with weighted scoring.
    """
    sw = semantic_weight or HYBRID_SEMANTIC_WEIGHT
    bw = bm25_weight or HYBRID_BM25_WEIGHT

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[sw, bw]
    )
    print(f"✅ Hybrid retriever initialized (semantic={sw}, BM25={bw})")
    return hybrid_retriever
