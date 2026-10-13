"""
Embedding and vector store module for the Finance Market Intelligence RAG system.
Creates HuggingFace embeddings and stores them in a FAISS vector store.
"""
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH

import warnings
warnings.filterwarnings("ignore")


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Initialize and return HuggingFace embedding model.
    
    Args:
        model_name: HuggingFace model name (defaults to config EMBEDDING_MODEL).
    
    Returns:
        HuggingFaceEmbeddings instance.
    """
    model_name = model_name or EMBEDDING_MODEL
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"✅ Initialized embedding model: {model_name}")
    return embeddings


def create_vector_store(chunks: list[Document],
                        embeddings: HuggingFaceEmbeddings = None,
                        index_path: str = None) -> FAISS:
    """
    Create a FAISS vector store from document chunks and persist to disk.
    
    Args:
        chunks: List of Document chunks to embed.
        embeddings: Embedding model (creates default if None).
        index_path: Path to save the FAISS index (defaults to config FAISS_INDEX_PATH).
    
    Returns:
        FAISS vector store instance.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    index_path = index_path or FAISS_INDEX_PATH

    print(f"🔄 Creating FAISS vector store with {len(chunks)} chunks...")
    faiss_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Persist to disk
    faiss_store.save_local(index_path)
    print(f"✅ FAISS index saved to: {index_path}")

    return faiss_store


def load_vector_store(index_path: str = None,
                      embeddings: HuggingFaceEmbeddings = None) -> FAISS:
    """
    Load a persisted FAISS vector store from disk.
    
    Args:
        index_path: Path to the FAISS index directory.
        embeddings: Embedding model (creates default if None).
    
    Returns:
        FAISS vector store instance.
    """
    index_path = index_path or FAISS_INDEX_PATH
    if embeddings is None:
        embeddings = get_embeddings()

    faiss_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ FAISS index loaded from: {index_path}")
    return faiss_store
