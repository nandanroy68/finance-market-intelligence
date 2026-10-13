"""
Text chunking module for the Finance Market Intelligence RAG system.
Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(documents: list[Document],
                  chunk_size: int = None,
                  chunk_overlap: int = None) -> list[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum size of each chunk (defaults to config CHUNK_SIZE).
        chunk_overlap: Overlap between consecutive chunks (defaults to config CHUNK_OVERLAP).
    
    Returns:
        List of chunked Document objects with metadata preserved.
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# Alias for compatibility
chunk_documents = create_chunks
