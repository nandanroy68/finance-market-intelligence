"""
Standalone ingestion script for Finance Market Intelligence.
Loads BFSI FAQs, Market Reports, and Stock Data, chunks them, and builds the FAISS vector index.
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure src is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import RAW_DATA_DIR, FAISS_INDEX_PATH
from src.ingestion.loader import load_all_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import create_vector_store


def main():
    print("=" * 60)
    print("📈 Finance Market Intelligence - Vector Store Ingestion")
    print("=" * 60)

    print(f"\n1. Scanning raw data directory: {RAW_DATA_DIR}")
    docs = load_all_documents(RAW_DATA_DIR)
    if not docs:
        print("❌ No documents found in data/raw. Please add JSON/CSV/PDF files.")
        return

    print(f"\n2. Chunking {len(docs)} documents...")
    chunks = chunk_documents(docs)
    print(f"✅ Generated {len(chunks)} chunks with overlapping window.")

    print(f"\n3. Embedding chunks and building FAISS index...")
    create_vector_store(chunks, index_path=FAISS_INDEX_PATH)
    print(f"🎉 Ingestion complete! FAISS index persisted at: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
