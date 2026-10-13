"""
Tests for document loading, chunking, and retrieval pipeline.
"""
import os
import pytest
from langchain_core.documents import Document

from src.ingestion.loader import load_bfsi_faqs, load_market_reports, load_stock_data, load_all_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.keyword_search import get_bm25_retriever
from src.config import RAW_DATA_DIR


def test_load_bfsi_faqs():
    faq_path = os.path.join(RAW_DATA_DIR, "bfsi_faqs.json")
    if os.path.exists(faq_path):
        docs = load_bfsi_faqs(faq_path)
        assert len(docs) > 0
        assert "Question:" in docs[0].page_content
        assert docs[0].metadata["doc_type"] == "faq"


def test_load_market_reports():
    report_path = os.path.join(RAW_DATA_DIR, "market_reports.json")
    if os.path.exists(report_path):
        docs = load_market_reports(report_path)
        assert len(docs) > 0
        assert "Title:" in docs[0].page_content
        assert docs[0].metadata["doc_type"] == "market_report"


def test_load_stock_data():
    stock_path = os.path.join(RAW_DATA_DIR, "stock_data.csv")
    if os.path.exists(stock_path):
        docs = load_stock_data(stock_path)
        assert len(docs) > 0
        assert docs[0].metadata["doc_type"] == "stock_data"


def test_chunking():
    sample_docs = [
        Document(
            page_content="Financial Market Intelligence Report. " * 50,
            metadata={"source": "test_report.pdf", "doc_type": "report"}
        )
    ]
    chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "test_report.pdf"


def test_bm25_retriever():
    docs = [
        Document(page_content="Reserve Bank of India increased repo rate by 25 basis points.", metadata={"id": "1"}),
        Document(page_content="Tech Mahindra and Infosys announced AI partnership.", metadata={"id": "2"}),
        Document(page_content="HDFC Bank reported net profit growth of 18% YoY.", metadata={"id": "3"}),
    ]
    retriever = get_bm25_retriever(docs, k=1)
    results = retriever.invoke("repo rate RBI")
    assert len(results) >= 1
    assert "Reserve Bank" in results[0].page_content
