"""
Document loader module for the Finance Market Intelligence RAG system.
Loads financial data from JSON, CSV, and PDF sources with metadata preservation.
"""
import os
import json
from langchain_community.document_loaders import JSONLoader, CSVLoader, PyPDFLoader
from langchain_core.documents import Document

import warnings
warnings.filterwarnings("ignore")


def _faq_metadata_func(record: dict, metadata: dict) -> dict:
    """Extract metadata from BFSI FAQ records."""
    metadata["id"] = record.get("id", "")
    metadata["category"] = record.get("category", "")
    metadata["source"] = record.get("source", "")
    metadata["doc_type"] = "faq"
    return metadata


def _report_metadata_func(record: dict, metadata: dict) -> dict:
    """Extract metadata from market report records."""
    metadata["id"] = record.get("id", "")
    metadata["title"] = record.get("title", "")
    metadata["category"] = record.get("category", "")
    metadata["sector"] = record.get("sector", "")
    metadata["date"] = record.get("date", "")
    metadata["sentiment"] = record.get("sentiment", "")
    metadata["source"] = record.get("source", "")
    metadata["doc_type"] = "market_report"
    return metadata


def load_bfsi_faqs(file_path: str) -> list[Document]:
    """
    Load BFSI FAQ documents from a JSON file.
    Each FAQ is converted to a Document with Q&A as content and metadata preserved.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    documents = []
    for faq in faqs:
        content = (
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}"
        )
        metadata = {
            "id": faq.get("id", ""),
            "category": faq.get("category", ""),
            "source": faq.get("source", ""),
            "doc_type": "faq",
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"✅ Loaded {len(documents)} BFSI FAQ documents")
    return documents


def load_market_reports(file_path: str) -> list[Document]:
    """
    Load market analysis reports from a JSON file.
    Each report is converted to a Document with full content and rich metadata.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reports = json.load(f)

    documents = []
    for report in reports:
        content = (
            f"Title: {report['title']}\n"
            f"Date: {report['date']}\n"
            f"Sector: {report.get('sector', 'General')}\n"
            f"Content: {report['content']}"
        )
        metadata = {
            "id": report.get("id", ""),
            "title": report.get("title", ""),
            "category": report.get("category", ""),
            "sector": report.get("sector", ""),
            "date": report.get("date", ""),
            "sentiment": report.get("sentiment", ""),
            "source": report.get("source", ""),
            "doc_type": "market_report",
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"✅ Loaded {len(documents)} market report documents")
    return documents


def load_stock_data(file_path: str) -> list[Document]:
    """
    Load stock market data from a CSV file using CSVLoader.
    Each row becomes a Document with company financial metrics.
    """
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    # Enrich metadata
    for doc in documents:
        doc.metadata["doc_type"] = "stock_data"
        doc.metadata["source"] = "Stock Market Data"

    print(f"✅ Loaded {len(documents)} stock data documents")
    return documents


def load_pdf_documents(pdf_dir: str) -> list[Document]:
    """
    Load all PDF documents from a directory.
    Useful for SEC filings, annual reports, etc.
    """
    documents = []
    if not os.path.exists(pdf_dir):
        print(f"⚠️ PDF directory not found: {pdf_dir}")
        return documents

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(file_path=pdf_path)
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata["doc_type"] = "pdf_report"
            doc.metadata["source"] = pdf_file
        documents.extend(pdf_docs)

    print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF files")
    return documents


def load_all_documents(data_dir: str) -> list[Document]:
    """
    Load all documents from the raw data directory.
    Combines BFSI FAQs, market reports, stock data, and any PDFs.
    """
    all_documents = []

    # Load BFSI FAQs
    faq_path = os.path.join(data_dir, "bfsi_faqs.json")
    if os.path.exists(faq_path):
        all_documents.extend(load_bfsi_faqs(faq_path))

    # Load Market Reports
    report_path = os.path.join(data_dir, "market_reports.json")
    if os.path.exists(report_path):
        all_documents.extend(load_market_reports(report_path))

    # Load Stock Data
    stock_path = os.path.join(data_dir, "stock_data.csv")
    if os.path.exists(stock_path):
        all_documents.extend(load_stock_data(stock_path))

    # Load PDFs (if any exist in the raw data dir)
    all_documents.extend(load_pdf_documents(data_dir))

    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    return all_documents
