# Config and environment setup for Finance Market Intelligence RAG
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Configuration ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ── Embedding Configuration ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Vector Store ──
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ── Retrieval Configuration ──
SEMANTIC_SEARCH_K = int(os.getenv("SEMANTIC_SEARCH_K", "8"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "8"))
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ── Chunking Configuration ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Database Configuration ──
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./finance_rag.db")

# ── Agent Configuration ──
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))

# ── Data Paths ──
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
