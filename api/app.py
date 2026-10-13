"""
FastAPI Backend Application for Finance Market Intelligence RAG.
Provides /query, /feedback, /health, /logs, and /ingest endpoints.
Orchestrates multi-agent LangGraph workflow and persists execution logs to database.
"""
import os
import time
import json
import traceback
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import (
    GEMINI_API_KEY,
    COHERE_API_KEY,
    FAISS_INDEX_PATH,
    RAW_DATA_DIR,
)
from src.ingestion.loader import load_all_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import create_vector_store, load_vector_store, get_embeddings
from src.retrieval.semantic_search import get_semantic_retriever
from src.retrieval.keyword_search import get_bm25_retriever
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.retrieval.reranker import get_reranked_retriever
from src.tools.rag_tool import setup_rag_tool
from src.graph.workflow import compile_workflow, get_initial_state

from api.database import init_db, get_db
from api.models import RAGLog, FeedbackLog
from api.schemas import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)

# ── Global State ──
_workflow_app = None
_vector_store = None
_retriever = None


def initialize_rag_system(force_rebuild: bool = False):
    """Initialize or load vector store, setup hybrid retriever & RAG tool, compile LangGraph."""
    global _workflow_app, _vector_store, _retriever

    print("🚀 Initializing Finance Market Intelligence System...")

    # 1. Ingestion / Vector Store Loading
    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print(f"📦 Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        try:
            _vector_store = load_vector_store(FAISS_INDEX_PATH)
        except Exception as e:
            print(f"⚠️ Error loading index: {e}, will attempt to rebuild...")
            _vector_store = None

    if _vector_store is None:
        print("🔨 Building vector store from raw data...")
        docs = load_all_documents(RAW_DATA_DIR)
        if docs:
            chunks = chunk_documents(docs)
            _vector_store = create_vector_store(chunks, index_path=FAISS_INDEX_PATH)
        else:
            print("⚠️ No raw data found in data/raw. Vector store will be initialized empty.")

    # 2. Setup Retrievers
    if _vector_store is not None:
        semantic_retriever = get_semantic_retriever(_vector_store)

        # For BM25, chunk documents if available
        docs = load_all_documents(RAW_DATA_DIR)
        if docs:
            chunks = chunk_documents(docs)
            bm25_retriever = get_bm25_retriever(chunks)
        else:
            bm25_retriever = semantic_retriever

        hybrid = get_hybrid_retriever(semantic_retriever, bm25_retriever)

        # Apply Cohere Rerank if key is present
        if COHERE_API_KEY:
            try:
                _retriever = get_reranked_retriever(hybrid)
            except Exception as e:
                print(f"⚠️ Cohere reranker setup failed ({e}), using hybrid retriever directly.")
                _retriever = hybrid
        else:
            print("ℹ️ Cohere API key not configured; using hybrid search retriever directly.")
            _retriever = hybrid

        # 3. Setup RAG tool
        setup_rag_tool(_retriever)

    # 4. Compile LangGraph workflow
    _workflow_app = compile_workflow()
    print("✨ LangGraph workflow ready for query execution!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    init_db()
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"⚠️ Warning during startup initialization: {e}")
        traceback.print_exc()
    yield
    print("🛑 Shutting down Finance Market Intelligence API...")


app = FastAPI(
    title="Finance Market Intelligence API",
    description="Agentic RAG for financial market insights, portfolio guidance, and risk assessment via LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure database tables exist immediately on import
init_db()

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Finance Market Intelligence API",
        "description": "Multi-agent RAG powered by LangGraph, FastAPI, and FAISS",
        "endpoints": {
            "query": "POST /query",
            "feedback": "POST /feedback",
            "health": "GET /health",
            "logs": "GET /logs",
            "ingest": "POST /ingest",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint verifying vector store, database, and LLM configuration."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    data_sources = []
    if os.path.exists(RAW_DATA_DIR):
        data_sources = [f for f in os.listdir(RAW_DATA_DIR) if not f.startswith(".")]

    vs_ok = (
        _vector_store is not None
        or (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")))
    )

    return HealthResponse(
        status="healthy" if (db_ok and vs_ok) else "degraded",
        vector_store_loaded=vs_ok,
        database_connected=db_ok,
        llm_configured=bool(GEMINI_API_KEY),
        available_data_sources=data_sources,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG & Agents"])
def process_query(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Execute a financial market query through the multi-agent LangGraph workflow.
    Routes dynamically through Planner → Retriever → Specialist → Validator.
    Persists query history to database.
    """
    global _workflow_app
    if _workflow_app is None:
        initialize_rag_system()

    start_time = time.time()

    initial_state = get_initial_state(
        query=req.query,
        session_id=req.session_id,
        risk_appetite=req.risk_appetite or "moderate",
    )

    try:
        # LangGraph execution with session thread configuration
        config = {"configurable": {"thread_id": req.session_id}}
        final_state = _workflow_app.invoke(initial_state, config=config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic graph execution failed: {str(e)}",
        )

    latency = round(time.time() - start_time, 2)
    answer = final_state.get("final_answer") or final_state.get("agent_answer") or "No answer generated."
    sources = final_state.get("sources", [])
    retrieved_docs = final_state.get("retrieved_docs", [])
    agents_invoked = final_state.get("agents_invoked", [])
    confidence = final_state.get("confidence", 0.8)
    intent = final_state.get("intent", "general")
    eval_verdict = final_state.get("evaluator_verdict", "approved")
    eval_score = final_state.get("evaluator_score", 85)

    # Risk assessment / Portfolio structures
    risk_assessment = None
    if final_state.get("risk_output"):
        risk_assessment = {
            "details": final_state.get("risk_output"),
            "risk_appetite": req.risk_appetite,
        }

    portfolio_suggestion = final_state.get("portfolio_output")

    # Persist log to DB
    try:
        log_entry = RAGLog(
            session_id=req.session_id,
            query=req.query,
            answer=answer,
            intent=intent,
            agents_invoked=json.dumps(agents_invoked),
            confidence=float(confidence),
            latency=latency,
            sources=json.dumps(sources),
            evaluator_verdict=eval_verdict,
            evaluator_score=int(eval_score) if isinstance(eval_score, (int, float)) else 85,
        )
        db.add(log_entry)
        db.commit()
    except Exception as db_err:
        print(f"⚠️ Error saving log to database: {db_err}")
        db.rollback()

    return QueryResponse(
        session_id=req.session_id,
        query=req.query,
        answer=answer,
        sources=sources,
        retrieved_context=retrieved_docs,
        agents_invoked=agents_invoked,
        confidence=confidence,
        intent=intent,
        risk_assessment=risk_assessment,
        portfolio_suggestion=portfolio_suggestion,
        latency=latency,
        evaluator_verdict=eval_verdict,
        evaluator_score=int(eval_score) if isinstance(eval_score, (int, float)) else 85,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback Loop"])
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Capture user rating (thumbs up/down) and qualitative feedback/corrections."""
    try:
        fb_entry = FeedbackLog(
            session_id=req.session_id,
            query=req.query,
            rating=req.rating,
            feedback_text=req.feedback_text,
            correction=req.correction,
        )
        db.add(fb_entry)
        db.commit()
        db.refresh(fb_entry)

        return FeedbackResponse(
            status="success",
            message="Feedback captured successfully. Thank you for helping refine the intelligence system!",
            feedback_id=fb_entry.id,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = Query(default=20, le=100), db: Session = Depends(get_db)):
    """Retrieve recent query execution logs and feedback for auditing and analytics."""
    queries = (
        db.query(RAGLog)
        .order_by(RAGLog.created_at.desc())
        .limit(limit)
        .all()
    )
    feedback = (
        db.query(FeedbackLog)
        .order_by(FeedbackLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "total_queries_returned": len(queries),
        "queries": [
            {
                "id": q.id,
                "session_id": q.session_id,
                "query": q.query,
                "answer": q.answer[:200] + "..." if q.answer and len(q.answer) > 200 else q.answer,
                "intent": q.intent,
                "agents_invoked": json.loads(q.agents_invoked) if q.agents_invoked else [],
                "confidence": q.confidence,
                "latency": q.latency,
                "evaluator_verdict": q.evaluator_verdict,
                "evaluator_score": q.evaluator_score,
                "created_at": q.created_at.isoformat() if q.created_at else None,
            }
            for q in queries
        ],
        "feedback": [
            {
                "id": fb.id,
                "session_id": fb.session_id,
                "rating": fb.rating,
                "feedback_text": fb.feedback_text,
                "correction": fb.correction,
                "created_at": fb.created_at.isoformat() if fb.created_at else None,
            }
            for fb in feedback
        ],
    }


@app.post("/ingest", tags=["Ingestion"])
def trigger_ingestion():
    """Trigger manual re-indexing of raw financial documents into the FAISS store."""
    try:
        initialize_rag_system(force_rebuild=True)
        return {"status": "success", "message": "Vector store re-indexed and refreshed successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
