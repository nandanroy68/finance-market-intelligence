"""
RAG Tool module - Exposes the RAG pipeline as a callable tool for agents.
This is the core shared tool that all specialist agents can invoke.
"""
import time
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# ── Module-level state (initialized by setup_rag_tool) ──
_rag_chain = None
_retriever = None


def setup_rag_tool(retriever: BaseRetriever):
    """
    Initialize the RAG chain with the given retriever using modern LCEL.
    Must be called once before using the rag_query tool.
    """
    global _rag_chain, _retriever
    _retriever = retriever

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        google_api_key=GEMINI_API_KEY,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are a Financial Market Intelligence expert with deep knowledge of Indian and global markets,
    banking regulations, investment strategies, risk management, and portfolio construction.
    
    Use ONLY the information provided in the context to answer the user's question.
    Provide clear, accurate, and data-driven responses. Include specific numbers, percentages,
    and references where available. If the context does not contain enough information,
    say "The answer is not available in the provided context."
    
    Always structure your response with:
    1. A direct answer to the question
    2. Supporting data/evidence from the context
    3. Any relevant caveats or disclaimers
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    )

    chain = prompt | llm | StrOutputParser()

    class RAGExecutor:
        def __init__(self, ret, ch):
            self.retriever = ret
            self.chain = ch

        def invoke(self, inputs: dict) -> dict:
            q = inputs.get("question", "")
            docs = self.retriever.invoke(q)
            context = "\n\n".join([doc.page_content for doc in docs])
            ans = self.chain.invoke({"context": context, "question": q})
            return {
                "answer": ans,
                "source_documents": docs,
            }

    _rag_chain = RAGExecutor(_retriever, chain)
    print("✅ RAG tool initialized")
    return _rag_chain


@tool
def rag_query(question: str) -> dict:
    """
    Query the financial knowledge base using the RAG pipeline.
    Retrieves relevant financial documents and generates a grounded answer.
    Returns the answer, retrieved documents, sources, and latency.
    """
    global _rag_chain
    if _rag_chain is None:
        return {"error": "RAG tool not initialized. Call setup_rag_tool() first."}

    start = time.time()
    result = _rag_chain.invoke({"question": question})
    latency = time.time() - start

    docs = result.get("source_documents", [])
    retrieved_docs = [doc.page_content[:300] for doc in docs]
    sources = [doc.metadata for doc in docs]

    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": round(latency, 2),
    }
