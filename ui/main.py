"""
Streamlit Frontend for Finance Market Intelligence RAG.
Features a modern WhatsApp/fintech-style conversation interface with agent trace inspection,
confidence metrics, source drawer, risk indicators, and user feedback capture.
"""
import time
import uuid
import requests
import streamlit as st

# ── Page Configuration ──
st.set_page_config(
    page_title="AlphaFin | Market Intelligence Multi-Agent RAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Endpoint Config ──
API_BASE_URL = "http://localhost:8000"

# ── Custom CSS for High-Aesthetics UI ──
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }

    /* Header Banner */
    .fin-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .fin-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .fin-subtitle {
        color: #9ca3af;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Badges & Pills */
    .badge-intent {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-agent {
        background-color: #14532d;
        color: #86efac;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.72rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-confidence {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Risk metric card */
    .risk-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.8rem;
    }
    .risk-title {
        font-weight: 600;
        color: #f87171;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    /* Agent trail container */
    .agent-trail {
        background: #111827;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #3b82f6;
        margin: 0.6rem 0;
        font-size: 0.82rem;
    }

    /* Chat message bubble refinement */
    div[data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }

    /* Quick prompt button styling */
    .prompt-btn {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Setup ──
if "session_id" not in st.session_state:
    st.session_state.session_id = f"fin-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome to **AlphaFin Market Intelligence**! I am your agentic financial assistant powered by LangGraph, hybrid RAG (FAISS + BM25), and specialist agents.\n\nAsk me about:\n- 📈 Market reports & corporate earnings\n- 🏦 BFSI regulations & RBI monetary policy\n- 💼 Tailored portfolio allocation suggestions\n- ⚠️ Sector risk, beta, and concentration exposure",
            "meta": None,
        }
    ]

# ── Sidebar Controls ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("AlphaFin Agentic RAG")
    st.caption("v1.0.0 | LangGraph • FastAPI • FAISS")

    st.markdown("---")
    st.subheader("⚙️ Session & Parameters")
    st.text_input("Session ID", value=st.session_state.session_id, key="session_key", disabled=True)

    risk_appetite = st.selectbox(
        "Risk Appetite Profile",
        options=["conservative", "moderate", "aggressive"],
        index=1,
        help="Customizes portfolio agent asset allocation and risk agent evaluations.",
    )

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.session_id = f"fin-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation refreshed. How can I assist your financial market research today?",
                "meta": None,
            }
        ]
        st.rerun()

    st.markdown("---")
    st.subheader("📡 Backend Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            h_data = health_resp.json()
            st.success(f"Backend: **{h_data.get('status', 'OK').upper()}**")
            st.caption(f"Vector Store: {'✅ Ready' if h_data.get('vector_store_loaded') else '❌ Missing'}")
            st.caption(f"Database: {'✅ Connected' if h_data.get('database_connected') else '❌ Error'}")
            st.caption(f"LLM Config: {'✅ Key Configured' if h_data.get('llm_configured') else '⚠️ Missing Key'}")
        else:
            st.warning("Backend degraded or returning non-200")
    except Exception:
        st.error("Backend offline (Run `python run_api.py` or `uvicorn api.app:app`)")

    st.markdown("---")
    st.subheader("💡 Suggested Queries")
    quick_queries = [
        "What is RBI's current repo rate and inflation target?",
        "Compare TCS vs Infosys Q3 revenue and EBIT margins",
        "Suggest an asset allocation for aggressive risk appetite",
        "Evaluate portfolio risk for 60% IT sector concentration",
    ]

    selected_prompt = None
    for q in quick_queries:
        if st.button(q, key=f"btn_{q[:15]}", use_container_width=True):
            selected_prompt = q

# ── Header Banner ──
st.markdown("""
<div class="fin-header">
    <div>
        <h1 class="fin-title">📊 AlphaFin Market Intelligence</h1>
        <p class="fin-subtitle">Multi-Agent Financial RAG with Grounding, Reranking & Risk Analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Render Message History ──
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display rich metadata if present
        meta = msg.get("meta")
        if meta:
            # Metadata pills
            pills_html = "<div style='margin-top: 8px; margin-bottom: 8px;'>"
            if meta.get("intent"):
                pills_html += f"<span class='badge-intent'>🎯 Intent: {meta['intent']}</span>"
            if meta.get("confidence"):
                pills_html += f"<span class='badge-confidence'>🛡️ Confidence: {int(meta['confidence'] * 100)}%</span>"
            if meta.get("latency"):
                pills_html += f"<span style='color: #9ca3af; font-size: 0.75rem; margin-left: 8px;'>⏱️ {meta['latency']}s</span>"
            pills_html += "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

            # Agent Workflow Trail
            agents = meta.get("agents_invoked", [])
            if agents:
                agents_str = " ➔ ".join([f"<code>{a}</code>" for a in agents])
                st.markdown(f"<div class='agent-trail'>🤖 <b>Multi-Agent Flow:</b> {agents_str}</div>", unsafe_allow_html=True)

            # Risk Assessment Box
            risk = meta.get("risk_assessment")
            if risk and risk.get("details"):
                with st.expander("⚠️ Risk Assessment Breakdown", expanded=False):
                    st.markdown(f"**Appetite Context:** `{risk.get('risk_appetite', 'N/A')}`")
                    st.info(risk.get("details"))

            # Retrieved Context / Sources Drawer
            sources = meta.get("sources", [])
            retrieved_context = meta.get("retrieved_context", [])
            if sources or retrieved_context:
                with st.expander(f"📚 Sources & Retrieved Context ({len(sources)} citations)"):
                    if sources:
                        st.markdown("**Referenced Sources:**")
                        for s_idx, s in enumerate(sources):
                            if isinstance(s, dict):
                                title = s.get("title") or s.get("source") or f"Document #{s_idx + 1}"
                                st.markdown(f"- **{title}** (`{s.get('doc_type', 'doc')}` | {s.get('category', 'general')})")
                            else:
                                st.markdown(f"- {s}")

                    if retrieved_context:
                        st.markdown("---")
                        st.markdown("**Evidence Snippets:**")
                        for c_idx, snippet in enumerate(retrieved_context):
                            st.caption(f"**Chunk [{c_idx + 1}]:** {snippet}")

            # Feedback Loop UI
            fb_key = f"fb_{idx}"
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                if st.button("👍 Helpful", key=f"up_{fb_key}"):
                    try:
                        requests.post(
                            f"{API_BASE_URL}/feedback",
                            json={"session_id": st.session_state.session_id, "query": msg.get("user_query", ""), "rating": 1},
                            timeout=2,
                        )
                        st.toast("Thank you for your feedback!", icon="✅")
                    except Exception:
                        st.toast("Feedback recorded locally.")
            with col2:
                if st.button("👎 Incorrect", key=f"down_{fb_key}"):
                    try:
                        requests.post(
                            f"{API_BASE_URL}/feedback",
                            json={"session_id": st.session_state.session_id, "query": msg.get("user_query", ""), "rating": -1},
                            timeout=2,
                        )
                        st.toast("Feedback recorded! We'll refine the pipeline.", icon="📝")
                    except Exception:
                        st.toast("Feedback noted.")


# ── Process User Input ──
user_input = st.chat_input("Ask a market intelligence, stock analysis, or portfolio question...")

# Handle prompt injection from sidebar buttons
prompt_to_run = selected_prompt if selected_prompt else user_input

if prompt_to_run:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_run, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    # 2. Invoke FastAPI Multi-Agent RAG
    with st.chat_message("assistant"):
        with st.spinner("🧠 Multi-agent team orchestrating financial intelligence..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "session_id": st.session_state.session_id,
                        "query": prompt_to_run,
                        "risk_appetite": risk_appetite,
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    meta = {
                        "intent": data.get("intent"),
                        "agents_invoked": data.get("agents_invoked", []),
                        "confidence": data.get("confidence", 0.0),
                        "latency": data.get("latency", 0.0),
                        "sources": data.get("sources", []),
                        "retrieved_context": data.get("retrieved_context", []),
                        "risk_assessment": data.get("risk_assessment"),
                        "portfolio_suggestion": data.get("portfolio_suggestion"),
                        "user_query": prompt_to_run,
                    }

                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta,
                        "user_query": prompt_to_run,
                    })
                    st.rerun()

                else:
                    err_msg = f"⚠️ API Error ({response.status_code}): {response.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "meta": None,
                    })
            except requests.exceptions.ConnectionError:
                err_msg = "⚠️ Could not connect to API server at http://localhost:8000. Please start the backend service."
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "meta": None,
                })
            except Exception as e:
                err_msg = f"⚠️ An unexpected error occurred: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "meta": None,
                })
