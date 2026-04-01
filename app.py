"""NeMo AgentIQ — Streamlit GUI for Enterprise Agentic RAG."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import OrchestratorAgent
from agents.doc_agent import DocAgent
from agents.sql_agent import SQLAgent
from agents.web_agent import WebAgent
from agents.api_agent import APIAgent
from agents.synthesis import SynthesisAgent

st.set_page_config(page_title="NeMo AgentIQ", page_icon="⚡", layout="wide")

# ── Session state init ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_agent" not in st.session_state:
    st.session_state.doc_agent = DocAgent()

doc_agent = st.session_state.doc_agent

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ NeMo AgentIQ")
    st.caption("Enterprise Agentic RAG with Multi-Source Reasoning")

    # ── Document upload ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop files to ingest into RAG",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if uploaded_files:
        if st.button("Ingest Documents", type="primary"):
            with st.spinner("Embedding and indexing..."):
                chunk_count = doc_agent.ingest_uploaded_files(uploaded_files)
            st.success(f"Ingested {len(uploaded_files)} file(s) → {chunk_count} chunks")

    # ── Data source toggles ──────────────────────────────────────────────
    st.divider()
    st.subheader("Data Sources")
    use_docs = st.checkbox("Documents (RAG)", value=True)
    use_sql = st.checkbox("SQL Database", value=True)
    use_web = st.checkbox("Web Search (Tavily)", value=False)
    use_api = st.checkbox("API Endpoints", value=False)

    # ── Model config ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Config")
    depth = st.selectbox("Analysis Depth", ["Auto", "Shallow", "Medium", "Deep"])
    enable_streaming = st.checkbox("Stream responses", value=True)

    # ── Clear chat ───────────────────────────────────────────────────────
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by NVIDIA NIM + LangChain")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("⚡ NeMo AgentIQ")
st.markdown("**Ask anything across your enterprise data sources.**")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("Reasoning Trace"):
                for step in msg["trace"]:
                    st.markdown(f"- {step}")
            if msg.get("tokens"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tokens Used", f"{msg['tokens']:,}")
                with col2:
                    st.metric("Est. Cost", f"${msg['tokens'] * 0.000001:.4f}")


def build_chat_history() -> str:
    """Format recent chat history for context."""
    history_window = st.session_state.messages[-10:]  # last 5 exchanges
    if not history_window:
        return "None"
    lines = []
    for msg in history_window:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long messages to save tokens
        content = msg["content"][:500]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── User input ───────────────────────────────────────────────────────────────

if query := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Build conversation memory
        chat_history = build_chat_history()

        # Progress tracking for agent pipeline
        with st.status("Thinking...", expanded=True) as status:

            # Step 1: Plan
            st.write("🧠 Planning query strategy...")
            orchestrator = OrchestratorAgent()
            sql_agent = SQLAgent()
            web_agent = WebAgent()
            api_agent = APIAgent()
            synthesis_agent = SynthesisAgent()

            graph = orchestrator.build_graph(
                doc_agent, sql_agent, web_agent, api_agent, synthesis_agent
            )

            # Step 2: Execute sub-agents (non-streaming part)
            st.write("🔍 Querying data sources...")
            # Build allowed sources from user toggles
            allowed_sources = []
            if use_docs:
                allowed_sources.append("docs")
            if use_sql:
                allowed_sources.append("sql")
            if use_web:
                allowed_sources.append("web")
            if use_api:
                allowed_sources.append("api")

            initial_state = {
                "query": query,
                "chat_history": chat_history,
                "plan": {},
                "sources": [],
                "allowed_sources": allowed_sources,
                "depth": depth.lower() if depth != "Auto" else "medium",
                "doc_results": "",
                "sql_results": "",
                "web_results": "",
                "api_results": "",
                "final_answer": "",
                "reasoning_trace": [],
                "token_usage": 0,
            }

            # Run the full graph to get sub-agent results
            result = graph.invoke(initial_state)

            status.update(label="Complete", state="complete")

        # Step 3: Display answer — stream if enabled
        has_results = any(result.get(k) for k in ("doc_results", "sql_results", "web_results", "api_results"))
        if enable_streaming and has_results:
            # Re-stream just the synthesis for a nice UX
            full_response = st.write_stream(
                synthesis_agent.stream(result)
            )
            if full_response:
                result["final_answer"] = full_response
        else:
            st.markdown(result["final_answer"])

        # Reasoning trace
        with st.expander("Reasoning Trace"):
            for step in result.get("reasoning_trace", []):
                st.markdown(f"- {step}")

        # Token usage
        tokens = result.get("token_usage", 0)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tokens Used", f"{tokens:,}")
        with col2:
            cost = tokens * 0.000001  # rough estimate
            st.metric("Est. Cost", f"${cost:.4f}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "trace": result.get("reasoning_trace", []),
        "tokens": tokens,
    })
