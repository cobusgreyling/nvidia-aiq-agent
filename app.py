"""NeMo AgentIQ — Streamlit GUI for Enterprise Agentic RAG."""

import datetime
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import OrchestratorAgent  # noqa: E402
from agents.doc_agent import DocAgent  # noqa: E402
from agents.sql_agent import SQLAgent  # noqa: E402
from agents.web_agent import WebAgent  # noqa: E402
from agents.api_agent import APIAgent  # noqa: E402
from agents.synthesis import SynthesisAgent  # noqa: E402
from agents.guardrails import GuardrailsAgent  # noqa: E402
import chat_store  # noqa: E402

st.set_page_config(page_title="NeMo AgentIQ", page_icon="⚡", layout="wide")

# ── Session state init ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_agent" not in st.session_state:
    st.session_state.doc_agent = DocAgent()
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

doc_agent = st.session_state.doc_agent


def _export_chat_markdown() -> str:
    """Format the conversation history as a Markdown document."""
    lines = [
        "# NeMo AgentIQ — Chat Export",
        f"*Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "---\n",
    ]
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"## {role}\n")
        lines.append(msg["content"] + "\n")
        if msg.get("trace"):
            lines.append("<details><summary>Reasoning Trace</summary>\n")
            for step in msg["trace"]:
                lines.append(f"- {step}")
            lines.append("\n</details>\n")
        if msg.get("tokens"):
            lines.append(f"*Tokens: {msg['tokens']:,} | Est. cost: ${msg['tokens'] * 0.000001:.4f}*\n")
        lines.append("---\n")
    return "\n".join(lines)


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

    if st.button("Re-index Documents"):
        with st.spinner("Re-indexing from data/documents/..."):
            from ingest import ingest as run_ingest
            run_ingest()
            st.session_state.doc_agent = DocAgent()
            doc_agent = st.session_state.doc_agent
            doc_agent.load_vectorstore()
        st.success("Documents re-indexed from disk")

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

    # ── Export / Clear ────────────────────────────────────────────────────
    st.divider()
    col_export, col_clear = st.columns(2)
    with col_export:
        if st.session_state.messages:
            st.download_button(
                "Export Chat",
                data=_export_chat_markdown(),
                file_name=f"agentiq_chat_{datetime.date.today()}.md",
                mime="text/markdown",
            )
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    # ── Saved Conversations ───────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Conversations")
    if st.button("New Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    conversations = chat_store.list_conversations()
    for conv in conversations[:20]:
        col_load, col_del = st.columns([4, 1])
        with col_load:
            label = f"{conv['title']} ({conv['message_count']} msgs)"
            if st.button(label, key=f"load_{conv['id']}"):
                st.session_state.conversation_id = conv["id"]
                st.session_state.messages = chat_store.load_messages(conv["id"])
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{conv['id']}"):
                chat_store.delete_conversation(conv["id"])
                if st.session_state.conversation_id == conv["id"]:
                    st.session_state.messages = []
                    st.session_state.conversation_id = None
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
            st.write("🛡️ Checking input guardrails...")
            orchestrator = OrchestratorAgent()
            sql_agent = SQLAgent()
            web_agent = WebAgent()
            api_agent = APIAgent()
            synthesis_agent = SynthesisAgent()
            guardrails_agent = GuardrailsAgent()

            graph = orchestrator.build_graph(
                doc_agent, sql_agent, web_agent, api_agent,
                synthesis_agent, guardrails_agent=guardrails_agent,
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
                "guardrail_violations": [],
                "guardrail_output_flags": [],
            }

            # Run the full graph to get sub-agent results
            pipeline_start = time.perf_counter()
            result = graph.invoke(initial_state)
            pipeline_ms = round((time.perf_counter() - pipeline_start) * 1000)

            status.update(label=f"Complete ({pipeline_ms}ms)", state="complete")

        # Check if input guardrails blocked the query
        violations = result.get("guardrail_violations", [])
        if violations:
            st.warning(
                "This query was flagged by safety guardrails: "
                + ", ".join(v.replace("_", " ") for v in violations)
            )
            result["final_answer"] = (
                "I'm unable to process this query as it was flagged by our safety guardrails. "
                "Please rephrase your question."
            )
            st.markdown(result["final_answer"])
        else:
            # Step 3: Display answer — stream if enabled
            has_results = any(
                result.get(k) for k in ("doc_results", "sql_results", "web_results", "api_results")
            )
            if enable_streaming and has_results:
                # Re-stream just the synthesis for a nice UX
                full_response = st.write_stream(
                    synthesis_agent.stream(result)
                )
                if full_response:
                    result["final_answer"] = full_response
            else:
                st.markdown(result["final_answer"])

        # Show output guardrail flags if any
        output_flags = result.get("guardrail_output_flags", [])
        if output_flags:
            flag_labels = [f.replace("_", " ").title() for f in output_flags]
            st.info(f"🛡️ Output guardrails applied: {', '.join(flag_labels)}")

        # Reasoning trace
        with st.expander("Reasoning Trace"):
            for step in result.get("reasoning_trace", []):
                st.markdown(f"- {step}")

        # Metrics panel
        tokens = result.get("token_usage", 0)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tokens Used", f"{tokens:,}")
        with col2:
            cost = tokens * 0.000001  # rough estimate
            st.metric("Est. Cost", f"${cost:.4f}")
        with col3:
            st.metric("Pipeline Latency", f"{pipeline_ms:,}ms")

    # Persist messages to SQLite
    if st.session_state.conversation_id is None:
        title = query[:60] + ("..." if len(query) > 60 else "")
        st.session_state.conversation_id = chat_store.create_conversation(title)

    chat_store.save_message(st.session_state.conversation_id, "user", query)
    chat_store.save_message(
        st.session_state.conversation_id, "assistant",
        result["final_answer"],
        trace=result.get("reasoning_trace", []),
        tokens=tokens,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "trace": result.get("reasoning_trace", []),
        "tokens": tokens,
    })
