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

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ NeMo AgentIQ")
    st.caption("Enterprise Agentic RAG with Multi-Source Reasoning")

    st.divider()
    st.subheader("Data Sources")
    use_docs = st.checkbox("Documents (RAG)", value=True)
    use_sql = st.checkbox("SQL Database", value=True)
    use_web = st.checkbox("Web Search", value=False)
    use_api = st.checkbox("API Endpoints", value=False)

    st.divider()
    st.subheader("Model Config")
    depth = st.selectbox("Analysis Depth", ["Auto", "Shallow", "Medium", "Deep"])

    st.divider()
    st.caption("Powered by NVIDIA NIM + LangChain")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("⚡ NeMo AgentIQ")
st.markdown("**Ask anything across your enterprise data sources.**")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Progress tracking
        with st.status("Thinking...", expanded=True) as status:

            # Step 1: Plan
            st.write("Planning query strategy...")
            orchestrator = OrchestratorAgent()
            doc_agent = DocAgent()
            sql_agent = SQLAgent()
            web_agent = WebAgent()
            api_agent = APIAgent()
            synthesis_agent = SynthesisAgent()

            graph = orchestrator.build_graph(
                doc_agent, sql_agent, web_agent, api_agent, synthesis_agent
            )

            # Step 2: Execute
            st.write("Executing agent pipeline...")
            initial_state = {
                "query": query,
                "plan": {},
                "sources": [],
                "depth": depth.lower() if depth != "Auto" else "medium",
                "doc_results": "",
                "sql_results": "",
                "web_results": "",
                "api_results": "",
                "final_answer": "",
                "reasoning_trace": [],
                "token_usage": 0,
            }

            result = graph.invoke(initial_state)

            status.update(label="Complete", state="complete")

        # Display answer
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

    st.session_state.messages.append(
        {"role": "assistant", "content": result["final_answer"]}
    )
