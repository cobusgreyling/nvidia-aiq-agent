"""Shared fixtures for the test suite."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure env vars are set so modules can import without errors
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

# Stub out heavy third-party modules that may not be installed in CI/dev.
# This must happen before any agent module is imported.
_STUB_MODULES = [
    "langchain_nvidia_ai_endpoints",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.document_loaders",
    "langchain_community.utilities",
    "langchain",
    "langchain.chains",
    "langchain.text_splitter",
    "langgraph",
    "langgraph.graph",
    "tavily",
    "nemoguardrails",
    "streamlit",
    "pypdf",
    "requests.utils",
]

for mod_name in _STUB_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Provide a fake END sentinel for langgraph
sys.modules["langgraph.graph"].END = "END"

# Provide a real TypedDict so AgentState works
sys.modules["langgraph.graph"].StateGraph = MagicMock()


@pytest.fixture
def sample_state():
    """Return a minimal AgentState dict for testing."""
    return {
        "query": "What is NeMo AgentIQ?",
        "chat_history": "None",
        "plan": {},
        "sources": [],
        "allowed_sources": ["docs", "sql", "web", "api"],
        "depth": "medium",
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
