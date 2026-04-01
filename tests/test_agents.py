"""Tests for sub-agents — doc, sql, web, api, synthesis."""

from unittest.mock import patch, MagicMock


class TestDocAgent:
    """Test document agent behavior."""

    def test_run_without_vectorstore(self, sample_state):
        with patch("agents.doc_agent.ChatNVIDIA"), \
             patch("agents.doc_agent.NVIDIAEmbeddings"):
            from agents.doc_agent import DocAgent

            agent = DocAgent.__new__(DocAgent)
            agent.vectorstore = None
            agent.embeddings = MagicMock()

            # Patch load_vectorstore to return False (no store on disk)
            agent.load_vectorstore = MagicMock(return_value=False)

            result = agent.run(sample_state)
            assert "No documents ingested" in result["doc_results"]
            assert any("no vector store" in s for s in result["reasoning_trace"])

    def test_run_with_vectorstore(self, sample_state):
        with patch("agents.doc_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.doc_agent.NVIDIAEmbeddings"), \
             patch("agents.doc_agent.RetrievalQA") as MockChain:

            mock_chain_instance = MagicMock()
            mock_chain_instance.invoke.return_value = {
                "result": "NeMo AgentIQ is an enterprise RAG system.",
                "source_documents": [
                    MagicMock(metadata={"source": "doc.pdf", "page": 1}),
                ],
            }
            MockChain.from_chain_type.return_value = mock_chain_instance

            from agents.doc_agent import DocAgent

            agent = DocAgent.__new__(DocAgent)
            agent.llm = MockLLM.return_value
            agent.embeddings = MagicMock()

            # Simulate a loaded vectorstore
            mock_vs = MagicMock()
            mock_vs.as_retriever.return_value = MagicMock()
            agent.vectorstore = mock_vs

            result = agent.run(sample_state)
            assert "enterprise RAG" in result["doc_results"]

    def test_deep_depth_uses_more_results(self, sample_state):
        sample_state["depth"] = "deep"
        with patch("agents.doc_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.doc_agent.NVIDIAEmbeddings"), \
             patch("agents.doc_agent.RetrievalQA") as MockChain:

            mock_chain_instance = MagicMock()
            mock_chain_instance.invoke.return_value = {
                "result": "answer",
                "source_documents": [],
            }
            MockChain.from_chain_type.return_value = mock_chain_instance

            from agents.doc_agent import DocAgent

            agent = DocAgent.__new__(DocAgent)
            agent.llm = MockLLM.return_value
            agent.embeddings = MagicMock()
            mock_vs = MagicMock()
            agent.vectorstore = mock_vs

            agent.run(sample_state)
            # Verify retriever was called with k=5 for deep depth
            mock_vs.as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5},
            )


class TestWebAgent:
    """Test web agent behavior."""

    def test_run_without_api_key(self, sample_state):
        with patch("agents.web_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.web_agent.TavilyClient"):

            mock_response = MagicMock()
            mock_response.content = "Summary of web results."
            MockLLM.return_value.invoke.return_value = mock_response

            from agents.web_agent import WebAgent

            agent = WebAgent.__new__(WebAgent)
            agent.llm = MockLLM.return_value
            agent.api_key = None
            agent.client = None

            result = agent.run(sample_state)
            assert result["web_results"] == "Summary of web results."

    def test_search_returns_structured_results(self):
        with patch("agents.web_agent.ChatNVIDIA"), \
             patch("agents.web_agent.TavilyClient"):

            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [
                    {"title": "Result 1", "content": "Content 1", "url": "https://example.com/1"},
                    {"title": "Result 2", "content": "Content 2", "url": "https://example.com/2"},
                ]
            }

            from agents.web_agent import WebAgent

            agent = WebAgent.__new__(WebAgent)
            agent.llm = MagicMock()
            agent.api_key = "test-key"
            agent.client = mock_client

            results = agent._search("test query", max_results=5)
            assert len(results) == 2
            assert results[0]["title"] == "Result 1"


class TestAPIAgent:
    """Test API agent behavior."""

    def test_run_without_endpoints(self, sample_state):
        with patch("agents.api_agent.ChatNVIDIA"):
            from agents.api_agent import APIAgent

            agent = APIAgent.__new__(APIAgent)
            agent.llm = MagicMock()
            agent.endpoints = {}

            result = agent.run(sample_state)
            assert "No API endpoints configured" in result["api_results"]

    def test_call_endpoint_get(self):
        with patch("agents.api_agent.ChatNVIDIA"), \
             patch("agents.api_agent.requests") as mock_requests:

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": "test"}
            mock_requests.get.return_value = mock_resp
            mock_requests.utils.quote.side_effect = lambda x: x

            from agents.api_agent import APIAgent

            agent = APIAgent.__new__(APIAgent)
            agent.endpoints = {
                "test": {"url": "https://api.example.com/data", "method": "GET", "query_param": "query"},
            }

            result = agent._call_endpoint("test", "hello")
            assert '"data": "test"' in result
            mock_requests.get.assert_called_once()


class TestSQLAgent:
    """Test SQL agent behavior."""

    def test_run_handles_sql_error(self, sample_state):
        with patch("agents.sql_agent.ChatNVIDIA"), \
             patch("agents.sql_agent.SQLDatabase"), \
             patch("agents.sql_agent.create_sql_query_chain") as MockChain:

            MockChain.return_value.invoke.return_value = "SELECT * FROM users"
            mock_db = MagicMock()
            mock_db.run.side_effect = Exception("table not found")

            from agents.sql_agent import SQLAgent

            agent = SQLAgent.__new__(SQLAgent)
            agent.llm = MagicMock()
            agent.db = mock_db

            result = agent.run(sample_state)
            assert "SQL error" in result["sql_results"]
            assert any("failed" in s for s in result["reasoning_trace"])


class TestSynthesisAgent:
    """Test synthesis agent behavior."""

    def test_run_produces_final_answer(self, sample_state):
        sample_state["doc_results"] = "NeMo is a RAG system."
        sample_state["web_results"] = "NVIDIA builds AI tools."

        with patch("agents.synthesis.ChatNVIDIA") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = "Synthesized answer with [docs] and [web] citations."
            mock_response.usage_metadata = None
            mock_response.response_metadata = {}
            MockLLM.return_value.invoke.return_value = mock_response

            from agents.synthesis import SynthesisAgent

            agent = SynthesisAgent.__new__(SynthesisAgent)
            agent.llm = MockLLM.return_value

            result = agent.run(sample_state)
            assert "Synthesized answer" in result["final_answer"]
            assert result["token_usage"] > 0

    def test_stream_yields_chunks(self, sample_state):
        sample_state["doc_results"] = "Some doc results."

        with patch("agents.synthesis.ChatNVIDIA") as MockLLM:
            chunk1 = MagicMock()
            chunk1.content = "Hello "
            chunk2 = MagicMock()
            chunk2.content = "world"
            MockLLM.return_value.stream.return_value = [chunk1, chunk2]

            from agents.synthesis import SynthesisAgent

            agent = SynthesisAgent.__new__(SynthesisAgent)
            agent.llm = MockLLM.return_value

            chunks = list(agent.stream(sample_state))
            assert chunks == ["Hello ", "world"]

    def test_token_usage_from_metadata(self, sample_state):
        with patch("agents.synthesis.ChatNVIDIA") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = "Answer."
            mock_response.usage_metadata = {"total_tokens": 1234}
            mock_response.response_metadata = {}
            MockLLM.return_value.invoke.return_value = mock_response

            from agents.synthesis import SynthesisAgent

            agent = SynthesisAgent.__new__(SynthesisAgent)
            agent.llm = MockLLM.return_value

            result = agent.run(sample_state)
            assert result["token_usage"] == 1234
