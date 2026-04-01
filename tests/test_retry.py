"""Tests for retry utilities and graceful degradation."""

from unittest.mock import patch, MagicMock


class TestRetryDecorators:
    """Test that retry decorators are importable and wrap functions correctly."""

    def test_llm_retry_wraps_function(self):
        """llm_retry should return a callable that behaves like the original."""
        from agents.retry import llm_retry

        call_count = 0

        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        wrapped = llm_retry(my_func)
        assert wrapped(5) == 10
        assert call_count == 1

    def test_http_retry_wraps_function(self):
        from agents.retry import http_retry

        def my_func(url):
            return f"response from {url}"

        wrapped = http_retry(my_func)
        assert wrapped("https://example.com") == "response from https://example.com"


class TestDocAgentGracefulDegradation:
    """Test that doc agent handles failures gracefully."""

    def test_run_handles_retrieval_failure(self, sample_state):
        with patch("agents.doc_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.doc_agent.NVIDIAEmbeddings"), \
             patch("agents.doc_agent.RetrievalQA") as MockChain:

            # Make chain.invoke raise
            mock_chain_instance = MagicMock()
            mock_chain_instance.invoke.side_effect = ConnectionError("NIM API down")
            MockChain.from_chain_type.return_value = mock_chain_instance

            from agents.doc_agent import DocAgent

            agent = DocAgent.__new__(DocAgent)
            agent.llm = MockLLM.return_value
            agent.embeddings = MagicMock()
            agent.vectorstore = MagicMock()

            result = agent.run(sample_state)
            assert "unavailable" in result["doc_results"]
            assert any("failed" in t for t in result["reasoning_trace"])


class TestWebAgentGracefulDegradation:
    """Test that web agent degrades gracefully."""

    def test_search_failure_returns_error(self, sample_state):
        with patch("agents.web_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.web_agent.TavilyClient"):

            from agents.web_agent import WebAgent

            agent = WebAgent.__new__(WebAgent)
            agent.llm = MockLLM.return_value
            agent.api_key = "test-key"
            agent.client = MagicMock()
            agent.client.search.side_effect = ConnectionError("Tavily down")

            result = agent.run(sample_state)
            assert "unavailable" in result["web_results"]

    def test_summarisation_failure_returns_raw_results(self, sample_state):
        with patch("agents.web_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.web_agent.TavilyClient"):

            from agents.web_agent import WebAgent

            agent = WebAgent.__new__(WebAgent)
            agent.llm = MockLLM.return_value
            agent.api_key = "test-key"

            # Search succeeds
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [{"title": "R1", "content": "Content 1", "url": "https://a.com"}]
            }
            agent.client = mock_client

            # LLM summarisation fails
            agent.llm.invoke.side_effect = ConnectionError("NIM API down")

            result = agent.run(sample_state)
            assert "raw" in result["web_results"].lower()
            assert "R1" in result["web_results"]


class TestAPIAgentGracefulDegradation:
    """Test that API agent degrades gracefully."""

    def test_endpoint_selection_failure(self, sample_state):
        with patch("agents.api_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.api_agent.requests"):

            from agents.api_agent import APIAgent

            agent = APIAgent.__new__(APIAgent)
            agent.llm = MockLLM.return_value
            agent.endpoints = {"weather": {"url": "https://wttr.in", "description": "Weather"}}

            # LLM fails during endpoint selection
            agent.llm.invoke.side_effect = ConnectionError("NIM down")

            result = agent.run(sample_state)
            assert "unavailable" in result["api_results"].lower()

    def test_api_call_failure_continues(self, sample_state):
        with patch("agents.api_agent.ChatNVIDIA") as MockLLM, \
             patch("agents.api_agent.requests") as mock_requests:

            mock_response = MagicMock()
            mock_response.content = "Summary of results."
            MockLLM.return_value.invoke.return_value = mock_response

            from agents.api_agent import APIAgent

            agent = APIAgent.__new__(APIAgent)
            agent.llm = MockLLM.return_value
            agent.endpoints = {
                "weather": {"url": "https://wttr.in/{query}?format=j1", "method": "GET",
                            "description": "Weather", "query_param": "path"},
            }

            # Simulate _select_endpoints returning a call
            agent._select_endpoints = MagicMock(return_value=[("weather", "London")])

            # HTTP call fails
            mock_requests.get.side_effect = ConnectionError("timeout")
            mock_requests.utils.quote.side_effect = lambda x: x

            result = agent.run(sample_state)
            # Should still produce results (with error noted)
            assert "reasoning_trace" in result
            assert any("weather" in t for t in result["reasoning_trace"])
