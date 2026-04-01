"""Tests for the orchestrator agent — planning and routing logic."""

from unittest.mock import patch, MagicMock


class TestOrchestratorPlanParsing:
    """Test that the orchestrator correctly parses LLM planning responses."""

    def _run_plan(self, llm_response_text, state):
        """Helper: patch the LLM and run plan_query."""
        mock_response = MagicMock()
        mock_response.content = llm_response_text

        with patch("agents.orchestrator.ChatNVIDIA") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            from agents.orchestrator import OrchestratorAgent

            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            agent.llm = MockLLM.return_value
            return agent.plan_query(state)

    def test_parses_multiple_sources(self, sample_state):
        text = "SOURCES: docs, web\nDEPTH: deep\nPLAN: Search docs and web"
        result = self._run_plan(text, sample_state)
        assert result["sources"] == ["docs", "web"]
        assert result["depth"] == "deep"

    def test_filters_to_allowed_sources(self, sample_state):
        sample_state["allowed_sources"] = ["docs"]
        text = "SOURCES: docs, sql, web\nDEPTH: medium\nPLAN: Check everything"
        result = self._run_plan(text, sample_state)
        assert result["sources"] == ["docs"]

    def test_empty_sources_fallback(self, sample_state):
        sample_state["allowed_sources"] = []
        text = "SOURCES: docs\nDEPTH: shallow\nPLAN: Quick lookup"
        result = self._run_plan(text, sample_state)
        # With no allowed_sources filter, all planned sources pass through
        assert result["sources"] == ["docs"]

    def test_reasoning_trace_populated(self, sample_state):
        text = "SOURCES: sql\nDEPTH: medium\nPLAN: Run SQL query"
        result = self._run_plan(text, sample_state)
        assert len(result["reasoning_trace"]) == 2
        assert "sql" in result["reasoning_trace"][1]


class TestOrchestratorRouting:
    """Test that route_after_plan returns correct node names."""

    def test_routes_to_doc_agent(self, sample_state):
        sample_state["sources"] = ["docs"]
        from agents.orchestrator import OrchestratorAgent

        with patch("agents.orchestrator.ChatNVIDIA"):
            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            assert agent.route(sample_state) == ["docs"]

    def test_routes_to_multiple_agents(self, sample_state):
        sample_state["sources"] = ["docs", "web", "sql"]
        from agents.orchestrator import OrchestratorAgent

        with patch("agents.orchestrator.ChatNVIDIA"):
            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            assert agent.route(sample_state) == ["docs", "web", "sql"]

    def test_routes_empty_sources(self, sample_state):
        sample_state["sources"] = []
        from agents.orchestrator import OrchestratorAgent

        with patch("agents.orchestrator.ChatNVIDIA"):
            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            assert agent.route(sample_state) == []
