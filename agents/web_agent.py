"""Web Agent — searches the web via Tavily and summarises results."""

import os
import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tavily import TavilyClient
from agents.retry import llm_retry, http_retry
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

SUMMARISE_PROMPT = """Summarise the following web search results in relation to this query: {query}

Results:
{results}

Provide a concise summary with key facts. Cite the source URL for each fact."""


class WebAgent:
    def __init__(self, search_api_key: str | None = None):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.api_key = search_api_key or os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    def _search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web using Tavily API."""
        if not self.client:
            return [{"title": "Tavily not configured", "snippet": "Set TAVILY_API_KEY in .env", "url": ""}]

        _search = http_retry(self.client.search)
        response = _search(
            query=query,
            max_results=max_results,
            include_answer=False,
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
            })
        return results

    def run(self, state: dict) -> dict:
        """Search the web and summarise findings."""
        max_results = 8 if state.get("depth") == "deep" else 5

        try:
            results = self._search(state["query"], max_results=max_results)
        except Exception as e:
            logger.error("Web agent search failed after retries: %s", e)
            state["web_results"] = f"Web search unavailable: {e}"
            state["reasoning_trace"].append(f"Step: Web agent search failed — {e}")
            return state

        formatted = "\n".join(
            f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results
        )

        try:
            _invoke = llm_retry(self.llm.invoke)
            response = _invoke(
                SUMMARISE_PROMPT.format(query=state["query"], results=formatted)
            )
        except Exception as e:
            logger.error("Web agent summarisation failed after retries: %s", e)
            # Graceful degradation: return raw results instead of summary
            state["web_results"] = f"Web results (raw — summarisation unavailable):\n{formatted}"
            state["reasoning_trace"].append(
                f"Step: Web agent found {len(results)} results but summarisation failed — returning raw"
            )
            return state

        urls = [r["url"] for r in results if r["url"]]
        logger.info("Web agent found %d results", len(results))
        state["web_results"] = response.content
        state["reasoning_trace"].append(
            f"Step: Web agent searched Tavily, found {len(results)} results from: {', '.join(urls[:3])}"
        )
        return state
