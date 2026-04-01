"""Web Agent — searches the web via Tavily and summarises results."""

import os
import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tavily import TavilyClient
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

        response = self.client.search(
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
        results = self._search(state["query"], max_results=max_results)

        formatted = "\n".join(
            f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results
        )

        response = self.llm.invoke(
            SUMMARISE_PROMPT.format(query=state["query"], results=formatted)
        )

        urls = [r["url"] for r in results if r["url"]]
        logger.info("Web agent found %d results", len(results))
        state["web_results"] = response.content
        state["reasoning_trace"].append(
            f"Step: Web agent searched Tavily, found {len(results)} results from: {', '.join(urls[:3])}"
        )
        return state
