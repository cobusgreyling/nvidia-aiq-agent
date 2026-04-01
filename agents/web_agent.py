"""Web Agent — searches the web and summarises results."""

import yaml
import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

SUMMARISE_PROMPT = """Summarise the following web search results in relation to this query: {query}

Results:
{results}

Provide a concise summary with key facts. Cite the source URL for each fact."""


class WebAgent:
    def __init__(self, search_api_key: str = None):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.search_api_key = search_api_key

    def _search(self, query: str) -> list[dict]:
        """Perform a web search. Override this with your preferred search API."""
        # Placeholder — integrate with SerpAPI, Tavily, or Brave Search
        return [{"title": "No search API configured", "snippet": "Set search_api_key", "url": ""}]

    def run(self, state: dict) -> dict:
        """Search the web and summarise findings."""
        results = self._search(state["query"])

        formatted = "\n".join(
            f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results
        )

        response = self.llm.invoke(
            SUMMARISE_PROMPT.format(query=state["query"], results=formatted)
        )

        state["web_results"] = response.content
        state["reasoning_trace"].append(
            f"Step: Web agent searched and found {len(results)} results"
        )
        return state
