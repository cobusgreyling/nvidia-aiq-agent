"""API Agent — calls configured REST/GraphQL endpoints."""

import yaml
import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

EXTRACT_PROMPT = """Given this API response for the query "{query}", extract the relevant information and summarise it clearly.

API Response:
{response}

Summarise the key data points relevant to the query."""


class APIAgent:
    def __init__(self, endpoints: dict = None):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.endpoints = endpoints or {}

    def _call_endpoint(self, name: str, params: dict) -> dict:
        """Call a configured API endpoint."""
        endpoint = self.endpoints.get(name, {})
        url = endpoint.get("url", "")
        method = endpoint.get("method", "GET").upper()

        if not url:
            return {"error": f"Endpoint '{name}' not configured"}

        if method == "GET":
            resp = requests.get(url, params=params, timeout=30)
        else:
            resp = requests.post(url, json=params, timeout=30)

        resp.raise_for_status()
        return resp.json()

    def run(self, state: dict) -> dict:
        """Call APIs and summarise results."""
        if not self.endpoints:
            state["api_results"] = "No API endpoints configured."
            state["reasoning_trace"].append("Step: API agent — no endpoints configured")
            return state

        all_results = []
        for name in self.endpoints:
            try:
                result = self._call_endpoint(name, {"query": state["query"]})
                all_results.append(f"{name}: {result}")
            except Exception as e:
                all_results.append(f"{name}: error — {e}")

        combined = "\n".join(all_results)
        response = self.llm.invoke(
            EXTRACT_PROMPT.format(query=state["query"], response=combined)
        )

        state["api_results"] = response.content
        state["reasoning_trace"].append(
            f"Step: API agent called {len(self.endpoints)} endpoints"
        )
        return state
