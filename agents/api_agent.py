"""API Agent — calls configured REST/GraphQL endpoints."""

import json
from urllib.parse import quote as url_quote
import yaml
import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from agents.retry import llm_retry, http_retry
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

SELECT_PROMPT = """You are an API routing agent. Given a user query and a list of available API endpoints, \
decide which endpoints are relevant and extract the query parameter for each.

Available endpoints:
{endpoints}

User query: {query}

Respond in this exact format (one line per relevant endpoint, skip irrelevant ones):
CALL: <endpoint_name> | PARAM: <extracted_parameter>

If no endpoints are relevant, respond with: NONE"""

EXTRACT_PROMPT = (
    "Given these API responses for the query \"{query}\", "
    "extract the relevant information and summarise it clearly.\n\n"
    "API Responses:\n{response}\n\n"
    "Summarise the key data points relevant to the query. "
    "Be specific with numbers and facts."
)


class APIAgent:
    def __init__(self, endpoints: dict | None = None):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.endpoints = endpoints or MODEL_CONFIG.get("api_endpoints", {})

    def _build_url(self, endpoint: dict, param: str) -> str:
        """Build the final URL, substituting path parameters if needed."""
        url = endpoint.get("url", "")
        query_param = endpoint.get("query_param", "query")
        if query_param == "path" and "{query}" in url:
            url = url.replace("{query}", url_quote(param))
        return url

    def _call_endpoint(self, name: str, param: str) -> str:
        """Call a configured API endpoint and return a text summary."""
        endpoint = self.endpoints.get(name, {})
        url = self._build_url(endpoint, param)
        method = endpoint.get("method", "GET").upper()

        if not url:
            return f"Endpoint '{name}' not configured"

        @http_retry
        def _do_request():
            if method == "GET":
                query_param = endpoint.get("query_param", "query")
                params = {} if query_param in ("path", "none") else {"query": param}
                return requests.get(url, params=params, timeout=30)
            else:
                return requests.post(url, json={"query": param}, timeout=30)

        resp = _do_request()
        resp.raise_for_status()

        # Handle both JSON and plain text responses
        try:
            data = resp.json()
            # Truncate large responses to avoid token overflow
            text = json.dumps(data, indent=2, default=str)
            if len(text) > 3000:
                text = text[:3000] + "\n... (truncated)"
            return text
        except (ValueError, json.JSONDecodeError):
            text = resp.text[:2000]
            return text

    def _select_endpoints(self, query: str) -> list[tuple[str, str]]:
        """Use the LLM to pick relevant endpoints and extract parameters."""
        endpoint_descriptions = "\n".join(
            f"- {name}: {ep.get('description', 'No description')}"
            for name, ep in self.endpoints.items()
        )

        _invoke = llm_retry(self.llm.invoke)
        response = _invoke(
            SELECT_PROMPT.format(endpoints=endpoint_descriptions, query=query)
        )

        calls = []
        for line in response.content.strip().split("\n"):
            if line.strip() == "NONE":
                break
            if line.startswith("CALL:"):
                parts = line.split("|")
                if len(parts) == 2:
                    name = parts[0].replace("CALL:", "").strip()
                    param = parts[1].replace("PARAM:", "").strip()
                    if name in self.endpoints:
                        calls.append((name, param))
        return calls

    def run(self, state: dict) -> dict:
        """Select relevant APIs, call them, and summarise results."""
        if not self.endpoints:
            state["api_results"] = "No API endpoints configured."
            state["reasoning_trace"].append("Step: API agent — no endpoints configured")
            return state

        # Ask LLM which endpoints to call
        try:
            calls = self._select_endpoints(state["query"])
        except Exception as e:
            logger.error("API agent endpoint selection failed after retries: %s", e)
            state["api_results"] = f"API agent unavailable: {e}"
            state["reasoning_trace"].append(f"Step: API agent failed during endpoint selection — {e}")
            return state

        if not calls:
            state["api_results"] = "No relevant API endpoints for this query."
            state["reasoning_trace"].append("Step: API agent — no relevant endpoints found")
            return state

        all_results = []
        called_names = []
        for name, param in calls:
            try:
                result = self._call_endpoint(name, param)
                desc = self.endpoints[name].get("description", name)
                all_results.append(f"[{desc}]\n{result}")
                called_names.append(name)
            except Exception as e:
                logger.error("API agent call to %s failed after retries: %s", name, e)
                all_results.append(f"{name}: error — {e}")
                called_names.append(f"{name}(failed)")

        combined = "\n\n".join(all_results)
        try:
            _invoke = llm_retry(self.llm.invoke)
            response = _invoke(
                EXTRACT_PROMPT.format(query=state["query"], response=combined)
            )
            state["api_results"] = response.content
        except Exception as e:
            logger.error("API agent summarisation failed after retries: %s", e)
            # Graceful degradation: return raw API results
            state["api_results"] = f"API results (raw — summarisation unavailable):\n{combined}"

        state["reasoning_trace"].append(
            f"Step: API agent called {len(called_names)} endpoint(s): {', '.join(called_names)}"
        )
        return state
