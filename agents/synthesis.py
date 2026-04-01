"""Synthesis Agent — merges results from all sub-agents into a cited answer."""

import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

SYNTHESIS_PROMPT = """You are a synthesis agent. Merge the following results from multiple data sources into a single, well-structured answer to the user's query.

**User Query:** {query}

**Document Results:**
{doc_results}

**SQL Results:**
{sql_results}

**Web Results:**
{web_results}

**API Results:**
{api_results}

Instructions:
1. Synthesise all available information into a coherent answer
2. Cite the source for each claim: [docs], [sql], [web], [api]
3. If sources conflict, note the discrepancy
4. Be concise but thorough
5. End with a confidence assessment (high/medium/low) based on source agreement"""


class SynthesisAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["synthesis"]["model"],
            temperature=MODEL_CONFIG["synthesis"]["temperature"],
            max_tokens=MODEL_CONFIG["synthesis"]["max_tokens"],
        )

    def run(self, state: dict) -> dict:
        """Merge all sub-agent results and produce a cited final answer."""
        response = self.llm.invoke(
            SYNTHESIS_PROMPT.format(
                query=state["query"],
                doc_results=state.get("doc_results", "No document results."),
                sql_results=state.get("sql_results", "No SQL results."),
                web_results=state.get("web_results", "No web results."),
                api_results=state.get("api_results", "No API results."),
            )
        )

        state["final_answer"] = response.content
        state["reasoning_trace"].append(
            f"Step: Synthesis agent merged results with citations"
        )

        # Estimate token usage (rough)
        total_input = sum(
            len(str(state.get(k, "")).split())
            for k in ["doc_results", "sql_results", "web_results", "api_results"]
        )
        state["token_usage"] = total_input * 2  # rough estimate

        return state
