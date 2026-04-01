"""Synthesis Agent — merges results from all sub-agents into a cited answer with streaming."""

import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

SYNTHESIS_PROMPT = """You are a synthesis agent. Merge the following results from multiple \
data sources into a single, well-structured answer to the user's query.

**Conversation History:**
{chat_history}

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
5. End with a confidence assessment (high/medium/low) based on source agreement
6. If this is a follow-up question, build on previous context naturally"""


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
                chat_history=state.get("chat_history", "None"),
                doc_results=state.get("doc_results", "No document results."),
                sql_results=state.get("sql_results", "No SQL results."),
                web_results=state.get("web_results", "No web results."),
                api_results=state.get("api_results", "No API results."),
            )
        )

        state["final_answer"] = response.content
        state["reasoning_trace"].append(
            "Step: Synthesis agent merged results with citations"
        )

        # Use actual token usage from model response when available
        usage = (
            getattr(response, "usage_metadata", None)
            or getattr(response, "response_metadata", {}).get("token_usage")
        )
        if usage and isinstance(usage, dict):
            state["token_usage"] = usage.get("total_tokens", 0) or (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
        elif usage and hasattr(usage, "total_tokens"):
            state["token_usage"] = usage.total_tokens
        else:
            # Fallback: estimate from word count (approx 1.3 tokens per word)
            total_words = sum(
                len(str(state.get(k, "")).split())
                for k in ["doc_results", "sql_results", "web_results", "api_results", "final_answer"]
            )
            state["token_usage"] = int(total_words * 1.3)

        logger.info("Synthesis complete, tokens=%s", state["token_usage"])
        return state

    def stream(self, state: dict):
        """Stream the synthesis response token by token."""
        prompt = SYNTHESIS_PROMPT.format(
            query=state["query"],
            chat_history=state.get("chat_history", "None"),
            doc_results=state.get("doc_results", "No document results."),
            sql_results=state.get("sql_results", "No SQL results."),
            web_results=state.get("web_results", "No web results."),
            api_results=state.get("api_results", "No API results."),
        )

        for chunk in self.llm.stream(prompt):
            yield chunk.content
