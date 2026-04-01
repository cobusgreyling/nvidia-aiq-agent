"""Orchestrator Agent — plans query strategy and routes to sub-agents."""

import copy
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.retry import llm_retry
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)


class AgentState(TypedDict):
    query: str
    chat_history: str
    plan: dict
    sources: list[str]
    allowed_sources: list[str]
    depth: str
    doc_results: str
    sql_results: str
    web_results: str
    api_results: str
    final_answer: str
    reasoning_trace: list[str]
    token_usage: int
    guardrail_violations: list[str]
    guardrail_output_flags: list[str]


PLANNING_PROMPT = """You are a query planning agent. Given a user query and conversation history, determine:

1. Which data sources to query: docs, sql, web, api (pick only relevant ones)
2. Analysis depth: shallow (quick lookup), medium (cross-reference), deep (comprehensive)
3. A brief plan of action

If the user is asking a follow-up question, use the conversation history to understand context.

Conversation history:
{chat_history}

Respond in this exact format:
SOURCES: <comma-separated list>
DEPTH: <shallow|medium|deep>
PLAN: <one line plan>

User query: {query}"""

# Map source names to their state result keys
SOURCE_RESULT_KEYS = {
    "docs": "doc_results",
    "sql": "sql_results",
    "web": "web_results",
    "api": "api_results",
}


class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["orchestrator"]["model"],
            temperature=MODEL_CONFIG["orchestrator"]["temperature"],
            max_tokens=MODEL_CONFIG["orchestrator"]["max_tokens"],
        )

    def plan_query(self, state: AgentState) -> AgentState:
        """Analyse the query and decide which sources to use."""
        _invoke = llm_retry(self.llm.invoke)
        response = _invoke(
            PLANNING_PROMPT.format(
                query=state["query"],
                chat_history=state.get("chat_history", "None"),
            )
        )
        text = response.content

        sources = []
        depth = "medium"
        plan = ""

        for line in text.strip().split("\n"):
            if line.startswith("SOURCES:"):
                sources = [s.strip().lower() for s in line.split(":", 1)[1].split(",")]
            elif line.startswith("DEPTH:"):
                depth = line.split(":", 1)[1].strip().lower()
            elif line.startswith("PLAN:"):
                plan = line.split(":", 1)[1].strip()

        # Filter to only user-allowed sources
        allowed = state.get("allowed_sources", [])
        if allowed:
            sources = [s for s in sources if s in allowed]
        state["sources"] = sources
        state["depth"] = depth
        state["plan"] = {"sources": sources, "depth": depth, "description": plan}
        state["reasoning_trace"] = [
            f"Step 1: Classified query — depth={depth}",
            f"Step 2: Selected sources → {', '.join(sources)}",
        ]
        logger.info("Query planned: depth=%s, sources=%s", depth, sources)
        return state

    def route(self, state: AgentState) -> list[str]:
        """Route to the appropriate sub-agents based on planning."""
        return state["sources"]

    def build_graph(self, doc_agent, sql_agent, web_agent, api_agent,
                    synthesis_agent, guardrails_agent=None):
        """Construct the LangGraph state machine with parallel agent execution."""
        agent_map = {
            "docs": doc_agent,
            "sql": sql_agent,
            "web": web_agent,
            "api": api_agent,
        }

        def execute_parallel(state: AgentState) -> AgentState:
            """Run all selected sub-agents concurrently."""
            sources = state.get("sources", [])
            if not sources:
                state["reasoning_trace"].append(
                    "Step: No sources selected — skipping agent execution"
                )
                return state

            futures = {}
            with ThreadPoolExecutor(max_workers=len(sources)) as executor:
                for source in sources:
                    if source in agent_map:
                        # Deep-copy state so agents don't interfere with each other
                        agent_state = copy.deepcopy(state)
                        futures[executor.submit(
                            agent_map[source].run, agent_state
                        )] = source

            traces = list(state.get("reasoning_trace", []))
            failed = []
            for future in as_completed(futures):
                source = futures[future]
                result_key = SOURCE_RESULT_KEYS[source]
                try:
                    result = future.result()
                    state[result_key] = result.get(result_key, "")
                    # Collect reasoning traces from the sub-agent
                    for t in result.get("reasoning_trace", []):
                        if t not in traces:
                            traces.append(t)
                except Exception as e:
                    logger.error("%s agent failed: %s", source, e)
                    state[result_key] = f"{source} agent unavailable: {e}"
                    traces.append(f"Step: {source} agent failed — {e}")
                    failed.append(source)

            if failed:
                remaining = [s for s in sources if s not in failed]
                traces.append(
                    f"Step: Graceful degradation — {', '.join(failed)} failed, "
                    f"continuing with {', '.join(remaining) or 'synthesis only'}"
                )

            state["reasoning_trace"] = traces
            return state

        graph = StateGraph(AgentState)

        if guardrails_agent:
            graph.add_node("input_guardrails", guardrails_agent.check_input)

        graph.add_node("planner", self.plan_query)
        graph.add_node("execute_agents", execute_parallel)
        graph.add_node("synthesis", synthesis_agent.run)

        if guardrails_agent:
            graph.add_node("output_guardrails", guardrails_agent.check_output)

        # Entry point: guardrails first if available
        if guardrails_agent:
            graph.set_entry_point("input_guardrails")

            def route_after_guardrails(state: AgentState) -> str:
                violations = state.get("guardrail_violations", [])
                if violations:
                    return "synthesis"
                return "planner"

            graph.add_conditional_edges("input_guardrails", route_after_guardrails)
        else:
            graph.set_entry_point("planner")

        graph.add_edge("planner", "execute_agents")
        graph.add_edge("execute_agents", "synthesis")

        if guardrails_agent:
            graph.add_edge("synthesis", "output_guardrails")
            graph.add_edge("output_guardrails", END)
        else:
            graph.add_edge("synthesis", END)

        return graph.compile()
