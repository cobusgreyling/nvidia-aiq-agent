"""Orchestrator Agent — plans query strategy and routes to sub-agents."""

import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, END
from typing import TypedDict

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)


class AgentState(TypedDict):
    query: str
    chat_history: str
    plan: dict
    sources: list[str]
    depth: str
    doc_results: str
    sql_results: str
    web_results: str
    api_results: str
    final_answer: str
    reasoning_trace: list[str]
    token_usage: int


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


class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["orchestrator"]["model"],
            temperature=MODEL_CONFIG["orchestrator"]["temperature"],
            max_tokens=MODEL_CONFIG["orchestrator"]["max_tokens"],
        )

    def plan_query(self, state: AgentState) -> AgentState:
        """Analyse the query and decide which sources to use."""
        response = self.llm.invoke(
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

        state["sources"] = sources
        state["depth"] = depth
        state["plan"] = {"sources": sources, "depth": depth, "description": plan}
        state["reasoning_trace"] = [
            f"Step 1: Classified query — depth={depth}",
            f"Step 2: Selected sources → {', '.join(sources)}",
        ]
        return state

    def route(self, state: AgentState) -> list[str]:
        """Route to the appropriate sub-agents based on planning."""
        return state["sources"]

    def build_graph(self, doc_agent, sql_agent, web_agent, api_agent, synthesis_agent):
        """Construct the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("planner", self.plan_query)
        graph.add_node("doc_agent", doc_agent.run)
        graph.add_node("sql_agent", sql_agent.run)
        graph.add_node("web_agent", web_agent.run)
        graph.add_node("api_agent", api_agent.run)
        graph.add_node("synthesis", synthesis_agent.run)

        graph.set_entry_point("planner")

        def route_after_plan(state: AgentState) -> list[str]:
            sources = state.get("sources", [])
            next_nodes = []
            if "docs" in sources:
                next_nodes.append("doc_agent")
            if "sql" in sources:
                next_nodes.append("sql_agent")
            if "web" in sources:
                next_nodes.append("web_agent")
            if "api" in sources:
                next_nodes.append("api_agent")
            return next_nodes if next_nodes else ["synthesis"]

        graph.add_conditional_edges("planner", route_after_plan)

        for node in ["doc_agent", "sql_agent", "web_agent", "api_agent"]:
            graph.add_edge(node, "synthesis")

        graph.add_edge("synthesis", END)

        return graph.compile()
