"""SQL Agent — generates and executes SQL queries against configured databases."""

import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from agents.retry import llm_retry
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)


class SQLAgent:
    def __init__(self, db_uri: str = "sqlite:///data/sample.db"):
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.db = SQLDatabase.from_uri(db_uri)

    def run(self, state: dict) -> dict:
        """Generate SQL from natural language, execute, and return results."""
        chain = create_sql_query_chain(self.llm, self.db)

        try:
            _invoke = llm_retry(chain.invoke)
            sql_query = _invoke({"question": state["query"]})
        except Exception as e:
            logger.error("SQL agent query generation failed after retries: %s", e)
            state["sql_results"] = f"SQL query generation unavailable: {e}"
            state["reasoning_trace"].append(f"Step: SQL agent failed to generate query — {e}")
            return state

        try:
            result = self.db.run(sql_query)
            state["sql_results"] = f"Query: {sql_query}\nResult: {result}"
            state["reasoning_trace"].append(
                f"Step: SQL agent executed → {sql_query.strip()}"
            )
        except Exception as e:
            logger.error("SQL agent execution failed: %s", e)
            state["sql_results"] = f"SQL error: {e}"
            state["reasoning_trace"].append(
                f"Step: SQL agent failed → {e}"
            )

        return state
