"""Centralised logging configuration for NeMo AgentIQ."""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include extra fields if set
        for key in ("agent", "duration_ms", "tokens", "endpoint", "source_count"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, default=str)


class ReadableFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def __init__(self):
        super().__init__(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger.

    Set LOG_FORMAT=json in environment for structured JSON output.
    """
    logger = logging.getLogger("agentiq")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        log_format = os.getenv("LOG_FORMAT", "readable").lower()
        if log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(ReadableFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


@contextmanager
def track_latency(agent_name: str, metrics: dict | None = None):
    """Context manager that tracks execution time for an agent.

    Usage:
        metrics = {}
        with track_latency("doc_agent", metrics):
            agent.run(state)
        # metrics["doc_agent_ms"] is now set
    """
    logger = logging.getLogger("agentiq")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "%s completed in %dms", agent_name, elapsed_ms,
            extra={"agent": agent_name, "duration_ms": elapsed_ms},
        )
        if metrics is not None:
            metrics[f"{agent_name}_ms"] = elapsed_ms
