"""Tests for logging and observability — JSON formatter and latency tracking."""

import json
import logging
import time
from unittest.mock import patch

from log_config import JSONFormatter, ReadableFormatter, setup_logging, track_latency


class TestJSONFormatter:
    """Test structured JSON log output."""

    def test_formats_as_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="agentiq", level=logging.INFO, pathname="",
            lineno=0, msg="Test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="agentiq", level=logging.INFO, pathname="",
            lineno=0, msg="Agent done", args=(), exc_info=None,
        )
        record.agent = "doc_agent"
        record.duration_ms = 150
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["agent"] == "doc_agent"
        assert parsed["duration_ms"] == 150

    def test_includes_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="agentiq", level=logging.ERROR, pathname="",
                lineno=0, msg="Failed", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestReadableFormatter:
    """Test human-readable formatter."""

    def test_produces_readable_output(self):
        formatter = ReadableFormatter()
        record = logging.LogRecord(
            name="agentiq", level=logging.INFO, pathname="",
            lineno=0, msg="Hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "agentiq" in output
        assert "Hello" in output


class TestSetupLogging:
    """Test logger configuration."""

    def test_default_is_readable(self):
        # Clear any existing handlers
        logger = logging.getLogger("agentiq")
        logger.handlers.clear()

        with patch.dict("os.environ", {"LOG_FORMAT": "readable"}):
            result = setup_logging()
        assert result.name == "agentiq"
        assert len(result.handlers) == 1
        assert isinstance(result.handlers[0].formatter, ReadableFormatter)
        logger.handlers.clear()

    def test_json_format(self):
        logger = logging.getLogger("agentiq")
        logger.handlers.clear()

        with patch.dict("os.environ", {"LOG_FORMAT": "json"}):
            result = setup_logging()
        assert isinstance(result.handlers[0].formatter, JSONFormatter)
        logger.handlers.clear()


class TestTrackLatency:
    """Test latency context manager."""

    def test_records_latency(self):
        metrics = {}
        with track_latency("test_agent", metrics):
            time.sleep(0.01)
        assert "test_agent_ms" in metrics
        assert metrics["test_agent_ms"] >= 10

    def test_works_without_metrics_dict(self):
        # Should not raise even without metrics dict
        with track_latency("test_agent"):
            pass

    def test_records_on_exception(self):
        metrics = {}
        try:
            with track_latency("failing_agent", metrics):
                raise ValueError("boom")
        except ValueError:
            pass
        assert "failing_agent_ms" in metrics
