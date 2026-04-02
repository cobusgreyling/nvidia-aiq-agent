"""Retry utilities — exponential backoff for LLM and HTTP calls."""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Transient exceptions worth retrying
_TRANSIENT = (ConnectionError, TimeoutError, OSError)

try:
    import requests.exceptions
    _HTTP_TRANSIENT = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
except ImportError:
    _HTTP_TRANSIENT: tuple[type[Exception], ...] = ()


def llm_retry(func):
    """Retry decorator for LLM / NIM API calls (3 attempts, exp backoff)."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_TRANSIENT + _HTTP_TRANSIENT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


def http_retry(func):
    """Retry decorator for external HTTP calls (3 attempts, exp backoff)."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(_TRANSIENT + _HTTP_TRANSIENT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
