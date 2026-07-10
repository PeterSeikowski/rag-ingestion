"""Application-layer exception hierarchy.

Distinguishes retryable infrastructure hiccups from permanent, non-retryable
failures so Celery task retry policy (workers/tasks.py) can branch on this
hierarchy rather than on library-specific exception types — that keeps
Celery-specific retry configuration out of application/ and adapters/.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all application-layer ingestion failures."""


class TransientIngestionError(IngestionError):
    """A failure likely to succeed on retry: a dependency (Redis,
    Elasticsearch, the embedding provider) was unreachable or timed out.
    """


class PermanentIngestionError(IngestionError):
    """A failure that will not succeed on retry: malformed input, a PDF
    the parser cannot handle, a validation error. Retrying would waste
    work and delay surfacing the real problem to the caller.
    """


class ConfigurationError(IngestionError):
    """Required configuration is missing or invalid (e.g. no
    LITELLM_API_KEY, unresolvable embedding dimensions). Always permanent —
    fail fast at startup or first use, never retry.
    """
