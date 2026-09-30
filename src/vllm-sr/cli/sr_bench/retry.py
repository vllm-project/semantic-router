"""Bounded retries for transient model-call failures.

Opt-in through the optional ``max_call_attempts`` and ``retry_backoff_s``
limits; without them every call has exactly one attempt.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

LIMIT_KEYS = ("max_call_attempts", "retry_backoff_s")
MAX_ATTEMPTS = 8
MAX_BACKOFF_S = 60
MAX_DELAY_S = 300
TRANSIENT_STATUS = 429
SERVER_ERROR = 500


def validate(limits):
    attempts = limits.get("max_call_attempts", 1)
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= MAX_ATTEMPTS
    ):
        raise ValueError(f"max_call_attempts must be an integer from 1 to {MAX_ATTEMPTS}")
    backoff = limits.get("retry_backoff_s", 1)
    if (
        not isinstance(backoff, (int, float))
        or isinstance(backoff, bool)
        or not 0 < backoff <= MAX_BACKOFF_S
    ):
        raise ValueError(f"retry_backoff_s must be in (0, {MAX_BACKOFF_S}]")


def transient_status(status):
    return status == TRANSIENT_STATUS or status >= SERVER_ERROR


def retry_after_s(value):
    """Parse a Retry-After header (delta-seconds or HTTP-date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def delay(limits, attempt, exc, jitter=random.random):
    """Seconds to wait before attempt ``attempt + 1``, or None to fail closed."""
    if attempt >= limits.get("max_call_attempts", 1) or not getattr(
        exc, "transient", False
    ):
        return None
    wait = limits.get("retry_backoff_s", 1) * 2 ** (attempt - 1) * (0.5 + jitter())
    if getattr(exc, "retry_after", None) is not None:
        wait = max(wait, exc.retry_after)
    return min(wait, MAX_DELAY_S)
