"""Time-bounded quarantine metadata for rollback-safe Kubernetes Secrets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

MANAGED_SECRET_QUARANTINE_ANNOTATION = (
    "semantic-router.vllm.ai/runtime-env-secret-delete-not-before"
)
# CLI deploys and the documented rollback command use a 10-minute Helm timeout.
# Keep every teardown-visible generation for another five minutes so an atomic
# or manual rollback that already loaded an old manifest cannot lose its Secret.
MANAGED_SECRET_OPERATION_GRACE = timedelta(minutes=15)
# Tolerate modest API-server/client skew without accepting an attacker-controlled
# or malformed annotation that would make ``vllm-sr stop`` wait indefinitely.
MANAGED_SECRET_MAX_CLOCK_SKEW = timedelta(minutes=5)

_RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)


@dataclass(frozen=True)
class ManagedSecretCandidate:
    """Trusted lifecycle metadata for one CLI-managed Secret generation."""

    created_at: datetime
    uid: str
    resource_version: str
    delete_not_before: datetime | None = None

    def protected_until(self) -> datetime:
        """Return the latest known time at which adoption remains possible."""
        created_grace = self.created_at + MANAGED_SECRET_OPERATION_GRACE
        if self.delete_not_before is None:
            return created_grace
        return max(created_grace, self.delete_not_before)

    def is_protected(self, now: datetime) -> bool:
        """Return whether deploy or teardown work may still adopt this Secret."""
        return now < self.protected_until()


def parse_rfc3339_timestamp(value: Any) -> datetime:
    """Parse one Kubernetes RFC3339 timestamp, rejecting ambiguous values."""
    if (
        not isinstance(value, str)
        or _RFC3339_TIMESTAMP.fullmatch(value) is None
        or value.endswith("-00:00")
    ):
        raise ValueError("timestamp must be RFC3339")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamp must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def candidate_from_metadata(metadata: Any) -> tuple[str, ManagedSecretCandidate]:
    """Build one fail-closed candidate from Kubernetes object metadata."""
    if not isinstance(metadata, dict):
        raise ValueError("managed Secret has malformed metadata")
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("managed Secret has malformed metadata")
    uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if not isinstance(uid, str) or not uid:
        raise ValueError("managed Secret has malformed metadata")
    if not isinstance(resource_version, str) or not resource_version:
        raise ValueError("managed Secret has malformed metadata")
    created_at = parse_rfc3339_timestamp(metadata.get("creationTimestamp"))

    annotations = metadata.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("managed Secret has malformed annotations")
    raw_delete_not_before = annotations.get(MANAGED_SECRET_QUARANTINE_ANNOTATION)
    delete_not_before = (
        parse_rfc3339_timestamp(raw_delete_not_before)
        if raw_delete_not_before is not None
        else None
    )
    return name, ManagedSecretCandidate(
        created_at=created_at,
        uid=uid,
        resource_version=resource_version,
        delete_not_before=delete_not_before,
    )


def quarantine_deadline(now: datetime) -> datetime:
    """Return the deletion boundary for work that starts at ``now``."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("quarantine reference time must include a timezone")
    return now.astimezone(timezone.utc) + MANAGED_SECRET_OPERATION_GRACE


def validate_quarantine_deadline(now: datetime, deadline: datetime) -> datetime:
    """Reject a lifecycle timestamp outside the bounded operation/skew window."""
    maximum = quarantine_deadline(now) + MANAGED_SECRET_MAX_CLOCK_SKEW
    if deadline > maximum:
        raise ValueError("Secret quarantine deadline exceeds the bounded window")
    return deadline


def format_rfc3339_timestamp(value: datetime) -> str:
    """Format one UTC timestamp for a Kubernetes metadata annotation."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
