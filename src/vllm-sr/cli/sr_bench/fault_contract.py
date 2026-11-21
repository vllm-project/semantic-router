from __future__ import annotations

import math
from typing import Any

_MIN_HTTP_STATUS = 100
_MAX_HTTP_STATUS = 599
_RECEIPT_FIELDS: frozenset[str] = frozenset(
    {"call_index", "status", "delay", "stream_cut_short", "provider_model"}
)


def validate_fault_schedule_entry(entry: Any, label: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} entries must be objects")
    call_index = entry.get("call_index", 0)
    if (
        isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 0
    ):
        raise ValueError(f"{label} call_index must be a non-negative integer")
    status = entry.get("status") or entry.get("status_code") or entry.get("http_status")
    if status is not None and (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not (_MIN_HTTP_STATUS <= status <= _MAX_HTTP_STATUS)
    ):
        raise ValueError(f"{label} status must be a valid HTTP status code integer")
    delay = entry.get("delay") or entry.get("delay_s")
    if delay is None and "delay_ms" in entry:
        delay = float(entry["delay_ms"]) / 1000.0
    elif delay is not None:
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
            raise ValueError(f"{label} delay must be a non-negative number")
        delay = float(delay)
    stream_cut_short = (
        entry.get("stream_cut_short")
        or entry.get("cut_short")
        or entry.get("fault") == "stream_cut_short"
        or entry.get("type") == "stream_cut_short"
    )
    if stream_cut_short is not None and not isinstance(stream_cut_short, bool):
        raise ValueError(f"{label} stream_cut_short must be a boolean")
    stream_cut_short = bool(stream_cut_short)
    if (
        status is None
        and delay is None
        and not stream_cut_short
        and not entry.get("fault")
    ):
        raise ValueError(
            f"{label} entry must specify at least one fault: status, delay, or stream_cut_short"
        )
    normalized: dict[str, Any] = {"call_index": call_index}
    if status is not None:
        normalized["status"] = status
    if delay is not None:
        normalized["delay"] = delay
    if stream_cut_short:
        normalized["stream_cut_short"] = True
    return normalized


def validate_fault_receipt(receipt: Any) -> dict[str, Any]:
    if (
        not isinstance(receipt, dict)
        or "call_index" not in receipt
        or set(receipt) - _RECEIPT_FIELDS
    ):
        raise ValueError("Invalid provider fault receipt fields")
    normalized = validate_fault_schedule_entry(
        entry=receipt, label="Provider fault receipt"
    )
    if "delay" in normalized and not math.isfinite(normalized["delay"]):
        raise ValueError("Provider fault receipt delay must be finite")
    if "provider_model" in receipt:
        model = receipt["provider_model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Provider fault receipt model must be nonempty text")
        normalized["provider_model"] = model
    return normalized


def fault_matches_schedule(
    observed: dict[str, Any] | None, scheduled: dict[str, Any] | None
) -> bool:
    return (
        isinstance(observed, dict)
        and isinstance(scheduled, dict)
        and observed.get("call_index") == scheduled.get("call_index")
        and all(
            (observed.get(key) or 0) == (scheduled.get(key) or 0)
            for key in ("status", "delay", "stream_cut_short")
        )
    )
