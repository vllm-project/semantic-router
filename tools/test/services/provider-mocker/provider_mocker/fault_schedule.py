"""Request-scoped fault schedule for provider mocker."""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

FAULT_KEY_HEADER = "x-vsr-fault-key"
SESSION_HEADER = "x-vsr-test-session-id"
FAULT_INJECTED_HEADER = "x-vsr-fault-injected"


@dataclass(frozen=True)
class FaultEntry:
    call_index: int
    status: int | None = None
    delay: float | None = None
    stream_cut_short: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def get_fault_key(headers: Mapping[str, str] | None) -> str:
    """Extract the request-scoped fault key from incoming HTTP headers."""
    if not headers:
        return ""
    for name, value in headers.items():
        norm = name.lower()
        if norm in (
            FAULT_KEY_HEADER,
            SESSION_HEADER,
            "x-session-id",
            "x-fault-schedule-id",
        ):
            val = value.strip()
            if val:
                return val
    return ""


def parse_fault_entry(call_index: int, data: dict[str, Any]) -> FaultEntry:
    status = data.get("status") or data.get("status_code") or data.get("http_status")
    if status is not None:
        status = int(status)

    delay = data.get("delay") or data.get("delay_s")
    if delay is None and "delay_ms" in data:
        delay = float(data["delay_ms"]) / 1000.0
    elif delay is not None:
        delay = float(delay)

    stream_cut_short = bool(
        data.get("stream_cut_short")
        or data.get("cut_short")
        or data.get("fault") == "stream_cut_short"
        or data.get("type") == "stream_cut_short"
    )

    fault_type = data.get("fault") or data.get("type")
    if fault_type == "status" and status is None:
        status = 503
    elif fault_type == "delay" and delay is None:
        delay = 1.0

    return FaultEntry(
        call_index=call_index,
        status=status,
        delay=delay,
        stream_cut_short=stream_cut_short,
        raw=dict(data),
    )


def parse_fault_schedule(raw: Any) -> dict[str, dict[int, FaultEntry]]:
    """Parse fault schedule from dict, list or JSON file/string."""
    if isinstance(raw, str):
        trimmed = raw.strip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            try:
                raw = json.loads(trimmed)
            except json.JSONDecodeError:
                return {}
        elif os.path.exists(trimmed):
            try:
                raw = json.loads(Path(trimmed).read_text(encoding="utf-8"))
            except Exception:
                return {}
        else:
            return {}

    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, dict[int, FaultEntry]] = defaultdict(dict)
    for key, entries in raw.items():
        key_str = str(key).strip()
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    idx = item.get("call_index", 0)
                    try:
                        idx_int = int(idx)
                    except (ValueError, TypeError):
                        idx_int = 0
                    parsed[key_str][idx_int] = parse_fault_entry(idx_int, item)
        elif isinstance(entries, dict):
            for idx_key, item in entries.items():
                try:
                    idx_int = int(idx_key)
                except (ValueError, TypeError):
                    continue
                if isinstance(item, dict):
                    parsed[key_str][idx_int] = parse_fault_entry(idx_int, item)

    return dict(parsed)


class FaultScheduleTracker:
    def __init__(self, initial_schedule: Any = None) -> None:
        self._lock = threading.Lock()
        self._schedule: dict[str, dict[int, FaultEntry]] = parse_fault_schedule(
            initial_schedule
        )
        self._call_counts: dict[str, int] = defaultdict(int)

    def set_schedule(self, schedule: Any) -> None:
        with self._lock:
            self._schedule = parse_fault_schedule(schedule)

    def reset(self) -> None:
        with self._lock:
            self._call_counts.clear()

    def record_call_and_match(self, key: str) -> tuple[int, FaultEntry | None]:
        if not key:
            return 0, None
        with self._lock:
            call_index = self._call_counts[key]
            self._call_counts[key] += 1
            key_schedule = self._schedule.get(key)
            if key_schedule and call_index in key_schedule:
                return call_index, key_schedule[call_index]
            return call_index, None

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "call_counts": dict(self._call_counts),
                "keys": list(self._schedule.keys()),
            }
