"""Request-scoped fault schedule for provider mocker."""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import AsyncContentStream

SESSION_HEADER = "x-vsr-test-session-id"
FAULT_INJECTED_HEADER = "x-vsr-fault-injected"
FAULT_SCHEDULE_ID_HEADER = "x-vsr-fault-schedule-id"
FAULT_SESSION_ID_HEADER = "x-vsr-fault-session-id"
FAULT_CALL_INDEX_HEADER = "x-vsr-fault-call-index"
FAULT_RECEIPT_HEADER = "x-vsr-fault-receipt"
_STREAM_TERMINATORS: tuple[bytes, ...] = (
    b"data: [DONE]",
    b"event: response.completed",
    b"event: message_stop",
)


@dataclass(frozen=True)
class FaultEntry:
    call_index: int
    status: int | None = None
    delay: float | None = None
    stream_cut_short: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaultExecution:
    tracker: FaultScheduleTracker
    counter_key: str
    call_index: int
    fault: FaultEntry | None


def request_fault(request: Request) -> FaultExecution | None:
    schedule_key, counter_key = get_fault_keys(request.headers)
    if not schedule_key:
        return None
    raw_index = request.headers.get(FAULT_CALL_INDEX_HEADER)
    index = None
    if raw_index is not None:
        try:
            if not raw_index.isascii() or not raw_index.isdecimal():
                raise ValueError
            index = int(raw_index)
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="x-vsr-fault-call-index must be a non-negative integer",
            ) from exc
    tracker = request.app.state.fault_tracker
    call_index, fault = tracker.record_call_and_match(
        schedule_key=schedule_key, counter_key=counter_key, request_index=index
    )
    return FaultExecution(
        tracker=tracker, counter_key=counter_key, call_index=call_index, fault=fault
    )


def apply_response_fault(
    *,
    response: Response | dict[str, Any],
    fault: FaultEntry | None,
    execution: FaultExecution | None = None,
    provider_model: str | None = None,
) -> Response | dict[str, Any]:
    applied = fault is not None and (
        fault.status is not None
        or fault.stream_cut_short
        or (fault.delay is not None and fault.delay > 0)
    )
    receipt = (
        execution.tracker.observed_fault(
            counter_key=execution.counter_key, call_index=execution.call_index
        )
        if execution is not None
        else None
    )
    if not applied and receipt is None:
        return response
    rendered = (
        response if isinstance(response, Response) else JSONResponse(content=response)
    )
    if applied and fault is not None and fault.stream_cut_short:
        if isinstance(rendered, StreamingResponse):
            rendered.body_iterator = _cut_stream(rendered.body_iterator)
            if "content-length" in rendered.headers:
                del rendered.headers["content-length"]
        else:
            rendered.body = rendered.body[:1]
            rendered.headers["content-length"] = str(len(rendered.body))
    if applied and fault is not None:
        receipt = {"call_index": fault.call_index}
        if fault.status is not None:
            receipt["status"] = fault.status
        if fault.delay is not None:
            receipt["delay"] = fault.delay
        if fault.stream_cut_short:
            receipt["stream_cut_short"] = True
        if provider_model:
            receipt["provider_model"] = provider_model
        if execution is not None:
            execution.tracker.record_fault(
                counter_key=execution.counter_key,
                call_index=execution.call_index,
                receipt=receipt,
            )
    rendered.headers[FAULT_INJECTED_HEADER] = "true"
    rendered.headers[FAULT_RECEIPT_HEADER] = json.dumps(
        receipt, separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    return rendered


async def _cut_stream(body: AsyncContentStream) -> AsyncIterator[bytes]:
    stream = aiter(body)
    pending = bytearray()
    overlap = max(map(len, _STREAM_TERMINATORS)) - 1
    emitted = False
    try:
        async for chunk in stream:
            pending.extend(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            positions = (pending.find(marker) for marker in _STREAM_TERMINATORS)
            terminal = min(
                (position for position in positions if position >= 0), default=None
            )
            if terminal is not None:
                if terminal:
                    yield bytes(pending[:terminal])
                elif not emitted:
                    yield bytes(pending[:1])
                return
            # A terminal frame can span the response iterator's chunk boundaries.
            if len(pending) > overlap:
                yield bytes(pending[:-overlap])
                del pending[:-overlap]
                emitted = True
        if pending and not emitted:
            yield bytes(pending[:1])
    finally:
        if isinstance(stream, AsyncGenerator):
            await stream.aclose()


def get_fault_keys(headers: Mapping[str, str] | None) -> tuple[str, str]:
    """Extract (schedule_key, counter_key) from incoming HTTP headers with fixed precedence."""
    if not headers:
        return "", ""
    norm_headers = {k.lower(): v.strip() for k, v in headers.items() if v and v.strip()}
    schedule_key = norm_headers.get(FAULT_SCHEDULE_ID_HEADER, "")
    # Explicit fault-session header has fixed precedence over session fallbacks
    counter_key = (
        norm_headers.get(FAULT_SESSION_ID_HEADER)
        or norm_headers.get(SESSION_HEADER)
        or norm_headers.get("x-session-id")
        or ""
    )
    if not schedule_key and counter_key:
        schedule_key = counter_key
    if not counter_key and schedule_key:
        counter_key = schedule_key
    return schedule_key, counter_key


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
        self._claimed_faults: set[tuple[str, int]] = set()
        self._observed_faults: dict[tuple[str, int], dict[str, Any]] = {}

    def set_schedule(self, schedule: Any) -> None:
        with self._lock:
            self._schedule = parse_fault_schedule(schedule)

    def reset(self) -> None:
        with self._lock:
            self._call_counts.clear()
            self._claimed_faults.clear()
            self._observed_faults.clear()

    def record_call_and_match(
        self,
        schedule_key: str,
        counter_key: str | None = None,
        *,
        request_index: int | None = None,
    ) -> tuple[int, FaultEntry | None]:
        counter = counter_key or schedule_key
        if not counter and not schedule_key:
            return 0, None
        with self._lock:
            call_index = (
                self._call_counts[counter] if request_index is None else request_index
            )
            self._call_counts[counter] += 1
            key_schedule = self._schedule.get(schedule_key) or self._schedule.get(
                counter
            )
            if key_schedule and call_index in key_schedule:
                claim = (counter, call_index)
                if claim in self._claimed_faults:
                    return call_index, None
                self._claimed_faults.add(claim)
                return call_index, key_schedule[call_index]
            return call_index, None

    def record_fault(
        self, *, counter_key: str, call_index: int, receipt: dict[str, Any]
    ) -> None:
        with self._lock:
            self._observed_faults[(counter_key, call_index)] = dict(receipt)

    def observed_fault(
        self, *, counter_key: str, call_index: int
    ) -> dict[str, Any] | None:
        with self._lock:
            receipt = self._observed_faults.get((counter_key, call_index))
            return dict(receipt) if receipt is not None else None

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "call_counts": dict(self._call_counts),
                "keys": list(self._schedule.keys()),
            }
