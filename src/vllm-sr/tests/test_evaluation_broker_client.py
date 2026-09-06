from __future__ import annotations

import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cli.evaluation.broker_client import (
    _OPERATION_TRACKS,
    _OPERATIONS,
    _READ_ONLY_OPERATIONS,
    WorkerHTTPBroker,
)


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = os.read(descriptor, size)
        if not chunk:
            raise AssertionError("broker request pipe closed early")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _read_frame(descriptor: int) -> dict[str, Any]:
    size = struct.unpack("!I", _read_exact(descriptor, 4))[0]
    value = json.loads(_read_exact(descriptor, size))
    assert isinstance(value, dict)
    return value


def _write_frame(descriptor: int, value: dict[str, Any]) -> None:
    body = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    frame = memoryview(struct.pack("!I", len(body)) + body)
    while frame:
        frame = frame[os.write(descriptor, frame) :]


def test_worker_http_broker_multiplexes_concurrent_out_of_order_responses() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    broker = WorkerHTTPBroker(request_write, response_read)
    observed_ids: list[int] = []

    def serve() -> None:
        requests = [_read_frame(request_read), _read_frame(request_read)]
        observed_ids.extend(request["id"] for request in requests)
        for request in reversed(requests):
            _write_frame(
                response_write,
                {
                    "id": request["id"],
                    "success": True,
                    "status_code": 200,
                    "payload": {"echo": request["operation"]},
                    "latency_ms": 1.25,
                    "fetched_at": "2026-08-31T00:00:00Z",
                    "headers": {},
                    "error": None,
                    "broker_receipt": "sha256:" + "a" * 64,
                },
            )

    server = threading.Thread(target=serve)
    server.start()
    operations = ("models.list", "routed-chat.completions")
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = tuple(
            pool.map(
                lambda operation: broker.request(
                    operation,
                    None if operation == "models.list" else {},
                    2,
                    track_id=None if operation == "models.list" else "capacity",
                    case_id=None if operation == "models.list" else "case-1",
                    attempt_id=None if operation == "models.list" else "attempt-1",
                ),
                operations,
            )
        )
    server.join(timeout=2)
    for descriptor in (request_read, request_write, response_read, response_write):
        os.close(descriptor)

    assert observed_ids == [1, 2]
    assert {response["payload"]["echo"] for response in responses} == set(operations)


def test_broker_chat_operations_are_track_specific_without_a_generic_alias() -> None:
    assert "chat.completions" not in _OPERATIONS
    assert _OPERATION_TRACKS["arm-chat.completions"] == {"model_pool"}
    assert _OPERATION_TRACKS["routed-chat.completions"] == {
        "joint",
        "multimodal",
        "capacity",
    }


def test_agent_task_ledger_is_a_read_only_agentic_broker_operation() -> None:
    assert "agent-task.ledger" in _OPERATIONS
    assert "agent-task.ledger" in _READ_ONLY_OPERATIONS
    assert _OPERATION_TRACKS["agent-task.ledger"] == {"agentic"}
