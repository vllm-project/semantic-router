"""Stream evidence reaches the filesystem before the final sync, even on failure."""

import json
import os
from unittest.mock import Mock

import pytest
import requests
from cli.sr_bench import transport


def _frame(content, finish_reason=None):
    event = {
        "model": "model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return ("data: " + json.dumps(event) + "\n\n").encode()


@pytest.fixture
def stream_call(monkeypatch):
    def call(payload, path, *, read_timeout=False):
        def chunks(chunk_size):
            for offset in range(0, len(payload), chunk_size):
                yield payload[offset : offset + chunk_size]
            if read_timeout:
                raise requests.ReadTimeout("synthetic timeout")

        response = Mock(status_code=200, headers={"content-type": "text/event-stream"})
        response.iter_content.side_effect = chunks
        monkeypatch.setattr(transport.requests, "post", Mock(return_value=response))
        try:
            return transport.chat(
                {
                    "kind": "single",
                    "model": "model",
                    "base_url": "http://fixture.invalid/v1",
                },
                [{"role": "user", "content": "A"}],
                {"max_tokens": 100},
                {
                    "max_output_tokens": 100,
                    "max_output_chars": 1000,
                    "repetition_window": 32,
                    "repetition_limit": 4,
                    "idle_timeout_s": 2,
                    "total_timeout_s": 5,
                },
                lambda: False,
                stream_path=path,
            )
        finally:
            response.close.assert_called_once()

    return call


@pytest.mark.parametrize(
    "payload, read_timeout, error",
    [
        pytest.param(b'data: {"choices": [', True, "ReadTimeout", id="read-timeout"),
        pytest.param(
            b"data: {broken}\n\n", False, "JSONDecodeError", id="invalid-json"
        ),
        pytest.param(
            _frame("x" * 1200), False, "Output character cap exceeded", id="output-cap"
        ),
        pytest.param(
            _frame("A") + b'data: {"usage":',
            True,
            "ReadTimeout",
            id="partial-after-flushed-frame",
        ),
        pytest.param(
            b": keepalive\n" * 700, True, "ReadTimeout", id="exceeds-write-buffer"
        ),
        pytest.param(
            b'data: {"choices": [', False, "Incomplete final response", id="early-eof"
        ),
        pytest.param(b"", True, "ReadTimeout", id="empty-stream"),
        pytest.param(
            _frame("A", "stop") + b"data: [DONE]\n\n", False, None, id="complete-stream"
        ),
    ],
)
def test_stream_evidence_is_flushed_before_fsync(
    tmp_path, monkeypatch, stream_call, payload, read_timeout, error
):
    path = tmp_path / "response.sse"
    synced = []
    fsync = os.fsync

    def observe_fsync(fd):
        fsync(fd)
        # A separate reader cannot see bytes still in the writer's Python buffer.
        synced.append(path.read_bytes())

    monkeypatch.setattr(transport.os, "fsync", observe_fsync)
    if error is None:
        result = stream_call(payload, path, read_timeout=read_timeout)
        assert result["final"] == "A"
    else:
        with pytest.raises(transport.CallFailure, match=error):
            stream_call(payload, path, read_timeout=read_timeout)
    assert synced == [payload]
    assert path.read_bytes() == payload


@pytest.mark.parametrize("boundary", ["flush", "fsync"])
def test_stream_file_closes_when_persistence_fails(
    tmp_path, monkeypatch, stream_call, boundary
):
    path = tmp_path / "response.sse"
    stream = path.open("xb", buffering=8192)
    writer = Mock(wraps=stream)
    monkeypatch.setattr(transport, "open", Mock(return_value=writer), raising=False)
    sync = Mock()
    monkeypatch.setattr(transport.os, "fsync", sync)
    failing_call = writer.flush if boundary == "flush" else sync
    failing_call.side_effect = OSError("synthetic persistence failure")
    try:
        with pytest.raises(OSError, match="synthetic persistence failure"):
            stream_call(b'data: {"choices": [', path, read_timeout=True)
        assert stream.closed
        if boundary == "flush":
            sync.assert_not_called()
    finally:
        stream.close()
