"""Buffered SSE reading must parse exactly like the byte-at-a-time reader."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.sr_bench import transport
from cli.sr_bench.transport import CallFailure, chat

LIMITS = {
    "total_timeout_s": 30,
    "idle_timeout_s": 30,
    "max_output_tokens": 1_000_000,
    "max_output_chars": 1_000_000,
    "repetition_window": 8,
    "repetition_limit": 5,
}
VOLATILE = {"latency_s", "ttft_s"}


def _event(delta, finish=None):
    return {
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(events, crlf=False):
    newline = "\r\n" if crlf else "\n"
    return "".join(
        f"data: {json.dumps(e, ensure_ascii=False)}{newline}{newline}" for e in events
    ).encode()


BODY = (
    _sse([_event({"reasoning_content": "思考 é"})], crlf=True)
    # One JSON event split across two data: lines.
    + b'data: {"model": "m", "choices": [{"index": 0,\ndata:  "delta": {"content": "H\xc3\xa9llo "}}]}\n\n'
    + _sse(
        [
            _event({"content": "世界 🙂"}),
            _event({"content": " done"}, finish="stop"),
            {"model": "m", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
        ]
    )
    + b"data: [DONE]\n\n"
)


class Replay(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        body, sizes, chunked = self.server.body, self.server.sizes, self.server.chunked
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        offset, index = 0, 0
        while offset < len(body):
            piece = body[offset : offset + sizes[index % len(sizes)]]
            offset, index = offset + len(piece), index + 1
            frame = b"%x\r\n%s\r\n" % (len(piece), piece) if chunked else piece
            self.wfile.write(frame)
            self.wfile.flush()
            if self.server.pause:
                time.sleep(self.server.pause)
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        self.close_connection = True


class Server(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def replay():
    servers = []

    def start(body, sizes, chunked, pause=0.0):
        handler = type("H", (Replay,), {"protocol_version": "HTTP/1.1" if chunked else "HTTP/1.0"})
        server = Server(("127.0.0.1", 0), handler)
        server.body, server.sizes, server.chunked, server.pause = body, sizes, chunked, pause
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return {
            "id": "t",
            "kind": "single",
            "model": "m",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        }

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _call(target, limits=LIMITS):
    result = chat(target, [{"role": "user", "content": "q"}], {}, limits, lambda: False)
    return {k: v for k, v in result.items() if k not in VOLATILE}


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize(
    "sizes",
    [[1], [2], [3], [7], [1, 5, 2, 11], [4096], [len(BODY)]],
)
def test_awkward_chunking_parses_identically_to_byte_reader(
    replay, monkeypatch, sizes, chunked
):
    buffered = _call(replay(BODY, sizes, chunked, pause=0.0005 if len(sizes) > 1 else 0))
    with monkeypatch.context() as patch:
        patch.setattr(transport, "read_chunks", lambda r: r.iter_content(chunk_size=1))
        reference = _call(replay(BODY, [len(BODY)], chunked))
    assert buffered == reference
    assert buffered["final"] == "Héllo 世界 🙂 done"
    assert buffered["reasoning"] == "思考 é"
    assert buffered["finish_reason"] == "stop"
    assert buffered["usage"]["output_tokens"] == 4


def test_guards_still_fire_inside_large_chunks(replay):
    repeated = _sse([_event({"content": "abcdefgh"}) for _ in range(6)])
    with pytest.raises(CallFailure, match="Repeated output guard"):
        _call(replay(repeated + b"data: [DONE]\n\n", [len(repeated) + 20], True))
    long_text = _sse([_event({"content": "x" * 200}, finish="stop")])
    with pytest.raises(CallFailure, match="Output character cap"):
        _call(
            replay(long_text, [len(long_text)], True),
            {**LIMITS, "max_output_chars": 150, "repetition_limit": 1000},
        )
    with pytest.raises(CallFailure, match="SSE line cap"):
        _call(
            replay(b"data: " + b"y" * 200, [4096], True),
            {**LIMITS, "max_output_chars": 50},
        )


def test_split_utf8_is_not_decoded_before_the_line_ends(replay):
    body = _sse([_event({"content": "🙂"}, finish="stop")]) + b"data: [DONE]\n\n"
    cut = body.index("🙂".encode()) + 2
    result = _call(replay(body, [cut, 1, 1, len(body)], True, pause=0.01))
    assert result["final"] == "🙂"
