"""In-process HTTP fixture for eval command integration tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest


def _make_handler(status: int, body: Any, content_type: str):
    if isinstance(body, bytes):
        body_bytes = body
    elif isinstance(body, str):
        body_bytes = body.encode()
    else:
        body_bytes = json.dumps(body).encode()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, _format, *_args):
            return

    return _Handler


@pytest.fixture()
def router_server(request):
    """Serve the response supplied by an indirect fixture parameter."""

    params = request.param
    handler = _make_handler(
        params["status"],
        params["body"],
        params.get("content_type", "application/json"),
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
