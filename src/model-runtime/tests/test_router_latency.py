"""The router latency tool's connection-error tolerance (``tools/router_latency.py``)."""

import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "router_latency.py"
# preview() retries once, so the failing first input costs two attempts in
# the sequential pass and two more for the diagnostic first_response call.
MIN_FIRST_INPUT_ATTEMPTS = 4


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("router_latency", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DroppingRouter(http.server.ThreadingHTTPServer):
    """Drops the connection for every request whose text starts with ``boom``."""

    lock = threading.Lock()
    attempts: ClassVar[dict[str, int]] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            with _DroppingRouter.lock:
                _DroppingRouter.attempts[text] = (
                    _DroppingRouter.attempts.get(text, 0) + 1
                )
            if text.startswith("boom"):
                self.connection.close()
                return
            payload = json.dumps(
                {"decision": {"decision_name": "stable"}, "recommended_model": "m1"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass


def test_run_survives_first_input_disconnects(tool, tmp_path, monkeypatch):
    """A first input that keeps disconnecting must not discard completed passes."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(["boom-one", "ok-one"]), encoding="utf-8")
    out = tmp_path / "out.json"
    _DroppingRouter.attempts = {}
    server = _DroppingRouter(("127.0.0.1", 0), _DroppingRouter.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "router_latency.py",
            "run",
            "--url",
            f"http://127.0.0.1:{server.server_port}",
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--label",
            "test",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--concurrency",
            "2",
        ],
    )
    try:
        assert tool.main() is None
    finally:
        server.shutdown()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert _DroppingRouter.attempts["boom-one"] >= MIN_FIRST_INPUT_ATTEMPTS
    assert result["first_response"].get("error") == "first input request failed"
    assert result["passes"]["sequential"]["errors"] == 1
    assert result["passes"]["concurrency_2"]["errors"] == 1
    assert result["outcomes"][0] == {
        "status": "error",
        "decision": None,
        "model": None,
    }
    assert result["outcomes"][1]["status"] == http.HTTPStatus.OK
    assert result["outcomes"][1]["decision"] == "stable"
