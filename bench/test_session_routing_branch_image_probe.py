import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar


def load_probe_module():
    path = Path(__file__).with_name("session_routing_branch_image_probe.py")
    spec = importlib.util.spec_from_file_location(
        "session_routing_branch_image_probe", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiagnosticHandler(BaseHTTPRequestHandler):
    headers_to_emit: ClassVar[dict[str, str]] = {
        "x-vsr-selected-model": "qwen-small",
        "x-vsr-selected-decision": "agentic-session-route",
        "x-vsr-replay-id": "replay-1",
        "x-vsr-selected-confidence": "0.0000",
        "x-vsr-context-token-count": "42",
    }

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = {
            "model": "qwen/qwen3.5-rocm",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for key, value in self.headers_to_emit.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, _fmt, *_args):
        return


def run_server():
    server = HTTPServer(("127.0.0.1", 0), DiagnosticHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_branch_image_probe_passes_with_diagnostic_headers(tmp_path):
    probe = load_probe_module()
    DiagnosticHandler.headers_to_emit = {
        "x-vsr-selected-model": "qwen-small",
        "x-vsr-selected-decision": "agentic-session-route",
        "x-vsr-replay-id": "replay-1",
        "x-vsr-selected-confidence": "0.0000",
        "x-vsr-context-token-count": "42",
    }
    server, thread = run_server()
    try:
        args = probe.parse_args(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = probe.summarize(args, probe.send_chat(args))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert summary["validation_failures"] == []
    assert summary["checks"]["diagnostic_headers_ok"] is True
    probe.write_outputs(summary, tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_branch_image_probe_fails_missing_or_invalid_diagnostics(tmp_path):
    probe = load_probe_module()
    DiagnosticHandler.headers_to_emit = {
        "x-vsr-selected-model": "qwen-small",
        "x-vsr-selected-decision": "agentic-session-route",
        "x-vsr-replay-id": "replay-1",
        "x-vsr-selected-confidence": "nan",
    }
    server, thread = run_server()
    try:
        args = probe.parse_args(
            [
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        summary = probe.summarize(args, probe.send_chat(args))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert summary["checks"]["diagnostic_headers_ok"] is False
    assert (
        "invalid diagnostic header: x-vsr-selected-confidence"
        in summary["validation_failures"]
    )
    assert (
        "missing diagnostic header: x-vsr-context-token-count"
        in summary["validation_failures"]
    )
