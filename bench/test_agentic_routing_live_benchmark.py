import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

EXPECTED_REQUESTS = 4


def load_live_module():
    path = Path(__file__).with_name("agentic_routing_live_benchmark.py")
    spec = importlib.util.spec_from_file_location(
        "agentic_routing_live_benchmark", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EchoHandler(BaseHTTPRequestHandler):
    selected_models: ClassVar[list[str]] = ["small", "small", "small", "frontier"]
    seen_session_headers: ClassVar[list[str]] = []
    request_count = 0

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        json.loads(raw.decode("utf-8"))
        EchoHandler.seen_session_headers.append(self.headers.get("x-session-id", ""))
        idx = EchoHandler.request_count
        EchoHandler.request_count += 1
        selected_model = EchoHandler.selected_models[
            idx % len(EchoHandler.selected_models)
        ]
        payload = {
            "id": f"resp_{idx}",
            "model": selected_model,
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100 + idx,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-vsr-selected-model", selected_model)
        self.send_header("x-vsr-selected-decision", "agentic-session-route")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, _fmt, *_args):
        return


def make_args(base_url, tmp_path, **overrides):
    args = {
        "base_url": base_url,
        "model": "auto",
        "api_key": "",
        "session_header": "x-session-id",
        "scenario": "tool-heavy",
        "sessions": 1,
        "turns": 4,
        "concurrency": 1,
        "max_tokens": 8,
        "temperature": 0.0,
        "timeout": 5.0,
        "label": "test",
        "include_previous_response_id": False,
        "metrics_url": "",
        "extra_header": [],
        "dry_run": False,
        "output_dir": tmp_path,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_live_benchmark_records_router_headers_and_violations(tmp_path):
    live = load_live_module()
    EchoHandler.request_count = 0
    EchoHandler.seen_session_headers = []
    server = HTTPServer(("127.0.0.1", 0), EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = make_args(f"http://127.0.0.1:{server.server_port}/v1", tmp_path)
        rows = live.run_benchmark(args)
        summary = live.summarize(rows)
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert len(rows) == EXPECTED_REQUESTS
    assert all(row["success"] for row in rows)
    assert all(EchoHandler.seen_session_headers)
    assert rows[0]["x-vsr-selected-decision"] == "agentic-session-route"
    assert summary["requests"] == EXPECTED_REQUESTS
    assert summary["success_rate"] == 1.0
    assert summary["model_switches"] == 1
    assert summary["tool_loop_switch_violations"] == 1
    assert summary["cached_prompt_ratio"] is not None


def test_previous_response_id_marks_context_portability_violation(tmp_path):
    live = load_live_module()
    args = make_args(
        "http://unused/v1", tmp_path, scenario="stateful-heavy", dry_run=True
    )
    rows = [
        {
            **live.row_from_result(
                args,
                live.TurnPlan("s", 0, "user_turn", "a", ""),
                {
                    "status": 200,
                    "latency_ms": 1,
                    "headers": {"x-vsr-selected-model": "small"},
                    "json": {"id": "resp_0", "model": "small", "usage": {}},
                    "error": "",
                },
                "",
            )
        },
        {
            **live.row_from_result(
                args,
                live.TurnPlan("s", 1, "provider_state", "b", "resp_0"),
                {
                    "status": 200,
                    "latency_ms": 1,
                    "headers": {"x-vsr-selected-model": "frontier"},
                    "json": {"id": "resp_1", "model": "frontier", "usage": {}},
                    "error": "",
                },
                "small",
            )
        },
    ]
    summary = live.summarize(rows)

    assert rows[1]["previous_response_id_sent"]
    assert rows[1]["context_portability_violation"]
    assert summary["context_portability_violations"] == 1
