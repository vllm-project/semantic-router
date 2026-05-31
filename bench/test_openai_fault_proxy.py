import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

HTTP_OK = 200
HTTP_UNAVAILABLE = 503


def load_fault_proxy_module():
    path = Path(__file__).with_name("openai_fault_proxy.py")
    spec = importlib.util.spec_from_file_location("openai_fault_proxy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UpstreamEchoHandler(BaseHTTPRequestHandler):
    seen_paths: ClassVar[list[str]] = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        json.loads(raw.decode("utf-8"))
        UpstreamEchoHandler.seen_paths.append(self.path)
        payload = {
            "id": "resp_echo",
            "model": "small",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _fmt, *_args):
        return


def start_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    thread.join(timeout=2)


def post_turn(base_url, session_idx, turn, prompt_suffix=""):
    prompt = f"Session {session_idx}, turn {turn}. {prompt_suffix or 'continue'}"
    payload = {"model": "auto", "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-session-id": f"session-{session_idx}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read())


def test_fault_proxy_injects_selected_turn_once_then_recovers(tmp_path):
    fault = load_fault_proxy_module()
    UpstreamEchoHandler.seen_paths = []
    upstream, upstream_thread = start_server(UpstreamEchoHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset({1}),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
    )
    handler = fault.make_handler(
        f"http://127.0.0.1:{upstream.server_port}",
        policy,
        tmp_path / "faults.jsonl",
    )
    proxy, proxy_thread = start_server(handler)

    try:
        base_url = f"http://127.0.0.1:{proxy.server_port}"
        assert post_turn(base_url, 0, 0)[0] == HTTP_OK

        status, headers, payload = post_turn(base_url, 0, 1)
        assert status == HTTP_UNAVAILABLE
        assert headers["x-vsr-fault-injected"] == "true"
        assert payload["error"]["type"] == "fault_proxy_injected"

        assert post_turn(base_url, 0, 1)[0] == HTTP_OK
        assert post_turn(base_url, 0, 2)[0] == HTTP_OK
    finally:
        stop_server(proxy, proxy_thread)
        stop_server(upstream, upstream_thread)

    events = [
        json.loads(line)
        for line in (tmp_path / "faults.jsonl").read_text().splitlines()
    ]
    assert [event["action"] for event in events] == [
        "forwarded",
        "injected",
        "forwarded",
        "forwarded",
    ]
    assert UpstreamEchoHandler.seen_paths == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]


def test_fault_proxy_session_modulo_selects_subset():
    fault = load_fault_proxy_module()
    upstream, upstream_thread = start_server(UpstreamEchoHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset({0}),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=2,
        fail_session_remainder=1,
        fail_once_per_session=True,
    )
    handler = fault.make_handler(f"http://127.0.0.1:{upstream.server_port}", policy)
    proxy, proxy_thread = start_server(handler)

    try:
        base_url = f"http://127.0.0.1:{proxy.server_port}"
        assert post_turn(base_url, 0, 0)[0] == HTTP_OK
        assert post_turn(base_url, 1, 0)[0] == HTTP_UNAVAILABLE
    finally:
        stop_server(proxy, proxy_thread)
        stop_server(upstream, upstream_thread)


def test_fault_proxy_phase_inference_targets_tool_loops():
    fault = load_fault_proxy_module()
    assert fault.phase_from_text("Use the provided tool result now") == "tool_loop"
    assert fault.phase_from_text("Continue from the previous response state") == (
        "provider_state"
    )
    assert fault.phase_from_text("This follows an idle pause") == "idle_boundary"
