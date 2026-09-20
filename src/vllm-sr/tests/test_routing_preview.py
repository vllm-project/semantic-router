"""Supported Preview envelopes stay equal across the CLI and benchmark paths."""

import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.commands.route import preview as preview_command
from cli.routing_preview import build_preview_request, case_request_fields
from cli.sr_bench.contracts import plan
from cli.sr_bench.engine import Engine
from cli.sr_bench.store import TERMINAL, Store
from click.testing import CliRunner


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.seen.append((self.path, payload))
            self.send_response(200)
            if self.path.startswith("/api/v1/routing/preview"):
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "decision_result": {
                                "decision_name": "route",
                                "plugins": [],
                            },
                            "selected_model": "model",
                            "selection_status": "selected",
                            "selection_method": "static",
                            "selection_provenance": {
                                "mode": "read_only_snapshot",
                                "state_dependent": True,
                                "sampled": True,
                                "sampling_seed": 17,
                            },
                        }
                    ).encode()
                )
            else:
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                event = {
                    "model": "model",
                    "choices": [
                        {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                }
                self.wfile.write(
                    ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def envelope():
    return {
        "model": "entrypoint",
        "messages": [
            {"role": "system", "content": "Use the available function."},
            {"role": "user", "content": [{"type": "text", "text": "Return A"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "A"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "metadata": {"request_class": "fixture"},
        "max_completion_tokens": 17,
        "preview_context": {
            "session_id": "from-file",
            "conversation_id": "conversation",
        },
    }


def test_shared_envelope_preserves_prompt_inputs_without_mutation():
    original = envelope()
    saved = copy.deepcopy(original)
    value = build_preview_request(original, preview_context={"sampling_seed": 17})
    assert original == saved
    assert value["messages"] == original["messages"]
    assert value["tools"] == original["tools"]
    assert value["preview_context"] == {
        **original["preview_context"],
        "sampling_seed": 17,
    }
    assert value["max_completion_tokens"] == 17
    assert case_request_fields({"metadata": {"reference": "private-answer"}}) == {}
    with pytest.raises(ValueError, match="metadata"):
        case_request_fields({"request_metadata": {"nested": {"key": "value"}}})


@pytest.mark.parametrize("field", ["temperature", "stream", "top_p", "seed", "user"])
def test_completion_only_fields_are_rejected_without_silent_filtering(field):
    with pytest.raises(ValueError, match="unsupported"):
        build_preview_request({**envelope(), field: 1})


def test_cli_request_file_context_and_readable_selection(endpoint, tmp_path):
    request = tmp_path / "preview.json"
    request.write_text(json.dumps(envelope()))
    result = CliRunner().invoke(
        preview_command,
        [
            "--request-file",
            str(request),
            "--endpoint",
            f"http://127.0.0.1:{endpoint.server_port}",
            "--session-id",
            "explicit-session",
            "--sampling-seed",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = endpoint.seen[0][1]
    assert payload["messages"] == envelope()["messages"]
    assert payload["tools"] == envelope()["tools"]
    assert payload["metadata"] == envelope()["metadata"]
    assert payload["max_completion_tokens"] == 17
    assert payload["preview_context"] == {
        "session_id": "explicit-session",
        "conversation_id": "conversation",
        "sampling_seed": 17,
    }
    assert "model" in result.output and "selected" in result.output
    assert (
        "snapshot" in result.output
        and "later live selection may differ" in result.output
    )
    for extra in (["--prompt", "conflict"], ["--messages", "[]"]):
        rejected = CliRunner().invoke(
            preview_command, ["--request-file", str(request), *extra]
        )
        assert rejected.exit_code != 0
    request.write_text(json.dumps({**envelope(), "temperature": 0}))
    rejected = CliRunner().invoke(preview_command, ["--request-file", str(request)])
    assert rejected.exit_code != 0
    assert len(endpoint.seen) == 1


def test_benchmark_preview_and_initial_subject_keep_identical_request_inputs(
    endpoint, tmp_path
):
    origin = f"http://127.0.0.1:{endpoint.server_port}"
    request = envelope()
    case = {
        "id": "one",
        "benchmark": "mmlu-pro",
        "answer": "A",
        "messages": request["messages"],
        "tools": request["tools"],
        "tool_choice": request["tool_choice"],
        "response_format": request["response_format"],
        "request_metadata": request["metadata"],
        "metadata": {"split": "dev", "reference": "NEVER_SEND_REFERENCE"},
    }
    document = {
        "version": "sr-bench-1.0",
        "cases": [case],
        "targets": [
            {
                "id": "model",
                "kind": "single",
                "model": "model",
                "base_url": origin + "/v1",
                "request_params": {"max_tokens": 17},
                "prices": {
                    "model": dict.fromkeys(
                        ("input", "cached_input", "cache_write", "output"), 1
                    )
                },
            }
        ],
        "limits": {"max_run_seconds": 10, "total_timeout_s": 2, "idle_timeout_s": 2},
    }
    store = Store(tmp_path)
    engine = Engine(store)
    for mode in ("live", "preview"):
        if mode == "preview":
            document = copy.deepcopy(document)
            document["mode"] = "preview"
            document["targets"][0].update(
                kind="mom", preview_url=origin + "/api/v1/routing/preview"
            )
            document["preview_context"] = {
                "session_id": "session",
                "conversation_id": "conversation",
                "sampling_seed": 17,
            }
        frozen = plan(document)
        assert frozen["cases"][0] == case
        run = engine.start(document)
        deadline = time.monotonic() + 5
        while (
            store.get(run["id"])["status"] not in TERMINAL
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert store.get(run["id"])["status"] == "completed"
    live, preview = [body for _, body in endpoint.seen]
    for key in (
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "metadata",
        "max_tokens",
    ):
        assert live[key] == preview[key]
    assert live["max_tokens"] == 17
    assert preview["preview_context"] == document["preview_context"]
    assert "NEVER_SEND_REFERENCE" not in json.dumps(endpoint.seen)
