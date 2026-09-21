"""Dataset downloads use the selected service and never fall back to local writes."""

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from cli.sr_bench import VERSION

command = importlib.import_module("cli.commands.benchmark")
preparations = importlib.import_module("cli.commands.benchmark_preparations")
PREPARATION_ID = "prep-" + "a1" * 16


@pytest.fixture
def service(tmp_path, monkeypatch):
    for name in ("SR_BENCH_URL", "SR_BENCH_STORE", "SR_BENCH_TOKEN_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SR_BENCH_TOKEN", "fixture-service-token")
    monkeypatch.setattr(preparations.time, "sleep", lambda _: None)
    local_prepare = Mock(side_effect=AssertionError("Remote preparation wrote locally"))
    monkeypatch.setattr(command.sources, "prepare_dataset", local_prepare)
    state = {
        "requests": [],
        "reads": 0,
        "status": "completed",
        "manifest": {
            "id": "b" * 64,
            "case_count": 5,
            "path": "/worker/store/cases.jsonl",
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def handle_request(self):
            assert self.headers["Authorization"] == "Bearer fixture-service-token"
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["requests"].append(
                (self.command, self.path, json.loads(body) if body else None)
            )
            status = 200
            preparation = {"id": PREPARATION_ID, "status": "queued", "phase": "queued"}
            if self.path == "/health":
                data = {"version": VERSION}
            elif self.command == "POST":
                status = state.get("post_status", 202)
                data = (
                    {"preparation": preparation}
                    if status == 202
                    else {"error": "Preparation service is unavailable"}
                )
            elif self.path.endswith("/" + PREPARATION_ID):
                state["reads"] += 1
                preparation["status"] = (
                    "running" if state["reads"] == 1 else state["status"]
                )
                preparation["phase"] = "downloading"
                if preparation["status"] == "completed":
                    preparation["dataset"] = state["manifest"]
                if preparation["status"] == "failed":
                    preparation["error"] = (
                        "Source access denied; configure worker credentials"
                    )
                data = {"preparation": preparation}
            elif self.path.endswith("/options"):
                data = {"benchmarks": [{"id": "mmlu-pro", "dependencies": ["pyarrow"]}]}
            else:
                data = {"preparations": [preparation]}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        do_GET = do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["arguments"] = ["--url", f"http://127.0.0.1:{server.server_port}", "dataset"]
    monkeypatch.setattr(command, "DEFAULT_STORE", tmp_path / "must-not-create")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not (tmp_path / "must-not-create").exists()


def invoke(service, *arguments):
    return CliRunner().invoke(command.benchmark, [*service["arguments"], *arguments])


def test_remote_prepare_waits_and_returns_worker_manifest(service):
    result = invoke(
        service,
        "prepare",
        "--benchmark",
        "mmlu-pro",
        "--profile",
        "smoke",
        "--seed",
        "42",
        "--limit",
        "3",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == service["manifest"]
    assert "Ctrl-C leaves the service working" in result.stderr
    posts = [request for request in service["requests"] if request[0] == "POST"]
    assert posts == [
        (
            "POST",
            "/api/sr-bench/v1/dataset-preparations",
            {
                "benchmark": "mmlu-pro",
                "profile": "smoke",
                "seed": 42,
                "limit": 3,
            },
        )
    ]
    assert service["reads"] == 2


def test_detached_prepare_is_shared_and_does_not_poll(service):
    result = invoke(service, "prepare", "--benchmark", "simpleqa-verified", "--no-wait")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["preparation"]["id"] == PREPARATION_ID
    assert service["reads"] == 0
    listed = invoke(service, "preparations")
    assert json.loads(listed.stdout)["preparations"][0]["id"] == PREPARATION_ID
    inspected = invoke(service, "preparations", PREPARATION_ID)
    assert json.loads(inspected.stdout)["preparation"]["status"] == "running"
    options = invoke(service, "options")
    assert json.loads(options.stdout)["benchmarks"][0]["id"] == "mmlu-pro"


@pytest.mark.parametrize("failed_post", [False, True])
def test_failure_is_clear_without_resubmission_or_local_fallback(service, failed_post):
    if failed_post:
        service["post_status"] = 503
    else:
        service["status"] = "failed"
    result = invoke(service, "prepare", "--benchmark", "gpqa-diamond")
    assert result.exit_code == 1
    assert (
        "Preparation service is unavailable" if failed_post else "Source access denied"
    ) in result.output
    assert sum(request[0] == "POST" for request in service["requests"]) == 1


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--source-path", "fixture.json"], "require --local"),
        (["--source-partition", "test"], "require --local"),
        (["--revision", "fixture"], "require --local"),
        (["--local"], "cannot be combined with --url"),
    ],
)
def test_remote_options_never_become_implicit_local_import(service, arguments, message):
    result = invoke(service, "prepare", "--benchmark", "mmlu-pro", *arguments)
    assert result.exit_code == 1
    assert message in result.output
    assert not service["requests"]


def test_invalid_preparation_id_is_rejected_before_http(service):
    result = invoke(service, "preparations", "../../datasets")
    assert result.exit_code == 1
    assert "Preparation ID must be" in result.output
    assert not service["requests"]


def test_interrupting_cli_wait_does_not_cancel_or_resubmit_preparation(
    service, monkeypatch
):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(preparations.time, "sleep", interrupt)
    result = invoke(service, "prepare", "--benchmark", "simpleqa-verified")
    assert result.exit_code == 1
    assert "Ctrl-C leaves the service working" in result.stderr
    assert [request[:2] for request in service["requests"] if request[0] == "POST"] == [
        ("POST", "/api/sr-bench/v1/dataset-preparations")
    ]
    listed = invoke(service, "preparations")
    assert json.loads(listed.stdout)["preparations"][0]["id"] == PREPARATION_ID


def test_explicit_local_import_keeps_existing_source_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("SR_BENCH_URL", raising=False)
    monkeypatch.delenv("SR_BENCH_TOKEN_ENV", raising=False)
    source = tmp_path / "source.json"
    source.write_text("[]")
    prepare = Mock(return_value={"id": "local-fixture"})
    monkeypatch.setattr(command.sources, "prepare_dataset", prepare)
    result = CliRunner().invoke(
        command.benchmark,
        [
            "--store",
            str(tmp_path / "store"),
            "--no-autostart",
            "dataset",
            "prepare",
            "--local",
            "--benchmark",
            "mmlu-pro",
            "--source-path",
            str(source),
            "--revision",
            "fixture-v1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": "local-fixture"}
    assert prepare.call_args.kwargs["source_path"] == str(source)
    assert prepare.call_args.kwargs["revision"] == "fixture-v1"
