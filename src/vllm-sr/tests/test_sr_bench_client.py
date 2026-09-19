"""Shared-worker connection, credential isolation, and first-start boundaries."""

import hashlib
import importlib
import json
from unittest.mock import Mock

import pytest
import requests
from cli.runtime_stack import resolve_runtime_stack
from cli.sr_bench import VERSION, service
from cli.sr_bench.client import Client
from click.testing import CliRunner

command = importlib.import_module("cli.commands.benchmark")
client_module = importlib.import_module("cli.sr_bench.client")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (
        "SR_BENCH_URL",
        "SR_BENCH_STORE",
        "SR_BENCH_TOKEN",
        "SR_BENCH_TOKEN_ENV",
        "VLLM_SR_STACK_NAME",
        "VLLM_SR_PORT_OFFSET",
        "VLLM_SR_STATE_ROOT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_custom_token_reference_is_shared_by_client_and_server(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_BENCH_TOKEN_ENV", "BENCH_TEST_TOKEN")
    monkeypatch.setenv("BENCH_TEST_TOKEN", "custom-test-token")
    monkeypatch.setenv("SR_BENCH_TOKEN", "unrelated-default-token")
    assert Client(store=tmp_path).headers == {
        "Authorization": "Bearer custom-test-token"
    }
    server = Mock()
    server.engine.threads = {}
    factory = Mock(return_value=server)
    monkeypatch.setattr(service, "Server", factory)
    monkeypatch.setattr(service, "Store", lambda path: path)
    monkeypatch.setattr(service.signal, "signal", Mock())
    service.serve(tmp_path, host="0.0.0.0")
    assert factory.call_args.args[2] == "custom-test-token"
    server.serve_forever.assert_called_once()


def test_missing_custom_token_never_falls_back_to_unrelated_default(monkeypatch):
    monkeypatch.setenv("SR_BENCH_TOKEN_ENV", "ABSENT_TEST_TOKEN")
    monkeypatch.delenv("ABSENT_TEST_TOKEN", raising=False)
    monkeypatch.setenv("SR_BENCH_TOKEN", "unrelated-default-token")
    with pytest.raises(ValueError, match="missing: ABSENT_TEST_TOKEN"):
        Client()


@pytest.mark.parametrize("reference", ["HOME", "SR_BENCH_URL", "x; echo secret"])
def test_unsafe_token_reference_is_rejected(reference, monkeypatch):
    monkeypatch.setenv("SR_BENCH_TOKEN_ENV", reference)
    result = CliRunner().invoke(command.benchmark, ["catalog"])
    assert result.exit_code == 1
    assert "safe environment variable" in result.output


@pytest.mark.parametrize(
    "marker",
    [
        "service.json",
        "service.lock",
        "service.log",
        "journal.sqlite3",
        "service-autostart.json",
    ],
)
def test_unavailable_initialized_store_never_restarts(tmp_path, marker, monkeypatch):
    (tmp_path / marker).write_text("saved evidence")
    client = Client(store=tmp_path)
    monkeypatch.setattr(client, "ready", lambda: False)
    launch = Mock()
    monkeypatch.setattr(client_module.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="automatic restart is disabled"):
        client.ensure()
    launch.assert_not_called()
    assert (tmp_path / marker).read_text() == "saved evidence"


def test_first_start_is_recorded_once_and_passes_custom_token_privately(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SR_BENCH_TOKEN_ENV", "BENCH_TEST_TOKEN")
    monkeypatch.setenv("BENCH_TEST_TOKEN", "private-test-token")
    client = Client(store=tmp_path)
    monkeypatch.setattr(client, "ready", Mock(side_effect=[False, True]))
    launch = Mock()
    monkeypatch.setattr(client_module.subprocess, "Popen", launch)
    client.ensure()
    launch.assert_called_once()
    assert launch.call_args.kwargs["env"]["SR_BENCH_TOKEN"] == "private-test-token"
    assert "private-test-token" not in str(launch.call_args.args)
    receipt = tmp_path / "service-autostart.json"
    assert json.loads(receipt.read_text())["store"] == str(tmp_path)
    assert receipt.stat().st_mode & 0o077 == 0
    assert "private-test-token" not in receipt.read_text()
    monkeypatch.setattr(client, "ready", lambda: False)
    with pytest.raises(ValueError, match="automatic restart is disabled"):
        client.ensure()
    launch.assert_called_once()


def test_failed_first_start_retains_receipt_and_cannot_repeat(tmp_path, monkeypatch):
    client = Client(store=tmp_path)
    monkeypatch.setattr(client, "ready", lambda: False)
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    launch = Mock()
    monkeypatch.setattr(client_module.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="did not become ready"):
        client.ensure()
    with pytest.raises(ValueError, match="automatic restart is disabled"):
        client.ensure()
    launch.assert_called_once()


def capture_client(monkeypatch):
    captured = {}

    def request(client, *_args, **_kwargs):
        captured.update(
            url=client.url,
            store=client.store,
            headers=client.headers,
            verify_store=client.verify_store,
            autostart=client.autostart,
        )
        return {"runs": []}

    monkeypatch.setattr(Client, "request", request)
    return captured


def managed_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(tmp_path))
    root = tmp_path / ".sr-bench" / resolve_runtime_stack().stack_name
    store = root / "store"
    store.mkdir(parents=True)
    token = root / "service-token"
    token.write_text("managed-private-token")
    token.chmod(0o600)
    return store


@pytest.mark.parametrize("use_env", [False, True])
def test_explicit_url_never_inherits_managed_store_or_token(
    tmp_path, monkeypatch, use_env
):
    managed_store(tmp_path, monkeypatch)
    fallback = tmp_path / "standalone-store"
    monkeypatch.setattr(command, "DEFAULT_STORE", fallback)
    captured = capture_client(monkeypatch)
    args = ["--url", "http://localhost:8190", "runs"]
    if use_env:
        monkeypatch.setenv("SR_BENCH_URL", "http://localhost:8190")
        args = ["runs"]
    result = CliRunner().invoke(command.benchmark, args)
    assert result.exit_code == 0, result.output
    assert captured["store"] == fallback
    assert captured["headers"] == {}
    assert captured["verify_store"] is False
    assert captured["autostart"] is False


def test_explicit_store_checks_identity_without_implicit_token(tmp_path, monkeypatch):
    store = managed_store(tmp_path, monkeypatch)
    captured = capture_client(monkeypatch)
    result = CliRunner().invoke(
        command.benchmark,
        ["--url", "http://localhost:8190", "--store", str(store), "runs"],
    )
    assert result.exit_code == 0, result.output
    assert captured["store"] == store
    assert captured["headers"] == {}
    assert captured["verify_store"] is True


def test_implicit_local_stack_keeps_managed_identity_and_token(tmp_path, monkeypatch):
    store = managed_store(tmp_path, monkeypatch)
    captured = capture_client(monkeypatch)
    result = CliRunner().invoke(command.benchmark, ["runs"])
    assert result.exit_code == 0, result.output
    assert captured["store"] == store
    assert captured["headers"] == {"Authorization": "Bearer managed-private-token"}
    assert captured["verify_store"] is True
    assert captured["autostart"] is False


def test_store_mismatch_is_not_treated_as_permission_to_spawn(tmp_path, monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {
        "version": VERSION,
        "store_id": hashlib.sha256(b"other-store").hexdigest(),
    }
    monkeypatch.setattr(client_module.requests, "get", Mock(return_value=response))
    launch = Mock()
    monkeypatch.setattr(client_module.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="different sr-bench store"):
        Client(store=tmp_path).ensure()
    launch.assert_not_called()


def test_submission_transport_failure_is_never_retried(tmp_path, monkeypatch):
    client = Client(store=tmp_path)
    monkeypatch.setattr(client, "ensure", lambda: None)
    request = Mock(side_effect=requests.Timeout())
    monkeypatch.setattr(client_module.requests, "request", request)
    with pytest.raises(ValueError, match="requests are never retried"):
        client.request("POST", "/runs", {"manifest": {}})
    request.assert_called_once()
