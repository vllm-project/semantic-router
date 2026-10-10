"""sr-bench public CLI contract (the retired evaluation commands are absent)."""

import json
import threading
import time

from click.testing import CliRunner
from test_sr_bench_replay import manifest, record

from cli.commands.benchmark import benchmark
from cli.sr_bench.contracts import plan
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2))
    return path


def _frozen_document():
    return plan(manifest())


def _register_command(store_dir, source):
    return [
        "--no-autostart",
        "--url",
        "http://127.0.0.1:1",
        "--store",
        str(store_dir),
        "target",
        "register",
        "--file",
        str(source),
    ]


def _serve(store, monkeypatch):
    server = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setenv("SR_BENCH_TOKEN", "fixture-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_command(server, store_dir, document, *extra):
    return [
        "--no-autostart",
        "--url",
        f"http://127.0.0.1:{server.server_port}",
        "--store",
        str(store_dir),
        "run",
        "--manifest",
        str(document),
        *extra,
    ]


def test_catalog_and_clean_command_surface():
    runner = CliRunner()
    result = runner.invoke(benchmark, ["catalog"])
    assert result.exit_code == 0, result.output
    catalog = json.loads(result.output)
    assert catalog["version"] == "sr-bench-1.0"
    assert len(catalog["benchmarks"]) == 9
    assert "tau3" in {b["id"] for b in catalog["benchmarks"]}
    assert "tau2" not in {b["id"] for b in catalog["benchmarks"]}
    assert set(benchmark.commands) == {
        "catalog",
        "setup",
        "dataset",
        "plan",
        "run",
        "runs",
        "show",
        "cancel",
        "report",
        "compare",
        "comparison-options",
        "serve",
        "preview",
        "target",
        "replay",
        "replay-options",
        "regrade",
        "reconcile-usage",
        "recover-plan",
        "recover",
        "export",
        "experiment",
        "candidate-plan",
    }
    assert set(benchmark.commands["experiment"].commands) == {
        "create",
        "list",
        "show",
        "attach",
        "delete",
    }
    assert runner.invoke(benchmark, ["intelligence", "--help"]).exit_code == 2


def test_plan_freezes_without_service_or_inference(tmp_path):
    manifest = {
        "version": "sr-bench-1.0",
        "cost_policy": "capability_only",
        "targets": [
            {
                "id": "single",
                "kind": "single",
                "model": "model",
                "base_url": "http://127.0.0.1:1/v1",
            }
        ],
        "cases": [
            {
                "id": "q1",
                "benchmark": "mmlu-pro",
                "messages": [{"role": "user", "content": "Return A"}],
                "answer": "A",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "frozen.json"
    result = CliRunner().invoke(
        benchmark, ["plan", "--manifest", str(path), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    frozen = json.loads(output.read_text())
    assert frozen["plan_sha256"] == json.loads(result.output)["plan_sha256"]
    assert len(frozen["case_sha256"]) == 64
    assert frozen["limits"]["total_timeout_s"] < 3600
    manifest["cases"].append({**manifest["cases"][0], "id": "q2"})
    manifest["execution_cells"] = [{"case_id": "q1", "target_id": "single"}]
    path.write_text(json.dumps(manifest))
    selected = CliRunner().invoke(benchmark, ["plan", "--manifest", str(path)])
    assert selected.exit_code == 0, selected.output
    assert json.loads(selected.output)["total"] == 1


def test_cli_candidate_plan_reuses_failed_protocol_without_dispatch(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    baseline = record(store)
    store.status(baseline["id"], "failed")
    target = {
        **manifest(True)["targets"][0],
        "config_hash": "a" * 64,
        "max_inference_calls": 1,
    }
    (tmp_path / "targets.json").write_text(json.dumps([target]))
    server = Server(("127.0.0.1", 0), store, "fixture-token")
    monkeypatch.setenv("SR_BENCH_TOKEN", "fixture-token")

    def no_dispatch(*_args, **_kwargs):
        raise AssertionError("Plan cannot dispatch")

    monkeypatch.setattr(server.engine, "start", no_dispatch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    original = list(store.db.iterdump())
    try:
        response = CliRunner().invoke(
            benchmark,
            [
                "--no-autostart",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--store",
                str(tmp_path),
                "candidate-plan",
                baseline["id"],
                "--target",
                target["id"],
            ],
        )
        assert response.exit_code == 0, response.output
        result = json.loads(response.output)
        assert result["model_requests"] == 0
        assert result["manifest"]["baseline_run_id"] == baseline["id"]
        assert result["manifest"]["case_sha256"] == baseline["manifest"]["case_sha256"]
        assert store.get(baseline["id"])["status"] == "failed"
        assert list(store.db.iterdump()) == original
    finally:
        server.shutdown()
        server.server_close()


VALID_TARGET = {
    "id": "single",
    "kind": "single",
    "model": "model",
    "base_url": "http://127.0.0.1:1/v1",
}


def test_target_register_replaces_registry_and_rereads_intact(tmp_path):
    source = _write_json(tmp_path / "registry.json", [VALID_TARGET])
    result = CliRunner().invoke(benchmark, _register_command(tmp_path, source))
    assert result.exit_code == 0, result.output
    registered = json.loads(result.output)
    assert registered["registered"] == 1
    assert [entry["id"] for entry in registered["targets"]] == ["single"]
    stored = tmp_path / "targets.json"
    assert json.loads(stored.read_text()) == registered["targets"]


def test_target_register_rejects_malformed_registry_without_writing(tmp_path):
    source = _write_json(tmp_path / "broken.json", [{"id": "broken"}])
    result = CliRunner().invoke(benchmark, _register_command(tmp_path, source))
    assert result.exit_code != 0, result.output
    assert not (tmp_path / "targets.json").exists()
    assert list(tmp_path.glob("targets-*.tmp")) == []


def test_target_register_stores_the_registry_as_0600(tmp_path):
    source = _write_json(tmp_path / "registry.json", [VALID_TARGET])
    result = CliRunner().invoke(benchmark, _register_command(tmp_path, source))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "targets.json").stat().st_mode & 0o777 == 0o600


def test_target_register_failure_preserves_the_previous_registry(tmp_path):
    source = _write_json(tmp_path / "registry.json", [VALID_TARGET])
    assert (
        CliRunner().invoke(benchmark, _register_command(tmp_path, source)).exit_code
        == 0
    )
    before = (tmp_path / "targets.json").read_text()
    broken = _write_json(tmp_path / "broken.json", [{"id": "broken"}])
    result = CliRunner().invoke(benchmark, _register_command(tmp_path, broken))
    assert result.exit_code != 0, result.output
    assert (tmp_path / "targets.json").read_text() == before
    assert list(tmp_path.glob("targets-*.tmp")) == []


def test_run_failed_terminal_state_exits_two(tmp_path, monkeypatch):
    store = Store(tmp_path)
    seeded = store.create(_frozen_document(), "local", request_key="run-key")
    store.status(seeded[0]["id"], "failed")
    document = _write_json(
        tmp_path / "manifest.json", store.get(seeded[0]["id"])["manifest"]
    )
    server = _serve(store, monkeypatch)
    try:
        runner = CliRunner()
        result = runner.invoke(
            benchmark,
            _run_command(server, tmp_path, document, "--idempotency-key", "run-key"),
        )
        assert result.exit_code == 2, result.output
        assert "submitted" in result.stderr
        assert json.loads(result.stdout)["run_id"] == seeded[0]["id"]
    finally:
        server.shutdown()
        server.server_close()


def test_run_cancelled_terminal_state_exits_two(tmp_path, monkeypatch):
    store = Store(tmp_path)
    seeded = store.create(_frozen_document(), "local", request_key="cancel-key")
    store.status(seeded[0]["id"], "cancelled")
    document = _write_json(
        tmp_path / "manifest.json", store.get(seeded[0]["id"])["manifest"]
    )
    server = _serve(store, monkeypatch)
    try:
        runner = CliRunner()
        result = runner.invoke(
            benchmark,
            _run_command(server, tmp_path, document, "--idempotency-key", "cancel-key"),
        )
        assert result.exit_code == 2, result.output
        assert json.loads(result.stdout)["run_id"] == seeded[0]["id"]
    finally:
        server.shutdown()
        server.server_close()


def test_run_detached_returns_without_polling(tmp_path, monkeypatch):
    store = Store(tmp_path)
    seeded = store.create(_frozen_document(), "local", request_key="detach-key")
    document = _write_json(
        tmp_path / "manifest.json", store.get(seeded[0]["id"])["manifest"]
    )
    server = _serve(store, monkeypatch)
    # The engine's recovery sweep interrupts non-terminal runs at construction;
    # seed the live state after it so the submission answer is a running run.
    store.status(seeded[0]["id"], "running")
    try:
        runner = CliRunner()
        result = runner.invoke(
            benchmark,
            _run_command(
                server,
                tmp_path,
                document,
                "--idempotency-key",
                "detach-key",
                "--detach",
            ),
        )
        assert result.exit_code == 0, result.output
        assert result.stderr == ""
        submitted = json.loads(result.stdout)
        assert submitted["id"] == seeded[0]["id"]
        assert submitted["status"] == "running"
        assert store.get(seeded[0]["id"])["status"] == "running"
    finally:
        server.shutdown()
        server.server_close()


def test_run_polls_until_the_run_reaches_a_terminal_state(tmp_path, monkeypatch):
    store = Store(tmp_path)
    seeded = store.create(_frozen_document(), "local", request_key="poll-key")
    document = _write_json(
        tmp_path / "manifest.json", store.get(seeded[0]["id"])["manifest"]
    )
    server = _serve(store, monkeypatch)
    store.status(seeded[0]["id"], "queued")

    def advance():
        time.sleep(0.6)
        store.status(seeded[0]["id"], "failed")

    threading.Thread(target=advance, daemon=True).start()
    try:
        started = time.monotonic()
        result = CliRunner().invoke(
            benchmark,
            _run_command(server, tmp_path, document, "--idempotency-key", "poll-key"),
        )
        elapsed = time.monotonic() - started
        assert result.exit_code == 2, result.output
        assert json.loads(result.stdout)["run_id"] == seeded[0]["id"]
        # The loop sleeps before each re-read, so a run that only turns
        # terminal after the submission cannot be answered in under one cycle.
        assert elapsed >= 0.9, elapsed
    finally:
        server.shutdown()
        server.server_close()
