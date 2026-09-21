"""Shared HTTP preparation ownership, persistence and installer boundaries."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from cli.sr_bench import preparation_runtime as runtime
from cli.sr_bench import preparation_worker as worker
from cli.sr_bench import preparations
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store


def prepared(root):
    return _write_dataset(
        root,
        [
            {
                "id": "synthetic-1",
                "benchmark": "mmlu-pro",
                "prompt": "What is 1 + 1?",
                "answer": "2",
            }
        ],
        "smoke",
        20260918,
        {"mmlu-pro": {"revision": "synthetic", "url": "https://example.test/fixture"}},
        True,
    )


def wait_for(manager, identifier, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = manager.get(identifier)
        if result["status"] == status:
            return result
        time.sleep(0.01)
    pytest.fail(f"Preparation did not reach {status}: {result}")


@pytest.fixture
def live_service(tmp_path):
    server = Server(("127.0.0.1", 0), Store(tmp_path), "private-service-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}{PREFIX}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    server.store.db.close()


def headers(role="write"):
    return {
        "Authorization": "Bearer private-service-token",
        "X-SR-Bench-Actor-ID": "synthetic-user",
        "X-SR-Bench-Actor-Role": role,
    }


def test_http_job_is_shared_deduplicated_and_dataset_becomes_visible(live_service):
    server, url = live_service
    entered, release = threading.Event(), threading.Event()
    calls = []

    def execute(body, store, progress, stopping):
        calls.append(body)
        progress("downloading")
        entered.set()
        assert release.wait(3)
        return prepared(store)

    server.preparations.executor = execute
    body = {"benchmark": "mmlu-pro", "profile": "smoke", "limit": 1}
    result = requests.post(
        url + "/dataset-preparations", json=body, headers=headers(), timeout=3
    )
    assert result.status_code == 202
    identifier = result.json()["preparation"]["id"]
    assert entered.wait(3)
    with ThreadPoolExecutor(max_workers=4) as pool:
        duplicates = list(
            pool.map(
                lambda _: requests.post(
                    url + "/dataset-preparations",
                    json=body,
                    headers=headers(),
                    timeout=3,
                ).json(),
                range(8),
            )
        )
    assert {row["preparation"]["id"] for row in duplicates} == {identifier}
    other = requests.post(
        url + "/dataset-preparations",
        json={"benchmark": "simpleqa-verified"},
        headers=headers(),
        timeout=3,
    )
    assert other.status_code == 409
    visible = requests.get(
        url + "/dataset-preparations", headers=headers("read"), timeout=3
    ).json()
    assert visible["preparations"][0]["phase"] == "downloading"
    release.set()
    completed = wait_for(server.preparations, identifier, "completed")
    datasets = requests.get(
        url + "/datasets", headers=headers("read"), timeout=3
    ).json()["datasets"]
    assert [row["id"] for row in datasets] == [completed["dataset"]["id"]]
    assert server.preparations.submit(body)["id"] == identifier
    assert len(calls) == 1


def test_http_requires_write_and_rejects_package_or_path_injection(live_service):
    _, url = live_service
    path = url + "/dataset-preparations"
    body = {"benchmark": "mmlu-pro", "profile": "smoke"}
    assert requests.post(path, json=body, timeout=3).status_code == 403
    assert (
        requests.post(path, json=body, headers=headers("read"), timeout=3).status_code
        == 403
    )
    for field in ["source_path", "url", "packages", "command"]:
        response = requests.post(
            path, json={**body, field: "arbitrary"}, headers=headers(), timeout=3
        )
        assert response.status_code == 400
    options = requests.get(path + "/options", headers=headers("read"), timeout=3)
    assert options.status_code == 200
    assert len(options.json()["benchmarks"]) == 9


@pytest.mark.parametrize(
    "body",
    [
        {"benchmark": []},
        {"benchmark": "mmlu-pro", "seed": True},
        {"benchmark": "mmlu-pro", "seed": 2**53},
        {"benchmark": "mmlu-pro", "limit": 0},
        {"benchmark": "mmlu-pro", "profile": "smoke", "limit": 15},
        {"benchmark": "mmlu-pro", "profile": []},
    ],
)
def test_preparation_request_validation(body):
    with pytest.raises(ValueError):
        preparations.validate_request(body)


def test_failed_job_can_retry_and_unexpected_error_is_not_exposed(tmp_path):
    def fail(*args):
        raise RuntimeError("secret-token and private source content")

    manager = preparations.Preparations(tmp_path, fail)
    body = {"benchmark": "mmlu-pro"}
    failed = wait_for(manager, manager.submit(body)["id"], "failed")
    assert "secret-token" not in json.dumps(failed)
    manager.executor = lambda body, store, progress, stop: prepared(store)
    retry = manager.submit(body)
    assert retry["id"] != failed["id"]
    wait_for(manager, retry["id"], "completed")
    manager.close()
    restored = preparations.Preparations(tmp_path)
    assert len(restored.list()["preparations"]) == 2


def test_restart_records_interruption_without_automatic_download(tmp_path):
    manager = preparations.Preparations(
        tmp_path, lambda *args: pytest.fail("must not execute")
    )
    identifier = "prep-" + "a" * 32
    job = {
        "id": identifier,
        "status": "running",
        "phase": "downloading",
        "created_at": "2026-01-01T00:00:00Z",
        "request": {"benchmark": "mmlu-pro"},
    }
    preparations.write_json(manager.root / (identifier + ".json"), job)
    restored = preparations.Preparations(tmp_path)
    assert restored.get(identifier)["status"] == "failed"
    assert "restarted" in restored.get(identifier)["error"]


def test_first_journal_failure_does_not_reserve_a_phantom_worker(tmp_path, monkeypatch):
    manager = preparations.Preparations(
        tmp_path, lambda body, store, progress, stop: prepared(store)
    )
    write_json = preparations.write_json
    monkeypatch.setattr(
        preparations,
        "write_json",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        manager.submit({"benchmark": "mmlu-pro"})
    assert manager.list()["preparations"] == []
    monkeypatch.setattr(preparations, "write_json", write_json)
    job = manager.submit({"benchmark": "mmlu-pro"})
    wait_for(manager, job["id"], "completed")
    manager.close()


def test_close_cancels_worker_and_closes_admission(tmp_path):
    entered = threading.Event()

    def execute(body, store, progress, stop):
        entered.set()
        assert stop.wait(3)
        raise runtime.PreparationError("Service stopped during preparation.")

    manager = preparations.Preparations(tmp_path, execute)
    job = manager.submit({"benchmark": "mmlu-pro"})
    assert entered.wait(3)
    manager.close()
    assert manager.get(job["id"])["status"] == "failed"
    with pytest.raises(preparations.PreparationBusyError):
        manager.submit({"benchmark": "mmlu-pro"})


def test_low_disk_fails_before_spawning_a_downloader(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})()
    )
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn")
    )
    with pytest.raises(runtime.PreparationError, match="free worker storage"):
        runtime.execute({}, tmp_path, lambda phase: None, threading.Event())


def test_runtime_shutdown_terminates_its_actual_child_process(tmp_path, monkeypatch):
    original = subprocess.Popen
    spawned, errors = [], []
    started = tmp_path / "started"
    stop = threading.Event()

    def spawn(command, **kwargs):
        code = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('started'); time.sleep(30)"
        process = original([sys.executable, "-c", code, str(started)], **kwargs)
        spawned.append(process)
        return process

    def run():
        try:
            runtime.execute({}, tmp_path, lambda phase: None, stop)
        except runtime.PreparationError as exc:
            errors.append(str(exc))

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 3
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5)
    assert started.exists()
    assert not thread.is_alive()
    assert spawned[0].poll() is not None
    assert errors and "stopped" in errors[0]


def test_csv_source_needs_no_optional_installer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **k: pytest.fail("no install required")
    )
    worker.ensure_dependencies("simpleqa-verified", tmp_path, lambda phase: None)


def test_dependency_cache_rejects_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: None)
    (tmp_path / "preparation-runtime").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="symlinks"):
        worker.ensure_dependencies("mmlu-pro", tmp_path, lambda phase: None)


def test_installer_bootstraps_pip_without_modifying_current_interpreter(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: None)
    created, commands = [], []

    class Venv:
        def __init__(self, with_pip):
            assert with_pip

        def create(self, path):
            created.append(path)

    monkeypatch.setattr(worker.venv, "EnvBuilder", Venv)
    monkeypatch.setattr(
        worker.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    worker.install_arrow(tmp_path, str(tmp_path / "staged"))
    assert created == [tmp_path / "installer"]
    assert commands[0][0] == str(tmp_path / "installer" / "bin" / "python")
    assert commands[0][-1] == "pyarrow==18.1.0"
    assert "--only-binary=:all:" in commands[0]


def test_worker_failure_is_controlled_not_raw_credentials(tmp_path, monkeypatch):
    def fail(*args):
        raise subprocess.CalledProcessError(1, "pip --token private")

    monkeypatch.setattr(worker, "ensure_dependencies", fail)
    state = tmp_path / "state.json"
    assert worker.run(tmp_path, state, {"benchmark": "mmlu-pro"}) == 1
    assert json.loads(state.read_text()) == {
        "phase": "failed",
        "error_code": "dependencies",
    }
