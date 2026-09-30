"""Collection preparation reuses frozen data and never dispatches inference."""

import json
import threading
import time
from pathlib import Path

import pytest
import requests
from cli.sr_bench import datasets, preparations
from cli.sr_bench.preparation_runtime import PreparationError
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store


def dataset(root, benchmark, profile="smoke", seed=42, suffix="one"):
    case = {
        "id": benchmark + "-" + suffix,
        "benchmark": benchmark,
        "messages": [
            {"role": "user", "content": "Synthetic arithmetic fixture: one plus one."}
        ],
        "answer": "2",
        "metadata": {"split": "holdout" if profile == "standard" else "dev"},
    }
    return _write_dataset(
        root,
        [case],
        profile,
        seed,
        {benchmark: {"url": "https://example.test/fixture", "revision": suffix}},
    )


def terminal(manager, identifier):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(identifier)
        if job["status"] in {"failed", "completed"}:
            return job
        time.sleep(0.01)
    pytest.fail(f"Collection did not finish: {job}")


def request(*benchmarks, **extra):
    return {"benchmarks": list(benchmarks), "profile": "smoke", **extra}


def no_download(*args):
    pytest.fail("Verified data must not invoke a downloader or installer")


@pytest.mark.parametrize("profile", ["smoke", "quick", "standard"])
def test_reuses_exact_sources_and_returns_verified_composite(tmp_path, profile):
    first = dataset(tmp_path, "mmlu-pro", profile)
    second = dataset(tmp_path, "simpleqa-verified", profile)
    manager = preparations.Preparations(tmp_path, no_download)
    job = terminal(
        manager,
        manager.submit(request("simpleqa-verified", "mmlu-pro", profile=profile))["id"],
    )
    assert job["status"] == "completed"
    assert job["seed"] == 42
    assert job["model_requests"] == 0
    assert job["dataset"]["case_count"] == 2
    assert job["dataset"]["profile"] == profile
    assert [item["source_ids"] for item in job["items"]] == [
        [first["id"]],
        [second["id"]],
    ]
    assert all(
        item["reused"] and item["status"] == "completed" for item in job["items"]
    )
    assert (
        datasets.DatasetReader(tmp_path).detail(job["dataset"]["id"])["case_count"] == 2
    )
    manager.close()


def test_prepares_only_missing_members_with_the_existing_seed(tmp_path):
    old = dataset(tmp_path, "mmlu-pro", seed=73)
    calls = []

    def execute(body, store, progress, stop):
        calls.append(body)
        progress("downloading")
        return dataset(store, body["benchmark"], body["profile"], body["seed"])

    manager = preparations.Preparations(tmp_path, execute)
    job = terminal(
        manager,
        manager.submit(request("mmlu-pro", "simpleqa-verified", "arc-agi-2"))["id"],
    )
    assert job["status"] == "completed"
    assert [body["benchmark"] for body in calls] == ["arc-agi-2", "simpleqa-verified"]
    assert all(body["seed"] == 73 and "limit" not in body for body in calls)
    assert job["dataset"]["seed"] == 73
    assert job["dataset"]["case_count"] == 3
    reused = next(item for item in job["items"] if item["benchmark"] == "mmlu-pro")
    assert reused["source_ids"] == [old["id"]]
    assert reused["reused"]
    manager.close()


@pytest.mark.parametrize(
    "body",
    [
        {"benchmarks": []},
        {"benchmarks": "mmlu-pro"},
        {"benchmarks": ["mmlu-pro", "mmlu-pro"]},
        {"benchmarks": ["unknown"]},
        {"benchmarks": [None]},
        {"benchmarks": ["mmlu-pro"], "benchmark": "mmlu-pro"},
        {"benchmarks": ["mmlu-pro"], "limit": 1},
        {"benchmarks": ["mmlu-pro"], "limit": None},
        {"benchmarks": ["mmlu-pro"], "seed": None},
        {"benchmarks": ["mmlu-pro"], "seed": True},
        {"benchmarks": ["mmlu-pro"], "source_path": "/private/input"},
        {"benchmarks": ["mmlu-pro"], "packages": ["unapproved-package"]},
    ],
)
def test_rejects_ambiguous_or_unapproved_collection_requests(body):
    with pytest.raises(ValueError):
        preparations.validate_request(body)


@pytest.mark.parametrize(
    "kind",
    [
        "source_conflict",
        "seed_conflict",
        "explicit_seed",
        "source_size_limit",
        "scan_budget_exhausted",
    ],
)
def test_selection_blockers_never_trigger_download(tmp_path, monkeypatch, kind):
    first = dataset(tmp_path, "mmlu-pro")
    body = request("mmlu-pro", "simpleqa-verified")
    expected = kind
    if kind == "source_conflict":
        dataset(tmp_path, "mmlu-pro", suffix="different")
    elif kind == "seed_conflict":
        dataset(tmp_path, "arc-agi-2", seed=99)
    elif kind == "explicit_seed":
        body["seed"] = 99
        expected = "seed_conflict"
    else:
        original = datasets.DatasetReader._fingerprint

        def limited(reader, identity, remaining):
            if identity == first["id"]:
                raise datasets.DatasetSizeLimitError(
                    reader._paths(identity)[1],
                    100,
                    1,
                    (
                        "scan_budget_exhausted"
                        if kind == "scan_budget_exhausted"
                        else "source_size_limit"
                    ),
                )
            return original(reader, identity, remaining)

        monkeypatch.setattr(datasets.DatasetReader, "_fingerprint", limited)
    manager = preparations.Preparations(tmp_path, no_download)
    job = terminal(manager, manager.submit(body)["id"])
    assert job["status"] == "failed"
    assert job["error_code"] == expected
    assert any(item.get("error_code") == expected for item in job["items"])
    assert "dataset" not in job
    manager.close()


def test_http_disconnect_deduplication_and_permissions_preserve_service_work(
    tmp_path, monkeypatch
):
    server = Server(("127.0.0.1", 0), Store(tmp_path), "fixture-service-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}{PREFIX}"
    entered, release = threading.Event(), threading.Event()
    calls = []

    def execute(body, store, progress, stop):
        calls.append(body)
        progress("downloading")
        entered.set()
        assert release.wait(3)
        return dataset(store, body["benchmark"], body["profile"], body["seed"])

    monkeypatch.setattr(
        server.engine,
        "start",
        lambda *a, **k: pytest.fail("Preparation cannot start inference"),
    )
    server.preparations.executor = execute
    auth = {
        "Authorization": "Bearer fixture-service-token",
        "X-SR-Bench-Actor-ID": "creator",
        "X-SR-Bench-Actor-Role": "write",
    }
    body = request("mmlu-pro", "simpleqa-verified")
    try:
        assert (
            requests.post(
                url + "/dataset-preparations", json=body, timeout=3
            ).status_code
            == 403
        )
        assert (
            requests.post(
                url + "/dataset-preparations",
                json=body,
                headers={**auth, "X-SR-Bench-Actor-Role": "read"},
                timeout=3,
            ).status_code
            == 403
        )
        with requests.Session() as browser:
            response = browser.post(
                url + "/dataset-preparations", json=body, headers=auth, timeout=3
            )
            assert response.status_code == 202
            identifier = response.json()["preparation"]["id"]
        assert entered.wait(3)
        duplicate = requests.post(
            url + "/dataset-preparations",
            json=request("simpleqa-verified", "mmlu-pro"),
            headers=auth,
            timeout=3,
        )
        assert duplicate.json()["preparation"]["id"] == identifier
        busy = requests.post(
            url + "/dataset-preparations",
            json={"benchmark": "arc-agi-2"},
            headers=auth,
            timeout=3,
        )
        assert busy.status_code == 409
        release.set()
        job = terminal(server.preparations, identifier)
        assert job["status"] == "completed" and job["dataset"]["case_count"] == 2
        assert [item["benchmark"] for item in calls] == [
            "mmlu-pro",
            "simpleqa-verified",
        ]
        assert all(item["seed"] == 20260918 for item in calls)
        assert server.store.list() == []
        restored = requests.get(
            url + "/dataset-preparations/" + identifier,
            headers={**auth, "X-SR-Bench-Actor-Role": "read"},
            timeout=3,
        )
        assert restored.json()["preparation"]["dataset"] == job["dataset"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        server.store.db.close()


def test_existing_global_seed_conflict_blocks_an_entirely_missing_subset(tmp_path):
    dataset(tmp_path, "mmlu-pro", seed=41)
    dataset(tmp_path, "arc-agi-2", seed=42)
    manager = preparations.Preparations(tmp_path, no_download)
    job = terminal(manager, manager.submit(request("hle", "simpleqa-verified"))["id"])
    assert job["status"] == "failed" and job["error_code"] == "seed_conflict"
    assert all(item["error_code"] == "seed_conflict" for item in job["items"])
    assert len(list((tmp_path / "datasets").glob("*/manifest.json"))) == 2
    manager.close()


def test_failed_collection_retry_reuses_completed_members_without_secret_leaks(
    tmp_path,
):
    calls = []

    def execute(body, store, progress, stop):
        calls.append(body["benchmark"])
        if body["benchmark"] == "simpleqa-verified" and len(calls) == 2:
            raise RuntimeError("private-token and private-source")
        return dataset(store, body["benchmark"], body["profile"], body["seed"])

    manager = preparations.Preparations(tmp_path, execute)
    body = request("mmlu-pro", "simpleqa-verified")
    failed = terminal(manager, manager.submit(body)["id"])
    assert failed["status"] == "failed"
    assert failed["items"][0]["status"] == "completed"
    assert failed["items"][1]["error_code"] == "preparation_failed"
    assert "private-" not in json.dumps(failed)
    retry = terminal(manager, manager.submit(body)["id"])
    assert retry["status"] == "completed"
    assert retry["id"] != failed["id"]
    assert calls == ["mmlu-pro", "simpleqa-verified", "simpleqa-verified"]
    assert retry["items"][0]["reused"]
    manager.close()


def test_completed_collection_rechecks_new_inventory_conflicts(tmp_path):
    dataset(tmp_path, "mmlu-pro")
    manager = preparations.Preparations(tmp_path, no_download)
    body = request("mmlu-pro")
    first = terminal(manager, manager.submit(body)["id"])
    assert first["status"] == "completed"
    dataset(tmp_path, "mmlu-pro", suffix="new-revision")
    later = terminal(manager, manager.submit(body)["id"])
    assert later["id"] != first["id"]
    assert later["status"] == "failed" and later["error_code"] == "source_conflict"
    manager.close()


def test_downloader_cannot_return_a_different_profile(tmp_path):
    manager = preparations.Preparations(
        tmp_path,
        lambda body, store, *args: dataset(
            store, body["benchmark"], "quick", body["seed"]
        ),
    )
    job = terminal(manager, manager.submit(request("mmlu-pro"))["id"])
    assert job["status"] == "failed"
    assert job["error_code"] == "prepared_dataset_mismatch"
    assert "dataset" not in job
    manager.close()


def test_composition_revalidates_reused_bytes_after_download(tmp_path):
    original = dataset(tmp_path, "mmlu-pro")

    def execute(body, store, *args):
        result = dataset(store, body["benchmark"], body["profile"], body["seed"])
        with Path(original["path"]).open("a") as stream:
            stream.write("{}\n")
        return result

    manager = preparations.Preparations(tmp_path, execute)
    job = terminal(
        manager, manager.submit(request("mmlu-pro", "simpleqa-verified"))["id"]
    )
    assert job["status"] == "failed" and job["error_code"] == "selection_unverified"
    assert "dataset" not in job
    manager.close()


def test_restart_keeps_completed_items_and_requires_explicit_retry(tmp_path):
    old = dataset(tmp_path, "mmlu-pro")
    manager = preparations.Preparations(tmp_path, no_download)
    body = request("mmlu-pro", "simpleqa-verified")
    identifier = "prep-" + "b" * 32
    job = {
        "id": identifier,
        **body,
        "request": body,
        "status": "running",
        "phase": "downloading",
        "created_at": "2026-01-01T00:00:00Z",
        "items": [
            {
                "benchmark": "mmlu-pro",
                "status": "completed",
                "source_ids": [old["id"]],
                "reused": False,
            },
            {
                "benchmark": "simpleqa-verified",
                "status": "running",
                "phase": "downloading",
                "source_ids": [],
                "reused": False,
            },
        ],
    }
    preparations.write_json(manager.root / (identifier + ".json"), job)
    restored = preparations.Preparations(tmp_path, no_download)
    recovered = restored.get(identifier)
    assert recovered["status"] == "failed"
    assert recovered["items"][0]["status"] == "completed"
    assert recovered["items"][1]["error_code"] == "service_restarted"
    assert restored.thread is None
    restored.executor = lambda body, store, *args: dataset(
        store, body["benchmark"], body["profile"], body["seed"]
    )
    retried = terminal(restored, restored.submit(body)["id"])
    assert retried["status"] == "completed" and retried["items"][0]["reused"]
    restored.close()


def test_service_stop_never_starts_the_next_collection_member(tmp_path):
    entered = threading.Event()
    calls = []

    def execute(body, store, progress, stopping):
        calls.append(body["benchmark"])
        entered.set()
        assert stopping.wait(3)
        raise PreparationError("Service stopped during preparation. Retry explicitly.")

    manager = preparations.Preparations(tmp_path, execute)
    job = manager.submit(request("mmlu-pro", "simpleqa-verified"))
    assert entered.wait(3)
    manager.close()
    assert manager.get(job["id"])["status"] == "failed"
    assert calls == ["mmlu-pro"]
