"""Collection polling stays compact while CLI/API detail preserves frozen evidence."""

import hashlib
import json
import threading

import requests
from cli.commands.benchmark import benchmark
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.store import Store
from click.testing import CliRunner


def _saved_run(store, tmp_path):
    cases = [
        {
            "id": f"case-{i}",
            "benchmark": "mmlu-pro",
            "messages": [{"role": "user", "content": "private-question-marker " * 50}],
            "answer": "A",
            "metadata": {"reference_explanation": "private-reference-marker"},
        }
        for i in range(12)
    ]
    data = tmp_path / "cases.json"
    data.write_text(json.dumps(cases))
    cells = [{"case_id": c["id"], "target_id": "single"} for c in cases]
    frozen = plan(
        {
            "version": "sr-bench-1.0",
            "name": "Collection integration",
            "targets": [
                {
                    "id": "single",
                    "kind": "single",
                    "model": "model",
                    "base_url": "http://127.0.0.1:1/v1",
                    "prices": {
                        "model": {
                            "input": 1,
                            "cached_input": 0.1,
                            "cache_write": 2,
                            "output": 3,
                        }
                    },
                }
            ],
            "dataset": {
                "id": "prepared-data",
                "path": str(data),
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "case_count": len(cases),
                "split": "dev",
                "custom_subset": True,
                "benchmarks": ["mmlu-pro"],
                "selection": {"cases": cases},
                "sources": [{"payload": "private-source-marker" * 100}],
            },
            "execution_cells": cells,
            "recipe_snapshot": {"payload": "private-recipe-marker" * 100},
            "recovery": {
                "parent_run_id": "parent-run",
                "mode": "undispatched",
                "selected_cells": cells,
                "selected_cells_sha256": digest(cells),
                "parent_snapshot": {
                    "progress": {"completed": 2, "total": 14},
                    "known_spend_usd": 0.1,
                    "spend_complete": True,
                },
                "recovery_subset": True,
                "new_attempt_acknowledged": False,
            },
        }
    )
    run, _ = store.create(frozen)
    store.result(run["id"], "case-0", "single", "completed", {"correct": True})
    store.status(run["id"], "failed", "Saved fixture stopped before further dispatch")
    return store.get(run["id"])


def test_collection_omits_questions_and_verbose_payloads_but_preserves_detail(
    tmp_path, monkeypatch
):
    store = Store(tmp_path / "store")
    original = _saved_run(store, tmp_path)
    service = Server(("127.0.0.1", 0), store, "collection-test-token")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{service.server_port}"
    headers = {"Authorization": "Bearer collection-test-token"}
    monkeypatch.setenv("SR_BENCH_TOKEN", "collection-test-token")
    monkeypatch.delenv("SR_BENCH_TOKEN_ENV", raising=False)
    try:
        response = requests.get(origin + PREFIX + "/runs", headers=headers, timeout=2)
        assert response.status_code == 200
        summary = response.json()["runs"][0]
        assert summary["manifest_summary"] is True
        assert summary["progress"] == original["progress"]
        assert summary["id"] == original["id"]
        assert summary["status"] == original["status"]
        assert summary["created_at"] == original["created_at"]
        frozen = original["manifest"]
        compact = summary["manifest"]
        for key in (
            "targets",
            "mode",
            "profile",
            "limits",
            "sampling",
            "plan_sha256",
            "case_sha256",
        ):
            assert compact[key] == frozen[key]
        assert compact["dataset"]["sha256"] == frozen["dataset"]["sha256"]
        assert compact["dataset"]["case_count"] == 12
        assert compact["recovery"]["parent_run_id"] == "parent-run"
        assert compact["recovery"]["parent_snapshot"] == (
            frozen["recovery"]["parent_snapshot"]
        )
        assert compact["recovery"]["selected_cell_count"] == 12
        assert compact["recovery"]["selected_cells_sha256"] == digest(
            frozen["execution_cells"]
        )
        assert "cases" not in compact and "execution_cells" not in compact
        assert "selected_cells" not in compact["recovery"]
        assert "selection" not in compact["dataset"]
        assert "sources" not in compact["dataset"]
        assert "recipe_snapshot" not in compact
        assert "private-" not in response.text
        assert '"answer"' not in response.text
        assert '"messages"' not in response.text
        detail = requests.get(
            origin + PREFIX + "/runs/" + original["id"], headers=headers, timeout=2
        ).json()
        assert detail == original
        assert store.get(original["id"]) == original
        assert store.list() == [original]
        assert store.list(summary=True) == [summary]

        runner = CliRunner()
        arguments = ["--url", origin, "--no-autostart"]
        listed = runner.invoke(benchmark, [*arguments, "runs"])
        assert listed.exit_code == 0, listed.output
        assert json.loads(listed.output) == {"runs": [summary]}
        shown = runner.invoke(benchmark, [*arguments, "show", original["id"]])
        assert shown.exit_code == 0, shown.output
        assert json.loads(shown.output) == original
        assert service.store.calls(original["id"]) == []
    finally:
        service.shutdown()
        service.server_close()
