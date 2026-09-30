"""sr-bench public CLI contract (the retired evaluation commands are absent)."""

import json
import threading

from cli.commands.benchmark import benchmark
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from click.testing import CliRunner
from test_sr_bench_replay import manifest, record


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
