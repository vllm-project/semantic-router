"""sr-bench public CLI contract (the retired evaluation commands are absent)."""

import json

from cli.commands.benchmark import benchmark
from click.testing import CliRunner


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
        "serve",
        "preview",
        "target",
        "replay",
        "regrade",
        "recover-plan",
        "recover",
        "export",
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
