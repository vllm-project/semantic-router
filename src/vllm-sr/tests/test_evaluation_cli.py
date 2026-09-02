from __future__ import annotations

import json
from pathlib import Path

from cli.commands.eval import eval
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contracts import RunManifest
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.worker import main as worker_main
from click.testing import CliRunner

_GOLDEN_RUN_ID = "00000000-0000-4000-8000-000000000001"
_CANDIDATE_RUN_ID = "00000000-0000-4000-8000-000000000003"


def _manifest_payload() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "evaluation" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_eval_group_keeps_prompt_mode_and_registers_plane_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(eval, ["catalog"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert tuple(row["id"] for row in payload["tracks"]) == TRACK_IDS
    assert {row["id"] for row in payload["targets"]} == {
        "benchmark-source",
        "fixture",
    }


def test_validate_run_report_and_gate_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    store_path = tmp_path / "store"

    validated = runner.invoke(eval, ["validate", "--manifest", str(manifest_path)])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"] is True

    executed = runner.invoke(
        eval,
        ["run", "--manifest", str(manifest_path), "--store", str(store_path)],
    )
    assert executed.exit_code == 0, executed.output
    assert json.loads(executed.output)["run"]["status"] == "completed"

    rendered = runner.invoke(
        eval, ["report", _GOLDEN_RUN_ID, "--store", str(store_path)]
    )
    assert rendered.exit_code == 0, rendered.output
    assert json.loads(rendered.output)["run"]["id"] == _GOLDEN_RUN_ID

    gated = runner.invoke(
        eval,
        [
            "gate",
            _GOLDEN_RUN_ID,
            "--store",
            str(store_path),
            "--allow-unavailable",
        ],
    )
    assert gated.exit_code == 0, gated.output
    assert json.loads(gated.output)["verdict"] == "unavailable"


def test_worker_emits_strict_lines_without_overwriting_server_control_state(
    tmp_path: Path, capfd: object
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    payload = _manifest_payload()
    run_id = str(payload["run_id"])
    run_dir = store.runs / run_id
    run_dir.mkdir(mode=0o700)
    staged_bytes = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    manifest_path = run_dir / "run-manifest.json"
    manifest_path.write_bytes(staged_bytes)
    status_path = run_dir / "status.json"
    status_bytes = b'{"owner":"go","status":"running"}\n'
    status_path.write_bytes(status_bytes)
    suite_store = NormalizedSuiteStore(tmp_path / "suites")

    exit_code = worker_main(
        [
            "--manifest",
            str(manifest_path),
            "--store",
            str(store.root),
            "--suite-store",
            str(suite_store.root),
            "--events-stdout",
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 0, captured.err
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["type"] == "snapshot"
    assert events[-1]["type"] == "completed"
    assert all("run_id" not in event and "timestamp" not in event for event in events)
    assert all(
        set(event) <= {"type", "message", "track_id", "progress", "payload"}
        for event in events
    )
    assert all(
        event.get("payload") is None
        for event in events
        if event["type"] not in {"track", "completed"}
    )
    assert all(
        set(event["payload"]) == {"record_count"}
        for event in events
        if event["type"] == "track"
    )
    assert set(events[-1]["payload"]) == {"verdict"}
    assert manifest_path.read_bytes() == staged_bytes
    assert status_path.read_bytes() == status_bytes
    assert not (run_dir / "events.jsonl").exists()


def test_worker_requires_the_exact_server_staged_manifest(
    tmp_path: Path, capfd: object
) -> None:
    payload = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    suite_store = NormalizedSuiteStore(tmp_path / "suites")

    exit_code = worker_main(
        [
            "--manifest",
            str(manifest_path),
            "--store",
            str(store.root),
            "--suite-store",
            str(suite_store.root),
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 1
    assert "requires a valid server-staged run-manifest.json" in captured.err
    assert not (store.runs / str(payload["run_id"]) / "run-manifest.json").exists()


def test_compare_command_rejects_unpaired_workloads(tmp_path: Path) -> None:
    runner = CliRunner()
    store_path = tmp_path / "store"
    baseline = _manifest_payload()
    candidate = (
        RunManifest.model_validate(_manifest_payload())
        .with_semantic_updates(
            run_id=_CANDIDATE_RUN_ID,
            baseline_run_id=_GOLDEN_RUN_ID,
            code_revision="sha256:" + "2" * 64,
            sample_limit=2,
        )
        .model_dump(mode="json")
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    for manifest_path in (baseline_path, candidate_path):
        result = runner.invoke(
            eval,
            ["run", "--manifest", str(manifest_path), "--store", str(store_path)],
        )
        assert result.exit_code == 0, result.output

    compared = runner.invoke(
        eval,
        [
            "compare",
            "--baseline",
            _GOLDEN_RUN_ID,
            "--candidate",
            _CANDIDATE_RUN_ID,
            "--store",
            str(store_path),
        ],
    )
    assert compared.exit_code != 0
    assert "workload_snapshot_digest" in compared.output
