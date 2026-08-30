from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from cli.commands.eval import eval
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.worker import main as worker_main
from click.testing import CliRunner


def _manifest_payload() -> dict[str, object]:
    path = files("cli.evaluation").joinpath("golden", "manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_eval_group_keeps_prompt_mode_and_registers_plane_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(eval, ["catalog"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert tuple(row["id"] for row in payload["tracks"]) == TRACK_IDS
    assert {row["id"] for row in payload["targets"]} == {"fixture", "runtime"}


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
        eval, ["report", "golden-replay", "--store", str(store_path)]
    )
    assert rendered.exit_code == 0, rendered.output
    assert json.loads(rendered.output)["run"]["id"] == "golden-replay"

    gated = runner.invoke(
        eval,
        [
            "gate",
            "golden-replay",
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

    exit_code = worker_main(
        [
            "--manifest",
            str(manifest_path),
            "--store",
            str(store.root),
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
    assert manifest_path.read_bytes() == staged_bytes
    assert status_path.read_bytes() == status_bytes


def test_compare_command_rejects_unpaired_workloads(tmp_path: Path) -> None:
    runner = CliRunner()
    store_path = tmp_path / "store"
    baseline = _manifest_payload()
    candidate = _manifest_payload()
    candidate["run_id"] = "different-workload"
    candidate["baseline_run_id"] = "golden-replay"
    candidate["code_revision"] = "sha256:" + "2" * 64
    candidate["sample_limit"] = 2
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
            "golden-replay",
            "--candidate",
            "different-workload",
            "--store",
            str(store_path),
        ],
    )
    assert compared.exit_code != 0
    assert "workload_snapshot_digest" in compared.output
