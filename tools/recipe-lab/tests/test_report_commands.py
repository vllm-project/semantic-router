"""Report delivery commands exercised with this stack's own receipts."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from recipe_lab import cli, loop, report_delivery, srbench
from recipe_lab.loop_evidence import digest

CONFIG = "4" * 64


def _awaiting_receipt(run_id):
    return {
        "schema": "recipe-lab-sr-bench-run-v1",
        "verdict": "INCONCLUSIVE",
        "baseline_run": run_id,
        "baseline_target": "parent",
        "candidate_target": "draft",
        "candidate_sha256": CONFIG,
        "case_sha256": "c" * 64,
        "reason": "candidate submitted; awaiting evidence",
    }


def test_explicit_evidence_order_and_safe_output(tmp_path, capsys):
    inputs = []
    for label in ("parent", "another-parent"):
        path = tmp_path / (label + ".json")
        path.write_text(json.dumps(_awaiting_receipt(label)), encoding="utf-8")
        inputs.extend(("--evidence", str(path)))
    dest = tmp_path / "report.md"
    args = [
        "report",
        *inputs,
        "--out",
        str(dest),
        "--background",
        "Review both attempts.",
    ]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["evidence_count"] == 2
    before = dest.read_bytes()
    assert b"Review both attempts." in before
    with pytest.raises(SystemExit) as error:
        cli.main(args[:-1] + ["Revised background."])
    assert error.value.code == 2 and dest.read_bytes() == before
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema":"unknown"}', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "report",
                "--evidence",
                str(invalid),
                "--out",
                str(tmp_path / "invalid.md"),
            ]
        )
    assert error.value.code == 2 and not (tmp_path / "invalid.md").exists()


def test_concurrent_studies_publish_separate_complete_snapshots(tmp_path):
    def publish(run):
        return report_delivery.snapshot(tmp_path, _awaiting_receipt(run))

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(publish, ["one", "two", "one", "two"]))
    assert reports[0] != reports[1] and reports[:2] == reports[2:]
    for path in map(Path, reports):
        assert {p.name for p in path.parent.iterdir()} == {"evidence.json", "report.md"}
        record = json.loads(path.with_name("evidence.json").read_bytes())
        assert path.read_text(encoding="utf-8") == report_delivery.render_report(
            [record]
        )


def _graded_fixtures(tmp_path):
    """One real config file plus two distinct physical episodes."""
    run = tmp_path / "run"
    config = run / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "version: v0.5\n"
        "entrypoints:\n"
        "  - model_names: [public]\n"
        "    recipe: main\n"
        "recipes:\n"
        "  - name: sibling\n"
        "  - name: main\n"
        "providers:\n"
        "  models:\n"
        "    - name: A\n",
        encoding="utf-8",
    )

    def episode(name):
        folder = run / "dsh" / "runs" / "task" / name / "main"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(
            json.dumps({"termination": "natural_completion", "run": name}),
            encoding="utf-8",
        )
        return {
            str(folder / "result.json"): digest((folder / "result.json").read_bytes())
        }

    def measurement(evidence):
        return dict(
            kind="measurement",
            split="dev",
            arm="fixed",
            recipe="main",
            public_model="public",
            strong_model="A",
            config=str(config),
            config_sha256=digest(config.read_bytes()),
            hypothesis="graded import",
            evidence=evidence,
            cases={},
        )

    return measurement


def test_loop_comparison_keeps_signed_record_and_delivers_report(tmp_path, capsys):
    ledger = loop.Ledger(tmp_path / "ledger")
    measurement = _graded_fixtures(tmp_path)
    base = ledger.append(measurement(_graded_episode(tmp_path, "fixed", 0)))
    other = ledger.append(measurement(_graded_episode(tmp_path, "draft", 2)))
    record = loop.compare(
        ledger,
        argparse.Namespace(
            baseline=base["record_id"],
            candidate=other["record_id"],
            disposition="keep-baseline",
            reason="Await more evidence.",
            workspace=tmp_path,
        ),
    )
    assert ledger.rows()[-1] == record
    message = capsys.readouterr().err.strip()
    assert message.startswith("Recipe Lab report: ")
    report = Path(message.removeprefix("Recipe Lab report: "))
    assert report.exists()
    assert json.loads(report.with_name("evidence.json").read_bytes()) == record


def _graded_episode(tmp_path, name, marker):
    folder = tmp_path / "run" / "dsh" / "runs" / "task" / name / "main"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(
        json.dumps({"termination": "natural_completion", "run": name, "n": marker}),
        encoding="utf-8",
    )
    return {str(folder / "result.json"): digest((folder / "result.json").read_bytes())}


def test_local_report_delivery_never_changes_the_benchmark_receipt(
    tmp_path, monkeypatch
):
    raw = _awaiting_receipt("test")
    first = raw.copy()
    srbench.save_result(tmp_path / "successful", first)

    def failed(*args):
        raise OSError("read-only reports")

    monkeypatch.setattr(srbench, "snapshot", failed)
    second = raw.copy()
    srbench.save_result(tmp_path / "unwritable", second)
    assert "report" in first and second["report_error"] == "read-only reports"
    assert Path(first["receipt"]).name == Path(second["receipt"]).name
    assert json.loads(Path(second["receipt"]).read_bytes()) == raw
