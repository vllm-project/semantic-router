"""sr-bench adapter tests: subprocess mocking + metric extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from recipe_lab.contract import ContractError
from recipe_lab.srbench import SrbenchCli, SrbenchError, extract_metric, require_metric


def fake_run(returncode: int = 0, stdout: str = "{}", stderr: str = ""):
    completed = mock.MagicMock()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = stderr
    return mock.patch("subprocess.run", return_value=completed)


def test_compare_parses_json_stdout():
    payload = {
        "baseline": {"capability.score": 0.80},
        "candidate": {"capability.score": 0.86},
    }
    with fake_run(stdout=json.dumps(payload)):
        result = SrbenchCli().compare("run-a", "run-b")
    assert result == payload


def test_compare_nonzero_exit_raises():
    with (
        fake_run(returncode=3, stderr="boom"),
        pytest.raises(SrbenchError, match="returned 3"),
    ):
        SrbenchCli().compare("run-a", "run-b")


def test_compare_non_json_is_kept_raw():
    with fake_run(stdout="not json"):
        result = SrbenchCli().compare("run-a", "run-b")
    assert isinstance(result, str) and "not json" in result


def test_extract_metric_recursive():
    payload = {"a": {"b": [{"capability.score": 0.9}]}}
    assert extract_metric(payload, "capability.score") == 0.9
    assert extract_metric(payload, "nope") is None


def test_require_metric_raises_contract_error():
    with pytest.raises(ContractError):
        require_metric({"x": 1}, "capability.score")


def test_report_reads_written_file(tmp_path):
    out = tmp_path / "report.json"
    out.write_text(json.dumps({"quality": {"accuracy": 0.83}}), encoding="utf-8")
    with fake_run():
        result = SrbenchCli().report("run-a", out)
    assert result["quality"]["accuracy"] == 0.83


def _completed(stdout: str):
    completed = mock.MagicMock()
    completed.returncode = 0
    completed.stdout = stdout
    completed.stderr = ""
    return completed


def _sequence(stdouts: list[str]):
    return mock.patch("subprocess.run", side_effect=[_completed(s) for s in stdouts])


def test_preview_run_candidate_plan_arg_shapes(tmp_path):
    """The thin wrappers pass the right sr-bench subcommand and arguments (I)."""
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    doc = json.dumps({"id": "run-x", "status": "completed"})
    submitted = json.dumps({"id": "run-x"})
    # preview: submit --detach, then poll show until terminal
    with _sequence([submitted, doc]) as run_mock:
        SrbenchCli().preview(manifest)
        assert run_mock.call_args_list[0][0][0][2:] == [
            "preview",
            "--manifest",
            str(manifest),
            "--detach",
        ]
        assert run_mock.call_args_list[1][0][0][2:] == ["show", "run-x"]
    # run --manifest: same submit-then-poll shape
    with _sequence([submitted, doc]) as run_mock:
        SrbenchCli().run_manifest(manifest)
        assert run_mock.call_args_list[0][0][0][2:] == [
            "run",
            "--manifest",
            str(manifest),
            "--detach",
        ]
    # candidate-plan: required repeatable --target, --mode, name/hypothesis;
    # the review never dispatches, so the reviewed manifest is then submitted
    # verbatim through `run --manifest --detach` (reviewed_plan_changed).
    reviewed = json.dumps(
        {
            "manifest": {"name": "reviewed", "targets": [{"id": "mom-a"}]},
            "plan_sha256": "p" * 64,
            "status": "validated",
            "model_requests": 0,
        }
    )
    dispatched: list[dict] = []

    def scripted_run(cmd, *args, **kwargs):
        command = list(cmd)
        if command[2:4] == ["run", "--manifest"]:
            # capture the dispatched manifest before the wrapper deletes it
            dispatched.append(json.loads(Path(command[4]).read_text(encoding="utf-8")))
            return _completed(submitted)
        if command[2] == "candidate-plan":
            return _completed(reviewed)
        return _completed(doc)

    with mock.patch("subprocess.run", side_effect=scripted_run) as run_mock:
        SrbenchCli().candidate_plan(
            "run-base",
            targets=["mom-a", "mom-b"],
            mode="live",
            name="trial",
            hypothesis="h",
        )
        assert run_mock.call_args_list[0][0][0][2:] == [
            "candidate-plan",
            "run-base",
            "--target",
            "mom-a",
            "--target",
            "mom-b",
            "--mode",
            "live",
            "--name",
            "trial",
            "--hypothesis",
            "h",
        ]
        second = run_mock.call_args_list[1][0][0][2:]
        assert second[:2] == ["run", "--manifest"] and second[-1] == "--detach"
        assert dispatched == [{"name": "reviewed", "targets": [{"id": "mom-a"}]}]
        assert run_mock.call_args_list[2][0][0][2:] == ["show", "run-x"]
    with fake_run(stdout="{}") as run_mock:
        SrbenchCli().experiment(["attach", "exp-1", "--run", "run-a"])
        assert run_mock.call_args[0][0][2:] == [
            "experiment",
            "attach",
            "exp-1",
            "--run",
            "run-a",
        ]


def test_candidate_plan_requires_targets():
    with pytest.raises(SrbenchError, match="at least one target"):
        SrbenchCli().candidate_plan("run-base", targets=[])


def test_wait_for_run_polls_until_terminal():
    responses = [
        json.dumps({"id": "run-x", "status": "queued"}),
        json.dumps({"id": "run-x", "status": "running"}),
        json.dumps({"id": "run-x", "status": "completed"}),
    ]
    with _sequence(responses) as run_mock, mock.patch("time.sleep") as sleep_mock:
        doc = SrbenchCli().wait_for_run("run-x")
    assert doc["status"] == "completed"
    assert sleep_mock.call_count == 2  # slept between the two non-terminal polls
    assert [c[0][0][2:] for c in run_mock.call_args_list] == [["show", "run-x"]] * 3


def test_wait_for_run_rejects_non_document():
    with (
        _sequence(["not json"]),
        pytest.raises(SrbenchError, match="did not return a run document"),
    ):
        SrbenchCli().wait_for_run("run-x")


def test_wait_for_run_rejects_missing_status():
    with (
        _sequence([json.dumps({"id": "run-x"})]),
        pytest.raises(SrbenchError, match="no status"),
    ):
        SrbenchCli().wait_for_run("run-x")


def test_submit_without_run_id_is_an_error():
    with (
        _sequence([json.dumps({"error": "nope"})]),
        pytest.raises(SrbenchError, match="did not return a run id"),
    ):
        SrbenchCli().run_manifest("m.json")
