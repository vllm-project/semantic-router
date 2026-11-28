"""Exercise score-candidate's public sr-bench subprocess contract."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from recipe_lab import cli


class Worker:
    def __init__(self, candidate):
        source_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        self.snapshot = {
            "source_config_hash": source_hash,
            "generated_runtime_hash": "a" * 64,
            "active_runtime_hash": "a" * 64,
        }
        self.base = {
            "mode": "live",
            "profile": "quick",
            "seed": 3,
            "cost_policy": "capability_only",
            "sampling": {"temperature": 0},
            "limits": {"max_output_tokens": 256},
            "case_sha256": "c" * 64,
            "plan_sha256": "b" * 64,
            "cases": [{"id": "guard"}, {"id": "task"}],
            "targets": [
                {"id": "direct", "kind": "single", "model": "model"},
                {
                    "id": "parent",
                    "kind": "mom",
                    "model": "recipe",
                    "config_hash": "f" * 64,
                },
            ],
        }
        self.plan = {
            **self.base,
            "baseline_run_id": "baseline-id",
            "plan_sha256": "d" * 64,
            "targets": [
                {
                    "id": "candidate",
                    "kind": "mom",
                    "model": "recipe",
                    "capture_recipe": True,
                    "config_hash": "a" * 64,
                }
            ],
        }
        self.base_rows = [self.row(case, "parent", 4) for case in ("guard", "task")]
        self.base_rows.append(self.row("task", "direct", 0.01))
        self.candidate_rows = [
            self.row(case, "candidate", 2) for case in ("guard", "task")
        ]
        self.invocations = []
        self.keys = []
        self.status = "completed"

    @staticmethod
    def row(case, target, elapsed):
        return {
            "case_id": case,
            "target_id": target,
            "correct": True,
            "status": "completed",
            "latency_s": elapsed,
        }

    def __call__(self, argv, **kwargs):
        assert argv[:2] == ["vllm-sr", "benchmark"]
        cmd = argv[2:]
        self.invocations.append(cmd)
        if cmd[0] == "show" and "--results" in cmd:
            rows = self.base_rows if cmd[1] == "baseline-id" else self.candidate_rows
            assert cmd[2:] in (
                ["--results", "--after", "0", "--limit", "500"],
                ["--results", "--after", "7", "--limit", "500"],
            )
            page = cmd[4] == "0" and len(rows) > 1
            data = {
                "results": rows[:1] if page else rows[(1 if cmd[4] == "7" else 0) :],
                "next_cursor": 7 if page else None,
            }
        elif cmd == ["show", "baseline-id"]:
            data = {"id": cmd[1], "manifest": self.base, "status": "completed"}
        elif cmd == ["show", "candidate-id"]:
            data = {"id": cmd[1], "manifest": self.plan, "status": self.status}
        elif cmd == [
            "candidate-plan",
            "baseline-id",
            "--target",
            "candidate",
            "--mode",
            "live",
        ]:
            data = {
                "manifest": self.plan,
                "status": "validated",
                "plan_sha256": "d" * 64,
                "model_requests": 0,
            }
        elif cmd[0] == "run":
            assert (
                cmd[1] == "--manifest"
                and json.loads(Path(cmd[2]).read_bytes()) == self.plan
            )
            assert cmd[3:5] == ["--detach", "--idempotency-key"]
            self.keys.append(cmd[5])
            data = {"id": "candidate-id"}
        elif cmd == ["report", "candidate-id"]:
            data = {
                "run_id": cmd[1],
                "status": "completed",
                "provenance": {
                    "plan_sha256": "d" * 64,
                    "runner": {"recipe_snapshots": {"candidate": self.snapshot}},
                },
            }
        elif cmd == ["compare", "baseline-id", "candidate-id"]:
            data = {
                "baseline_run_id": cmd[1],
                "candidate_run_id": cmd[2],
                "baseline_selected_target_id": "direct",
            }
        else:
            pytest.fail(f"unexpected sr-bench invocation: {cmd}")
        return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")


@pytest.fixture
def experiment(tmp_path, monkeypatch, capsys):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        "version: v0.5\nentrypoints: [{model_names: [recipe], recipe: routes}]\n"
        "recipes: [{name: routes}]\n",
        encoding="utf-8",
    )
    worker = Worker(candidate)
    monkeypatch.setattr(subprocess, "run", worker)
    args = [
        "score-candidate",
        "--workspace",
        str(tmp_path),
        "--candidate",
        str(candidate),
        "--proposer",
        "agent:planner",
        "--baseline-run",
        "baseline-id",
        "--baseline-target",
        "parent",
        "--candidate-target",
        "candidate",
        "--active-runtime-hash",
        "a" * 64,
        "--guard-case",
        "guard",
    ]

    def run(*extra):
        code = cli.main([*args, *extra])
        receipt = json.loads(capsys.readouterr().out)
        saved = json.loads(Path(receipt["receipt"]).read_bytes())
        assert saved == {k: v for k, v in receipt.items() if k != "receipt"}
        return code, receipt

    return run, worker


def test_live_dispatch_named_parent_and_reopen(experiment):
    run, worker = experiment
    code, result = run()
    assert code == 0 and result["verdict"] == "PASS"
    assert result["proposer"] == "agent:planner" and result["recipe"] == "routes"
    assert result["metrics"]["speedup_pct"] == 50
    assert (
        result["baseline_target"] == "parent"
        and result["sr_bench_selected_single"] == "direct"
    )
    assert run()[0] == 0 and len(worker.keys) == 2 and len(set(worker.keys)) == 1
    assert run("--candidate-run", "candidate-id")[0] == 0 and len(worker.keys) == 2


def test_guard_loss_cannot_be_offset_by_win(experiment):
    run, worker = experiment
    worker.base_rows[1]["correct"] = False
    worker.candidate_rows[0]["correct"] = False
    worker.base["cases"].append({"id": "shared"})
    worker.base_rows.append(worker.row("shared", "parent", 4))
    worker.candidate_rows.append(worker.row("shared", "candidate", 2))
    code, result = run()
    assert code == 1 and result["verdict"] == "FAIL"
    assert result["metrics"]["guard_losses"] == ["guard"]


def test_timing_gap_takes_precedence_over_a_guard_failure(experiment):
    run, worker = experiment
    worker.candidate_rows[0]["correct"] = False
    worker.candidate_rows[1].pop("latency_s")
    code, result = run()
    assert code == 2 and result["metrics"]["guard_losses"] == ["guard"]


def test_malformed_yaml_omits_values_from_saved_receipt(experiment, tmp_path):
    run, _ = experiment
    (tmp_path / "candidate.yaml").write_text(
        "providers: [example-secret-token", encoding="utf-8"
    )
    code, result = run()
    assert code == 2 and result["reason"] == "invalid configuration YAML"
    assert "example-secret-token" not in json.dumps(result)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_source",
        "missing_snapshot",
        "wrong_runtime",
        "timing",
        "missing_case",
        "failed_grade",
        "foreign_baseline",
        "nonterminal",
        "recovery",
        "guard",
    ],
)
def test_incomplete_evidence_is_inconclusive(experiment, fault):
    run, worker = experiment
    extras = ["--candidate-run", "candidate-id", "--wait-seconds", "0"]
    if fault == "wrong_source":
        worker.snapshot["source_config_hash"] = "3" * 64
    elif fault == "missing_snapshot":
        worker.snapshot = {}
    elif fault == "wrong_runtime":
        worker.plan["targets"][0]["config_hash"] = "3" * 64
    elif fault == "timing":
        worker.candidate_rows[0]["latency_s"] = None
    elif fault == "missing_case":
        worker.candidate_rows.pop()
    elif fault == "failed_grade":
        worker.candidate_rows[0]["status"] = "failed"
    elif fault == "foreign_baseline":
        worker.plan["baseline_run_id"] = "other-run"
    elif fault == "nonterminal":
        worker.status = "running"
    elif fault == "recovery":
        worker.plan["execution_cells"] = []
    else:
        extras += ["--guard-case", "unplanned"]
    code, result = run(*extras)
    assert code == 2 and result["verdict"] == "INCONCLUSIVE" and result["reason"]
    assert not worker.keys


def test_no_dispatch_if_registered_hash_differs(experiment):
    run, worker = experiment
    worker.plan["targets"][0]["config_hash"] = "9" * 64
    assert run()[0] == 2 and not worker.keys


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "drift",
    [
        "reasoning_effort",
        "chat_template_kwargs",
        "session_mode",
        "max_inference_calls",
        "limits",
        "native_limits",
        "output_policy",
        "sampling",
    ],
)
def test_check_selected_subject_protocol_before_dispatch_and_on_reopen(
    experiment, reuse, drift
):
    run, worker = experiment
    parent = worker.base["targets"][1]
    child = worker.plan["targets"][0]
    if drift == "reasoning_effort":
        parent["request_params"] = {drift: "high"}
        child["request_params"] = {drift: "low"}
    elif drift == "chat_template_kwargs":
        parent["request_params"] = {drift: {"synthetic-private-setting": True}}
        child["request_params"] = {drift: {"synthetic-private-setting": 1}}
    elif drift == "session_mode":
        child[drift] = "session_aware"
    elif drift == "max_inference_calls":
        parent[drift], child[drift] = 4, 1
    elif drift == "output_policy":
        worker.plan[drift] = "native"
        child["native_limits"] = {
            "model": {"context_window": 256, "max_output_tokens": 64}
        }
    elif drift == "native_limits":
        worker.base["output_policy"] = worker.plan["output_policy"] = "native"
        parent[drift] = {"model": {"context_window": 512, "max_output_tokens": 128}}
        child[drift] = {"model": {"context_window": 512, "max_output_tokens": 64}}
    elif drift == "sampling":
        worker.plan["sampling"] = {"temperature": 0.25}
    else:
        worker.plan["limits"] = {"max_output_tokens": 128}
    code, receipt = run(*(["--candidate-run", "candidate-id"] if reuse else []))
    assert code == 2 and receipt["verdict"] == "INCONCLUSIVE"
    assert "different frozen subject protocol" in receipt["reason"]
    assert (
        receipt["baseline_subject_protocol_sha256"]
        != receipt["candidate_subject_protocol_sha256"]
    )
    assert not worker.keys
    if drift in {"reasoning_effort", "chat_template_kwargs"}:
        assert receipt["changed_request_fields"] == [drift]
        assert "synthetic-private-setting" not in json.dumps(receipt)


def test_equal_effective_subject_parameters_ignore_redundant_overrides(experiment):
    run, worker = experiment
    worker.base["targets"][1]["request_params"] = {"temperature": 0.0}
    result, receipt = run()
    assert result == 0
    assert (
        receipt["baseline_subject_protocol_sha256"]
        == receipt["candidate_subject_protocol_sha256"]
    )


def test_equal_observed_time_cannot_advance(experiment):
    run, worker = experiment
    for row in worker.candidate_rows:
        row["latency_s"] = 4
    code, result = run()
    assert code == 1 and result["metrics"]["speedup_pct"] == 0


def test_usage_error_exits_2(capsys):
    # The structured error receipt lands with the harness-grade CLI (PR 5).
    with pytest.raises(SystemExit) as stopped:
        cli.main(["score-candidate"])
    assert stopped.value.code == 2
    assert "--workspace" in capsys.readouterr().err
