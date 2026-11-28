"""Check the public development-to-proposal command with saved native call shapes."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recipe_lab import bench_depth, cli
from recipe_lab.loop_evidence import digest
from tests.fixtures import DEPTH_FAST, DEPTH_PUBLIC, DEPTH_STRONG, depth_doc


@pytest.fixture
def study(tmp_path, monkeypatch):
    source = tmp_path / "speed.yaml"
    source.write_text(yaml.safe_dump(depth_doc()), encoding="utf-8")
    runtime = "a" * 64
    manifest = {
        "profile": "quick",
        "mode": "live",
        "cases": [{"id": x} for x in ("lower", "guard")],
        "targets": [
            {
                "id": "parent",
                "model": DEPTH_PUBLIC,
                "kind": "mom",
                "config_hash": runtime,
                "capture_recipe": True,
            },
            {"id": "direct", "model": DEPTH_FAST, "kind": "single"},
        ],
    }
    state = {
        "runtime": runtime,
        "with_overlap": False,
        "same_request": True,
        "fast_solves_guard": False,
    }

    def call(target, case):
        reasoning = target == "parent" and not state.get("already_fast")
        return {
            "id": target + "-" + case,
            "role": "subject",
            "case_id": case,
            "target_id": target,
            "status": "completed",
            "finish_reason": (
                "length" if target == "direct" and case == "guard" else "stop"
            ),
            "model": DEPTH_STRONG if reasoning else DEPTH_FAST,
            "decision": "reasoning" if reasoning else "fast",
            "selected_model": DEPTH_STRONG if reasoning else DEPTH_FAST,
            "config_hash": runtime,
            "request": {
                "effective_body": {
                    **(state.get("lower_request", {}) if case == "lower" else {}),
                    **({"parallel_tool_calls": False} if case == "tool-case" else {}),
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                case
                                if target == "parent" or state["same_request"]
                                else "other"
                            ),
                        }
                    ],
                }
            },
        }

    def result_row(target, case):
        return {
            "id": target + "-" + case,
            "role": "subject",
            "case_id": case,
            "target_id": target,
            "status": "completed",
            "correct": (
                target == "parent" or case == "lower" or state["fast_solves_guard"]
            ),
            "latency_s": 4 if target == "parent" else 2,
        }

    class Service:
        def __init__(self, executable):
            pass

        def run_command(self, argv):
            command, run, *flags = argv
            if command == "report":
                snapshot = {
                    "source_config_hash": digest(source.read_bytes()),
                    "active_runtime_hash": runtime,
                    "generated_runtime_hash": runtime,
                }
                return {
                    "run_id": run,
                    "status": "completed",
                    "provenance": {
                        "runner": {"recipe_snapshots": {"parent": snapshot}}
                    },
                }
            if not flags:
                return {"id": run, "status": "completed", "manifest": manifest}
            if flags[0] == "--calls":
                t = "parent" if flags[2] == "0" else "direct"
                return {
                    "calls": [call(t, c["id"]) for c in manifest["cases"]],
                    "next_cursor": 2 if t == "parent" else None,
                }
            if flags[0] == "--results":
                return {
                    "results": [
                        result_row(t, c["id"])
                        for t in ("parent", "direct")
                        for c in manifest["cases"]
                    ],
                    "next_cursor": None,
                }
            assert flags[0] == "--call-id"
            return call(*flags[1].split("-", 1))

    def preview(args, body):
        return {
            "config_hash": state["runtime"],
            "requested_model": DEPTH_PUBLIC,
            "selection_status": "selected",
            "selected_model": DEPTH_STRONG,
            "decision_result": {"decision_name": "reasoning"},
            "signal_values": {
                "systemone:depth": (
                    0.66 if body["messages"][0]["content"] == "lower" else 0.90
                )
            },
            "eval_trace": [
                {"decision_name": name, "matched": match}
                for name, match in (
                    ("reasoning", True),
                    ("fast", True),
                    ("tools", state["with_overlap"]),
                )
            ],
        }

    monkeypatch.setattr(bench_depth, "SrbenchCli", Service)
    monkeypatch.setattr(bench_depth, "preview", preview)
    dest = tmp_path / "candidate.yaml"
    args = [
        "suggest-depth",
        "--workspace",
        str(tmp_path),
        "--config",
        str(source),
        "--out",
        str(dest),
        "--public-model",
        DEPTH_PUBLIC,
        "--fast",
        DEPTH_FAST,
        "--strong",
        DEPTH_STRONG,
        "--parent-target",
        "parent",
        "--fast-target",
        "direct",
        "--dev-run",
        "dev-1",
        "--endpoint",
        "test",
    ]
    return args, state, manifest, dest


def test_proposal_from_server_rows_validates_exact_edit(study, capsys):
    args, _, _, dest = study
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["status"], result["before"], result["after"]) == (
        "proposed",
        0.6,
        0.78,
    )
    assert digest(dest.read_bytes()) == result["candidate_sha256"]
    assert {r["role"] for r in result["observations"]} == {"move", "protect"}
    guard = next(r for r in result["observations"] if r["role"] == "protect")
    assert (
        result["output_policy"] == "bounded" and guard["fast_finish_reason"] == "length"
    )
    assert Path(result["receipt"]).is_file()


def test_fast_parent_explains_why_the_cutoff_cannot_help(study, capsys):
    args, state, _, dest = study
    state["already_fast"] = True
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "keep-parent" and not dest.exists()
    assert all(
        r["decision"] == "fast"
        and r["selected"] == DEPTH_FAST
        and "strong reasoning lane" in r["reason"]
        for r in result["observations"]
    )


def test_unrepresentable_opening_call_preserves_other_observations(study, capsys):
    args, _, manifest, dest = study
    manifest["cases"].append({"id": "tool-case"})
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "proposed" and dest.is_file()
    skipped = next(r for r in result["observations"] if r["case"] == "tool-case")
    assert skipped["role"] == "skip" and skipped["unsupported_fields"] == [
        "parallel_tool_calls"
    ]


@pytest.mark.parametrize(
    "extra,reason",
    [
        (
            {"tools": [{"type": "function", "function": {"name": "read_status"}}]},
            "tools capability",
        ),
        ({"max_completion_tokens": 12000}, "exceeds_output_limit"),
    ],
)
def test_successful_direct_backend_must_also_be_eligible_in_sr(
    study, capsys, extra, reason
):
    args, state, _, dest = study
    state["lower_request"] = extra
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "keep-parent" and not dest.exists()
    lower = next(r for r in result["observations"] if r["case"] == "lower")
    assert lower["role"] == "skip" and reason in lower["reason"]


@pytest.mark.parametrize(
    "change",
    ["with_overlap", "same_request", "fast_solves_guard", "runtime", "holdout"],
)
def test_no_publication_without_separating_development_evidence(study, capsys, change):
    args, state, manifest, dest = study
    if change == "holdout":
        manifest["cases"][0]["metadata"] = {"split": "holdout"}
    elif change == "runtime":
        state[change] = "b" * 64
    else:
        state[change] = not state[change]
    invalid = change in ("runtime", "holdout")
    assert cli.main(args) == (2 if invalid else 0)
    assert json.loads(capsys.readouterr().out)["status"] == (
        "inconclusive" if invalid else "keep-parent"
    )
    assert not dest.exists()
