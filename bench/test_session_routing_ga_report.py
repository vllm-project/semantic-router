import importlib.util
import json
import sys
from pathlib import Path


def load_report_module():
    path = Path(__file__).with_name("session_routing_ga_report.py")
    spec = importlib.util.spec_from_file_location("session_routing_ga_report", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data) + "\n")
    return path


def complete_inputs(tmp_path: Path) -> dict[str, Path]:
    matrix = write_json(
        tmp_path / "matrix.json",
        {
            "overall": {
                "turns": 1200,
                "switch_reduction_pct": 80.0,
                "cost_reduction_pct": 70.0,
                "quality_delta": -0.02,
                "agentic_tool_loop_switch_violations": 0,
                "agentic_context_portability_violations": 0,
            }
        },
    )
    ablation = write_json(
        tmp_path / "ablation.json",
        {
            "by_policy": [
                {"policy": "single-turn"},
                {"policy": "sticky-session"},
                {"policy": "ACR no tool lock"},
                {
                    "policy": "full-acr",
                    "tool_loop_switch_violations": 0,
                    "context_portability_violations": 0,
                },
            ]
        },
    )
    live = write_json(
        tmp_path / "live.json",
        {
            "rows": [
                {
                    "name": "balanced",
                    "router_requests": 64,
                    "router_success_rate": 1.0,
                    "rps_ratio": 0.92,
                    "overhead_p95_ms": 18.0,
                    "tool_loop_switch_violations": 0,
                    "context_portability_violations": 0,
                    "validation_failures": [],
                }
            ]
        },
    )
    failure = write_json(
        tmp_path / "failure.json",
        {
            "rows": [
                {
                    "name": "tool-loop-recovery",
                    "requests": 40,
                    "success_rate": 0.9,
                    "injected": 4,
                    "sessions_with_errors": 2,
                    "session_recovery_rate_after_error": 1.0,
                    "tool_loop_switch_violations": 0,
                    "context_portability_violations": 0,
                    "validation_failures": [],
                }
            ]
        },
    )
    tasks = write_json(
        tmp_path / "tasks.json",
        {
            "requests": 12,
            "task_instances": 4,
            "success_rate": 1.0,
            "task_success_rate": 1.0,
            "task_score_mean": 1.0,
            "tool_loop_switch_violations": 0,
            "context_portability_violations": 0,
            "missing_router_header_counts": {"x-vsr-selected-model": 0},
            "invalid_router_header_counts": {"x-vsr-selected-confidence": 0},
        },
    )
    cache = write_json(
        tmp_path / "cache.json",
        {
            "router": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "positive",
                "cached_token_field_rate": 1.0,
                "cached_prompt_ratio": 0.2,
            },
            "baseline": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "positive",
                "cached_token_field_rate": 1.0,
                "cached_prompt_ratio": 0.15,
            },
        },
    )
    branch = write_json(tmp_path / "branch.json", {"validation_failures": []})
    return {
        "matrix": matrix,
        "ablation": ablation,
        "live": live,
        "failure": failure,
        "tasks": tasks,
        "cache": cache,
        "branch": branch,
    }


def test_complete_evidence_passes_ga_gate(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    args = report_mod.parse_args(
        [
            "--synthetic-matrix-summary",
            str(inputs["matrix"]),
            "--synthetic-ablation-summary",
            str(inputs["ablation"]),
            "--live-aggregate",
            str(inputs["live"]),
            "--failure-aggregate",
            str(inputs["failure"]),
            "--agent-task-summary",
            str(inputs["tasks"]),
            "--cache-aggregate",
            str(inputs["cache"]),
            "--branch-image-summary",
            str(inputs["branch"]),
        ]
    )

    report = report_mod.generate_report(args)

    assert report["ga_ready"] is True
    assert report["blocker_count"] == 0
    assert {item["status"] for item in report["requirements"]} == {"passed"}


def test_missing_positive_cache_and_branch_image_block_ga(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    cache = write_json(
        tmp_path / "cache-missing.json",
        {
            "router": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "missing",
                "cached_token_field_present": 0,
                "cached_prompt_ratio": 0.0,
            }
        },
    )
    args = report_mod.parse_args(
        [
            "--synthetic-matrix-summary",
            str(inputs["matrix"]),
            "--synthetic-ablation-summary",
            str(inputs["ablation"]),
            "--live-aggregate",
            str(inputs["live"]),
            "--failure-aggregate",
            str(inputs["failure"]),
            "--agent-task-summary",
            str(inputs["tasks"]),
            "--cache-aggregate",
            str(cache),
        ]
    )

    report = report_mod.generate_report(args)
    statuses = {item["id"]: item["status"] for item in report["requirements"]}

    assert report["ga_ready"] is False
    assert statuses["cache_token_reporting"] == "blocked"
    assert statuses["branch_image_amd_validation"] == "missing"


def test_main_writes_report_and_respects_allow_blockers(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = report_mod.main(
        [
            "--synthetic-matrix-summary",
            str(inputs["matrix"]),
            "--synthetic-ablation-summary",
            str(inputs["ablation"]),
            "--cache-aggregate",
            str(inputs["cache"]),
            "--allow-blockers",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "ga-readiness.json").exists()
    assert (output_dir / "ga-readiness.md").exists()
