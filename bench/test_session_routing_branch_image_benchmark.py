import importlib.util
import json
from pathlib import Path

REQUIRED_LONG_HORIZON_TASKS = [
    "multi-file-regression",
    "code-review-followup",
    "research-synthesis",
    "maintainer-handoff",
    "cluster-boundary",
    "session-switch-policy",
    "cache-economics",
    "release-triage",
    "observability-debug",
    "test-fix-iteration",
    "codebase-refactor-planning",
    "research-artifact-review",
    "tool-error-recovery-loop",
    "paper-evidence-audit",
    "multi-agent-delegation",
]


def load_benchmark_module():
    module_path = Path(__file__).with_name("session_routing_branch_image_benchmark.py")
    spec = importlib.util.spec_from_file_location(
        "session_routing_branch_image_benchmark", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def diagnostic_summary(ref="77e6b573", image_tag="pr1989-current-branch-image"):
    return {
        "validation_kind": "branch-image-diagnostic-probe",
        "label": "branch-image-diagnostic",
        "ref": ref,
        "image_tag": image_tag,
        "checks": {
            "chat_completion_ok": True,
            "diagnostic_headers_ok": True,
        },
        "missing_diagnostic_headers": [],
        "invalid_diagnostic_headers": [],
        "validation_failures": [],
    }


def live_aggregate():
    return {
        "rows": [
            {
                "name": "balanced-branch-image",
                "router_requests": 128,
                "success_rate": 1.0,
                "tool_loop_switch_violations": 0,
                "context_portability_violations": 0,
                "validation_failures": [],
            }
        ]
    }


def failure_aggregate():
    return {
        "rows": [
            {
                "name": "tool-loop-recovery",
                "requests": 128,
                "success_rate": 0.9,
                "injected": 16,
                "sessions_with_errors": 4,
                "session_recovery_rate_after_error": 1.0,
                "tool_loop_switch_violations": 0,
                "context_portability_violations": 0,
                "validation_failures": [],
            }
        ]
    }


def complete_agent_task_summary():
    return {
        "requests": 255,
        "success_rate": 1.0,
        "successes": 255,
        "task_count": 15,
        "tasks": 15,
        "task_instances": 45,
        "task_success_rate": 1.0,
        "task_score_mean": 1.0,
        "task_names": REQUIRED_LONG_HORIZON_TASKS,
        "scored_task_names": REQUIRED_LONG_HORIZON_TASKS,
        "phase_counts": {
            "user_turn": 45,
            "tool_loop": 93,
            "provider_state": 42,
            "topic_drift": 21,
            "idle_boundary": 9,
            "final": 45,
        },
        "missing_router_header_counts": {
            "x-vsr-selected-model": 0,
            "x-vsr-selected-decision": 0,
            "x-vsr-replay-id": 0,
            "x-vsr-session-phase": 0,
            "x-vsr-selected-confidence": 0,
            "x-vsr-context-token-count": 0,
        },
        "invalid_router_header_counts": {
            "x-vsr-selected-model": 0,
            "x-vsr-selected-decision": 0,
            "x-vsr-replay-id": 0,
            "x-vsr-session-phase": 0,
            "x-vsr-selected-confidence": 0,
            "x-vsr-context-token-count": 0,
        },
        "tool_loop_switch_violations": 0,
        "context_portability_violations": 0,
        "validation_failures": [],
    }


def complete_args(tmp_path):
    diagnostic = write_json(tmp_path / "diagnostic.json", diagnostic_summary())
    live = write_json(tmp_path / "live.json", live_aggregate())
    failure = write_json(tmp_path / "failure.json", failure_aggregate())
    tasks = write_json(tmp_path / "tasks.json", complete_agent_task_summary())
    return [
        "--diagnostic-summary",
        str(diagnostic),
        "--live-aggregate",
        str(live),
        "--failure-aggregate",
        str(failure),
        "--agent-task-summary",
        str(tasks),
        "--ref",
        "77e6b573",
        "--image-tag",
        "pr1989-current-branch-image",
    ]


def test_full_branch_image_summary_passes_with_complete_evidence(tmp_path):
    benchmark = load_benchmark_module()
    args = benchmark.parse_args(complete_args(tmp_path))

    summary = benchmark.build_summary(args)

    assert summary["validation_kind"] == "full-branch-image-benchmark"
    assert summary["branch_image_benchmark"] is True
    assert summary["checks"]["branch_image_benchmark_ok"] is True
    assert summary["validation_failures"] == []


def test_full_branch_image_summary_blocks_mounted_binary_and_stale_tasks(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    arg_values[arg_values.index("pr1989-current-branch-image")] = (
        "pr1989-current-mounted-binary"
    )
    stale_tasks = complete_agent_task_summary()
    stale_tasks.update(
        {
            "requests": 96,
            "task_count": 6,
            "task_instances": 18,
            "missing_router_header_counts": {
                "x-vsr-selected-model": 0,
                "x-vsr-selected-decision": 0,
                "x-vsr-replay-id": 0,
                "x-vsr-session-phase": 96,
                "x-vsr-selected-confidence": 96,
                "x-vsr-context-token-count": 96,
            },
            "task_names": REQUIRED_LONG_HORIZON_TASKS[:6],
            "scored_task_names": REQUIRED_LONG_HORIZON_TASKS[:6],
        }
    )
    write_json(tmp_path / "tasks.json", stale_tasks)
    write_json(
        tmp_path / "diagnostic.json",
        diagnostic_summary(image_tag="pr1989-current-mounted-binary"),
    )
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["branch_image_benchmark_ok"] is False
    assert (
        "mounted-binary evidence cannot satisfy branch-image benchmark"
        in summary["validation_failures"]
    )
    assert "agent task requests 96.0 < 255" in summary["validation_failures"]
    assert (
        "agent task missing router headers: "
        "{'x-vsr-session-phase': 96, 'x-vsr-selected-confidence': 96, "
        "'x-vsr-context-token-count': 96}" in summary["validation_failures"]
    )


def test_full_branch_image_summary_writes_outputs(tmp_path):
    benchmark = load_benchmark_module()
    output_dir = tmp_path / "out"
    args = benchmark.parse_args(
        [*complete_args(tmp_path), "--output-dir", str(output_dir)]
    )
    summary = benchmark.build_summary(args)

    benchmark.write_outputs(summary, output_dir)

    assert (output_dir / "summary.json").exists()
    assert "full-branch-image-benchmark" in (output_dir / "summary.md").read_text()
