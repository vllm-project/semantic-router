import importlib.util
import json
import sys
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
    "issue-pr-maintenance-loop",
    "configuration-contract-review",
    "repo-bisect-debug",
    "dependency-upgrade-regression",
    "literature-data-extraction",
]
REQUIRED_LONG_HORIZON_PHASES = [
    "user_turn",
    "tool_loop",
    "provider_state",
    "topic_drift",
    "idle_boundary",
    "final",
]
CACHE_PROBE_METADATA = {
    "probe_kind": "repeated-prefix-cache-token-probe",
    "probe_repeats": 8,
    "stable_prefix_chars": 13000,
    "stable_prefix_sha256": "abc123def4567890",
    "unique_suffix_pattern": "probe_turn_index",
}


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


def complete_agent_task_summary() -> dict:
    return {
        "requests": 345,
        "tasks": len(REQUIRED_LONG_HORIZON_TASKS),
        "task_count": len(REQUIRED_LONG_HORIZON_TASKS),
        "task_names": REQUIRED_LONG_HORIZON_TASKS,
        "scored_task_count": len(REQUIRED_LONG_HORIZON_TASKS),
        "scored_task_names": REQUIRED_LONG_HORIZON_TASKS,
        "task_instances": len(REQUIRED_LONG_HORIZON_TASKS) * 3,
        "success_rate": 1.0,
        "task_success_rate": 1.0,
        "task_score_mean": 1.0,
        "tool_loop_switch_violations": 0,
        "context_portability_violations": 0,
        "phase_counts": dict.fromkeys(REQUIRED_LONG_HORIZON_PHASES, 3),
        "missing_router_header_counts": {
            "x-vsr-selected-model": 0,
            "x-vsr-selected-decision": 0,
            "x-vsr-replay-id": 0,
            "x-vsr-session-phase": 0,
            "x-vsr-selected-confidence": 0,
            "x-vsr-context-token-count": 0,
        },
        "invalid_router_header_counts": {
            "x-vsr-session-phase": 0,
            "x-vsr-selected-confidence": 0,
            "x-vsr-context-token-count": 0,
        },
    }


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
                {"policy": "acr-initial"},
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
        complete_agent_task_summary(),
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
                **CACHE_PROBE_METADATA,
            },
            "baseline": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "positive",
                "cached_token_field_rate": 1.0,
                "cached_prompt_ratio": 0.15,
                **CACHE_PROBE_METADATA,
            },
        },
    )
    branch = write_json(
        tmp_path / "branch.json",
        {
            "validation_kind": "full-branch-image-benchmark",
            "branch_image_benchmark": True,
            "image_tag": "pr1989-current-branch-image",
            "checks": {
                "diagnostic_ok": True,
                "live_matrix_ok": True,
                "failure_recovery_ok": True,
                "agent_task_ok": True,
                "cache_token_probe_ok": True,
                "mounted_binary_absent": True,
                "branch_image_benchmark_ok": True,
            },
            "validation_failures": [],
        },
    )
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
    assert report["blockers"] == []
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
    assert [item["id"] for item in report["blockers"]] == [
        "cache_token_reporting",
        "branch_image_amd_validation",
    ]
    assert statuses["cache_token_reporting"] == "blocked"
    assert statuses["branch_image_amd_validation"] == "missing"


def test_positive_cache_without_probe_metadata_blocks_ga(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    cache = write_json(
        tmp_path / "cache-without-probe-metadata.json",
        {
            "router": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "positive",
                "cached_token_field_rate": 1.0,
                "cached_prompt_ratio": 0.2,
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
            "--branch-image-summary",
            str(inputs["branch"]),
        ]
    )

    report = report_mod.generate_report(args)
    cache_requirement = next(
        item for item in report["requirements"] if item["id"] == "cache_token_reporting"
    )

    assert report["ga_ready"] is False
    assert cache_requirement["status"] == "blocked"
    assert (
        "router: probe_kind missing != repeated-prefix-cache-token-probe"
        in cache_requirement["failures"]
    )


def test_positive_cache_without_direct_backend_baseline_blocks_ga(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    cache = write_json(
        tmp_path / "router-only-cache.json",
        {
            "router": {
                "requests": 8,
                "successes": 8,
                "success_rate": 1.0,
                "cached_token_reporting": "positive",
                "cached_token_field_rate": 1.0,
                "cached_prompt_ratio": 0.2,
                **CACHE_PROBE_METADATA,
            },
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
            "--branch-image-summary",
            str(inputs["branch"]),
        ]
    )

    report = report_mod.generate_report(args)
    cache_requirement = next(
        item for item in report["requirements"] if item["id"] == "cache_token_reporting"
    )

    assert report["ga_ready"] is False
    assert cache_requirement["status"] == "blocked"
    assert (
        "direct backend baseline cache evidence missing; run "
        "cache_token_probe.py with --baseline-base-url" in cache_requirement["failures"]
    )
    assert cache_requirement["metrics"]["baseline_required"] is True
    assert cache_requirement["metrics"]["baseline_present"] is False


def test_stale_agent_task_suite_blocks_ga(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    stale_tasks = complete_agent_task_summary()
    stale_tasks["missing_router_header_counts"].pop("x-vsr-session-phase")
    stale_tasks.update(
        {
            "requests": 96,
            "task_count": 6,
            "tasks": 6,
            "task_instances": 18,
            "task_names": REQUIRED_LONG_HORIZON_TASKS[:6],
            "scored_task_count": 6,
            "scored_task_names": REQUIRED_LONG_HORIZON_TASKS[:6],
            "phase_counts": {
                "user_turn": 18,
                "tool_loop": 36,
                "provider_state": 18,
                "topic_drift": 12,
                "final": 18,
            },
        }
    )
    tasks = write_json(tmp_path / "tasks-stale.json", stale_tasks)
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
            str(tasks),
            "--cache-aggregate",
            str(inputs["cache"]),
            "--branch-image-summary",
            str(inputs["branch"]),
        ]
    )

    report = report_mod.generate_report(args)
    task_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "amd_agent_task_quality_1"
    )

    assert report["ga_ready"] is False
    assert task_requirement["status"] == "blocked"
    assert "requests 96.0 < 345" in task_requirement["failures"]
    assert "task_count 6.0 < 20" in task_requirement["failures"]
    assert "task_instances 18.0 < 60" in task_requirement["failures"]
    assert (
        "missing router headers: {'x-vsr-session-phase': 96}"
        in task_requirement["failures"]
    )
    assert any(
        failure.startswith("missing task names:")
        for failure in task_requirement["failures"]
    )
    assert "missing task phases: ['idle_boundary']" in task_requirement["failures"]


def test_diagnostic_probe_does_not_satisfy_branch_image_benchmark(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    diagnostic = write_json(
        tmp_path / "branch-diagnostic.json",
        {
            "validation_kind": "branch-image-diagnostic-probe",
            "label": "pr1989-current-mounted-binary",
            "image_tag": "pr1989-current-mounted-binary",
            "checks": {
                "chat_completion_ok": True,
                "diagnostic_headers_ok": True,
            },
            "validation_failures": [],
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
            str(inputs["cache"]),
            "--branch-image-summary",
            str(diagnostic),
        ]
    )

    report = report_mod.generate_report(args)
    branch_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "branch_image_amd_validation"
    )

    assert report["ga_ready"] is False
    assert branch_requirement["status"] == "blocked"
    assert (
        "full branch-image benchmark marker missing; diagnostic probes do not satisfy GA"
        in branch_requirement["failures"]
    )
    assert (
        "mounted-binary diagnostic does not satisfy full branch-image AMD benchmark"
        in branch_requirement["failures"]
    )
    assert (
        "branch-image check cache_token_probe_ok is not true"
        in branch_requirement["failures"]
    )


def test_branch_image_summary_requires_all_child_checks(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    branch = write_json(
        tmp_path / "branch-missing-cache-check.json",
        {
            "validation_kind": "full-branch-image-benchmark",
            "branch_image_benchmark": True,
            "image_tag": "pr1989-current-branch-image",
            "checks": {
                "diagnostic_ok": True,
                "live_matrix_ok": True,
                "failure_recovery_ok": True,
                "agent_task_ok": True,
                "mounted_binary_absent": True,
                "branch_image_benchmark_ok": True,
            },
            "validation_failures": [],
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
            str(inputs["cache"]),
            "--branch-image-summary",
            str(branch),
        ]
    )

    report = report_mod.generate_report(args)
    branch_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "branch_image_amd_validation"
    )

    assert report["ga_ready"] is False
    assert branch_requirement["status"] == "blocked"
    assert branch_requirement["failures"] == [
        "branch-image check cache_token_probe_ok is not true"
    ]


def test_missing_initial_policy_baseline_blocks_ga(tmp_path):
    report_mod = load_report_module()
    inputs = complete_inputs(tmp_path)
    ablation = write_json(
        tmp_path / "ablation-no-initial.json",
        {
            "by_policy": [
                {"policy": "single-turn"},
                {"policy": "sticky-session"},
                {"policy": "acr-no-tool-lock"},
                {
                    "policy": "acr-full",
                    "tool_loop_switch_violations": 0,
                    "context_portability_violations": 0,
                },
            ]
        },
    )
    args = report_mod.parse_args(
        [
            "--synthetic-matrix-summary",
            str(inputs["matrix"]),
            "--synthetic-ablation-summary",
            str(ablation),
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
    ablation_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "synthetic_policy_ablation"
    )

    assert report["ga_ready"] is False
    assert ablation_requirement["status"] == "blocked"
    assert "initial implementation baseline missing" in ablation_requirement["failures"]


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


def test_main_stdout_lists_blockers_for_maintainer_ops(tmp_path, capsys):
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

    exit_code = report_mod.main(
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
            "--allow-blockers",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    stdout = json.loads(capsys.readouterr().out)

    expected_blockers = [
        {
            "id": "cache_token_reporting",
            "status": "blocked",
            "title": "Cache-token reporting",
        },
        {
            "id": "branch_image_amd_validation",
            "status": "missing",
            "title": "Branch-image AMD benchmark",
        },
    ]

    assert exit_code == 0
    assert stdout["blocker_count"] == len(expected_blockers)
    assert stdout["blockers"] == expected_blockers


def test_markdown_groups_failures_under_blocker_column():
    report_mod = load_report_module()
    markdown = report_mod.render_markdown(
        {
            "generated_at": "2026-05-31T00:00:00Z",
            "ga_ready": False,
            "passed_count": 0,
            "blocker_count": 1,
            "requirements": [
                {
                    "title": "Cache-token reporting",
                    "status": "blocked",
                    "evidence": "cache.json",
                    "metrics": {"cached_token_reporting": "missing"},
                    "failures": [
                        "router: cached_token_reporting missing < positive",
                        "baseline: probe_kind missing != repeated-prefix-cache-token-probe",
                    ],
                }
            ],
        }
    )

    assert "| Requirement | Status | Evidence | Key Metrics | Blockers |" in markdown
    assert "Cache-token reporting failure" not in markdown
    assert (
        "1. router: cached_token_reporting missing < positive<br>2. baseline"
        in markdown
    )
