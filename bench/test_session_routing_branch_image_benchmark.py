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
    "issue-pr-maintenance-loop",
    "configuration-contract-review",
    "repo-bisect-debug",
    "dependency-upgrade-regression",
    "literature-data-extraction",
    "stale-pr-rebase-triage",
    "benchmark-regression-root-cause",
    "paper-figure-quality-review",
    "feature-implementation-loop",
    "research-claim-grounding-loop",
    "tool-timeout-retry-loop",
    "ci-patch-review-loop",
    "paper-rebuttal-revision-loop",
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


def live_aggregate(ref="77e6b573", image_tag="pr1989-current-branch-image"):
    return {
        "evidence_ref": ref,
        "evidence_image_tag": image_tag,
        "rows": [
            {
                "name": "balanced-branch-image",
                "router_requests": 128,
                "success_rate": 1.0,
                "tool_loop_switch_violations": 0,
                "context_portability_violations": 0,
                "validation_failures": [],
            }
        ],
    }


def failure_aggregate(ref="77e6b573", image_tag="pr1989-current-branch-image"):
    return {
        "evidence_ref": ref,
        "evidence_image_tag": image_tag,
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
        ],
    }


def complete_agent_task_summary(
    ref="77e6b573", image_tag="pr1989-current-branch-image"
):
    return {
        "evidence_ref": ref,
        "evidence_image_tag": image_tag,
        "requests": 489,
        "success_rate": 1.0,
        "successes": 489,
        "task_count": 28,
        "tasks": 28,
        "task_instances": 84,
        "task_success_rate": 1.0,
        "task_score_mean": 1.0,
        "task_names": REQUIRED_LONG_HORIZON_TASKS,
        "scored_task_names": REQUIRED_LONG_HORIZON_TASKS,
        "phase_counts": {
            "user_turn": 84,
            "tool_loop": 180,
            "provider_state": 81,
            "topic_drift": 45,
            "idle_boundary": 15,
            "final": 84,
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


def cache_aggregate(ref="77e6b573", image_tag="pr1989-current-branch-image"):
    router = {
        "evidence_ref": ref,
        "evidence_image_tag": image_tag,
        "label": "branch-image-cache-router",
        "base_url": "http://router.local/v1",
        "model": "auto",
        "requests": 8,
        "successes": 8,
        "success_rate": 1.0,
        "cached_token_reporting": "missing",
        "cached_token_field_present": 0,
        "cached_token_field_rate": 0.0,
        "cached_prompt_ratio": 0.0,
        "probe_kind": "repeated-prefix-cache-token-probe",
        "probe_repeats": 8,
        "stable_prefix_chars": 12719,
        "stable_prefix_sha256": "abc123",
        "unique_suffix_pattern": "probe_turn_index",
    }
    baseline = {
        "label": "branch-image-cache-baseline",
        "base_url": "http://backend.local/v1",
        "model": "qwen/qwen3.5-rocm",
        "requests": 8,
        "successes": 8,
        "success_rate": 1.0,
        "cached_token_reporting": "missing",
        "cached_token_field_present": 0,
        "cached_token_field_rate": 0.0,
        "cached_prompt_ratio": 0.0,
        "probe_kind": "repeated-prefix-cache-token-probe",
        "probe_repeats": 8,
        "stable_prefix_chars": 12719,
        "stable_prefix_sha256": "abc123",
        "unique_suffix_pattern": "probe_turn_index",
    }
    return {"router": router, "baseline": baseline}


def complete_args(tmp_path):
    diagnostic = write_json(tmp_path / "diagnostic.json", diagnostic_summary())
    live = write_json(tmp_path / "live.json", live_aggregate())
    failure = write_json(tmp_path / "failure.json", failure_aggregate())
    tasks = write_json(tmp_path / "tasks.json", complete_agent_task_summary())
    cache = write_json(tmp_path / "cache.json", cache_aggregate())
    return [
        "--diagnostic-summary",
        str(diagnostic),
        "--live-aggregate",
        str(live),
        "--failure-aggregate",
        str(failure),
        "--agent-task-summary",
        str(tasks),
        "--cache-aggregate",
        str(cache),
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
    assert summary["checks"]["cache_token_probe_ok"] is True
    assert summary["evidence"]["cache_aggregates"][0]["baseline_present"] is True
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
    assert "agent task requests 96.0 < 489" in summary["validation_failures"]
    assert (
        "agent task missing router headers: "
        "{'x-vsr-session-phase': 96, 'x-vsr-selected-confidence': 96, "
        "'x-vsr-context-token-count': 96}" in summary["validation_failures"]
    )


def test_full_branch_image_summary_blocks_unbound_subsummaries(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    unbound_live = live_aggregate()
    unbound_live.pop("evidence_ref")
    unbound_live.pop("evidence_image_tag")
    write_json(tmp_path / "live.json", unbound_live)
    write_json(
        tmp_path / "failure.json",
        failure_aggregate(ref="old-ref", image_tag="old-image"),
    )
    write_json(
        tmp_path / "tasks.json", complete_agent_task_summary(image_tag="old-image")
    )
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["branch_image_benchmark_ok"] is False
    assert (
        "balanced-branch-image evidence_ref missing" in summary["validation_failures"]
    )
    assert (
        "balanced-branch-image evidence_image_tag missing"
        in summary["validation_failures"]
    )
    assert (
        "tool-loop-recovery evidence_ref old-ref != 77e6b573"
        in summary["validation_failures"]
    )
    assert (
        "tool-loop-recovery evidence_image_tag old-image != "
        "pr1989-current-branch-image"
    ) in summary["validation_failures"]
    assert (
        "agent task evidence_image_tag old-image != pr1989-current-branch-image"
        in summary["validation_failures"]
    )


def test_full_branch_image_summary_requires_diagnostic_identity(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    diagnostic = diagnostic_summary()
    diagnostic["validation_kind"] = "manual-diagnostic"
    diagnostic.pop("ref")
    diagnostic.pop("image_tag")
    write_json(tmp_path / "diagnostic.json", diagnostic)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["diagnostic_ok"] is False
    assert summary["checks"]["branch_image_benchmark_ok"] is False
    assert (
        "diagnostic validation_kind manual-diagnostic != "
        "branch-image-diagnostic-probe"
    ) in summary["validation_failures"]
    assert "diagnostic ref missing" in summary["validation_failures"]
    assert "diagnostic image_tag missing" in summary["validation_failures"]


def test_full_branch_image_summary_requires_cache_aggregate(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    index = arg_values.index("--cache-aggregate")
    del arg_values[index : index + 2]
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert "at least one cache aggregate is required" in summary["validation_failures"]


def test_full_branch_image_summary_blocks_unbound_cache_probe(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    unbound_cache = cache_aggregate(ref="old-ref", image_tag="old-image")
    write_json(tmp_path / "cache.json", unbound_cache)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert (
        "cache router evidence_ref old-ref != 77e6b573"
        in summary["validation_failures"]
    )
    assert (
        "cache router evidence_image_tag old-image != pr1989-current-branch-image"
        in summary["validation_failures"]
    )


def test_full_branch_image_summary_blocks_non_probe_cache_aggregate(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    invalid_cache = cache_aggregate()
    invalid_cache["router"].pop("probe_kind")
    invalid_cache.pop("baseline")
    write_json(tmp_path / "cache.json", invalid_cache)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert (
        "cache aggregate direct backend baseline missing; run "
        "cache_token_probe.py with --baseline-base-url"
    ) in summary["validation_failures"]
    assert (
        "cache router probe_kind missing != repeated-prefix-cache-token-probe"
        in summary["validation_failures"]
    )


def test_full_branch_image_summary_blocks_cache_probe_identity_mismatch(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    mismatched_cache = cache_aggregate()
    mismatched_cache["baseline"]["stable_prefix_sha256"] = "different-prefix"
    write_json(tmp_path / "cache.json", mismatched_cache)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert (
        "cache router/baseline stable_prefix_sha256 mismatch: "
        "abc123 != different-prefix"
    ) in summary["validation_failures"]


def test_full_branch_image_summary_requires_direct_backend_serving_path_identity(
    tmp_path,
):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    same_serving_path_cache = cache_aggregate()
    same_serving_path_cache["baseline"]["base_url"] = "http://router.local/v1/"
    write_json(tmp_path / "cache.json", same_serving_path_cache)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert (
        "cache baseline base_url must differ from router base_url for "
        "direct-backend serving-path cache evidence"
    ) in summary["validation_failures"]


def test_full_branch_image_summary_blocks_cache_runner_validation_failures(tmp_path):
    benchmark = load_benchmark_module()
    arg_values = complete_args(tmp_path)
    invalid_cache = cache_aggregate()
    invalid_cache["validation_failures"] = [
        "direct-backend: base_url must differ from router base_url"
    ]
    invalid_cache["router"]["validation_failures"] = [
        "router: cached_token_reporting missing < reported_zero"
    ]
    write_json(tmp_path / "cache.json", invalid_cache)
    args = benchmark.parse_args(arg_values)

    summary = benchmark.build_summary(args)

    assert summary["checks"]["cache_token_probe_ok"] is False
    assert (
        "cache aggregate validation_failure: "
        "direct-backend: base_url must differ from router base_url"
    ) in summary["validation_failures"]
    assert (
        "cache router validation_failure: "
        "router: cached_token_reporting missing < reported_zero"
    ) in summary["validation_failures"]


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
