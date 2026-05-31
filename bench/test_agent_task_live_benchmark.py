import importlib.util
import sys
from pathlib import Path

HALF_SCORE = 0.5
MEAN_SCORE = 0.75
MISSING_HEADER_COUNT = 2


def load_benchmark_module():
    path = Path(__file__).with_name("agent_task_live_benchmark.py")
    spec = importlib.util.spec_from_file_location("agent_task_live_benchmark", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_answer_reports_missing_terms():
    bench = load_benchmark_module()

    score, missing = bench.score_answer(
        "ROOT_CAUSE=mutable-default",
        ("ROOT_CAUSE=mutable-default", "FIX=default-factory"),
    )

    assert score == HALF_SCORE
    assert missing == ["FIX=default-factory"]


def test_scored_turn_messages_include_exact_matching_instruction():
    bench = load_benchmark_module()
    task = bench.task_specs("smoke")[0]

    messages = bench.build_messages(task, 2)

    assert "Scoring uses exact substring matching" in messages[-1]["content"]
    assert "ROOT_CAUSE=mutable-default" in messages[-1]["content"]
    assert "FIX=default-factory" in messages[-1]["content"]


def test_dry_run_completes_all_tasks(tmp_path):
    bench = load_benchmark_module()
    args = bench.parse_args([])
    args.dry_run = True
    args.output_dir = tmp_path
    args.min_success_rate = 1.0
    args.min_task_score = 1.0
    args.max_tool_loop_violations = 0
    args.max_context_portability_violations = 0
    args.require_router_header = [
        "x-vsr-selected-model",
        "x-vsr-selected-decision",
        "x-vsr-replay-id",
    ]

    rows, summary = bench.run_tasks(args)
    bench.write_outputs(rows, summary, tmp_path)
    failures = bench.validate_summary(args, summary)

    assert len(rows) == len(bench.task_specs()) * 3
    assert summary["success_rate"] == 1.0
    assert summary["task_score_mean"] == 1.0
    assert summary["task_success_rate"] == 1.0
    assert summary["task_instances"] == len(bench.task_specs())
    assert summary["task_exact_success_rate"] == 1.0
    assert summary["missing_router_header_counts"]["x-vsr-replay-id"] == 0
    assert failures == []
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "turns.csv").exists()
    assert (tmp_path / "turns.jsonl").exists()
    assert (tmp_path / "summary.md").exists()


def test_long_horizon_suite_repetitions_expand_task_instances():
    bench = load_benchmark_module()
    args = bench.parse_args(["--suite", "long-horizon", "--task-repetitions", "2"])
    args.dry_run = True

    rows, summary = bench.run_tasks(args)
    expected_tasks = len(bench.task_specs("long-horizon"))

    expected_turns = sum(len(task.turns) for task in bench.task_specs("long-horizon"))

    assert len(rows) == expected_turns * 2
    assert summary["task_count"] == expected_tasks
    assert summary["task_instances"] == expected_tasks * 2
    assert summary["task_success_rate"] == 1.0
    assert rows[0]["task_suite"] == "long-horizon"
    assert rows[0]["task_repetition"] == 0
    assert rows[-1]["task_repetition"] == 1


def test_summary_tracks_quality_and_continuity_violations():
    bench = load_benchmark_module()
    rows = [
        {
            "task": "a",
            "success": True,
            "status": 200,
            "latency_ms": 10.0,
            "answer_score": 1.0,
            "scored_turn": True,
            "model_switched": False,
            "tool_loop_switch_violation": False,
            "context_portability_violation": False,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "phase": "final",
            "selected_model": "m1",
            "missing_terms": "",
            "answer_excerpt": "",
        },
        {
            "task": "b",
            "success": True,
            "status": 200,
            "latency_ms": 20.0,
            "answer_score": 0.5,
            "scored_turn": True,
            "model_switched": True,
            "tool_loop_switch_violation": True,
            "context_portability_violation": False,
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "cached_tokens": 0,
            "phase": "tool_loop",
            "selected_model": "m2",
            "missing_terms": "FIX=default-factory",
            "answer_excerpt": "ROOT_CAUSE=mutable-default",
        },
    ]

    summary = bench.summarize(rows, elapsed_seconds=2.0, label="router")

    assert summary["task_score_mean"] == MEAN_SCORE
    assert summary["task_exact_successes"] == 1
    assert summary["tool_loop_switch_violations"] == 1
    assert summary["model_switches"] == 1
    assert (
        summary["missing_router_header_counts"]["x-vsr-selected-model"]
        == MISSING_HEADER_COUNT
    )
    assert summary["failed_tasks"] == [("b", "FIX=default-factory")]

    args = bench.parse_args(["--require-router-header", "x-vsr-selected-model"])
    assert bench.validate_summary(args, summary) == [
        "missing_router_header x-vsr-selected-model: 2 successful requests"
    ]
