import importlib.util
import sys
from pathlib import Path

HALF_SCORE = 0.5
MEAN_SCORE = 0.75


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


def test_dry_run_completes_all_tasks(tmp_path):
    bench = load_benchmark_module()
    args = bench.parse_args([])
    args.dry_run = True
    args.output_dir = tmp_path
    args.min_success_rate = 1.0
    args.min_task_score = 1.0
    args.max_tool_loop_violations = 0
    args.max_context_portability_violations = 0

    rows, summary = bench.run_tasks(args)
    bench.write_outputs(rows, summary, tmp_path)
    failures = bench.validate_summary(args, summary)

    assert len(rows) == len(bench.task_specs()) * 3
    assert summary["success_rate"] == 1.0
    assert summary["task_score_mean"] == 1.0
    assert summary["task_success_rate"] == 1.0
    assert summary["task_exact_success_rate"] == 1.0
    assert failures == []
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "turns.csv").exists()
    assert (tmp_path / "turns.jsonl").exists()
    assert (tmp_path / "summary.md").exists()


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
        },
    ]

    summary = bench.summarize(rows, elapsed_seconds=2.0, label="router")

    assert summary["task_score_mean"] == MEAN_SCORE
    assert summary["task_exact_successes"] == 1
    assert summary["tool_loop_switch_violations"] == 1
    assert summary["model_switches"] == 1
    assert summary["failed_tasks"] == [("b", "FIX=default-factory")]
