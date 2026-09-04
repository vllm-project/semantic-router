"""Minimal source-native agentic exports for normalizer contract tests."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _twin(root: Path) -> None:
    messages = [{"role": "user", "content": "agent step"}]
    bank = {
        "id": "row-1",
        "benchmark": "swebench",
        "scenario": "code_swe",
        "instance_id": "instance-1",
        "step_index": 1,
        "total_steps": 1,
        "messages": messages,
        "target_tier": "low",
        "target_tier_id": 0,
        "benchmark_display": "SWE",
        "benchmark_subset": "fixture",
        "benchmark_version": "fixture-v1",
        "pipeline_stage": "degradation_search_done",
        "collector": "fixture",
        "collected_at": "2026-01-01",
        "notes": "fixture",
    }
    _jsonl(root / "question_bank.jsonl", [bank])
    row = {
        "id": "row-1",
        "benchmark": "swebench",
        "gold_tier_id": 0,
        "pred_tier_id": 0,
        "match": True,
        "passed": True,
        "instance_id": "instance-1",
        "step_index": 1,
        "total_steps": 1,
        "messages": messages,
    }
    _json(
        root / "eval_summary.json",
        {
            "classifier": "fixture",
            "shard": "fixture",
            "sample_mode": "full",
            "sampled": 1,
            "seed": 42,
            "proportional_quotas": None,
            "benchmark_counts": {"swebench": 1},
            "exact_match": 1,
            "tier_match_accuracy": 1.0,
            "accuracy_excluding_errors": 1.0,
            "api_errors": 0,
            "valid_response_rate": 1.0,
            "scores_v2": {},
            "section_11": {},
            "router_accounting": {},
            "by_benchmark": {},
            "errors": [],
            "rows": [row],
        },
    )


def _ace(root: Path) -> None:
    result = {
        "task_id": "task-1",
        "scores": {"overall_score": 0.8, "privacy_score": 1.0},
        "error": None,
        "usage": {},
        "task_route": "LOCAL",
    }
    _json(
        root / "summary.json",
        {
            "category": "fixture",
            "model": "fixture",
            "total_tasks": 1,
            "returned_tasks": 1,
            "scored_tasks": 1,
            "overall_average": 0.8,
            "overall_bar": "",
            "pass_threshold": 0.7,
            "pass_count": 1,
            "pass_rate": 1.0,
            "privacy_average": 1.0,
            "privacy_bar": "",
            "usage": {},
            "final_scores": [],
            "results": [result],
        },
    )


def _continuity(root: Path) -> None:
    _json(
        root / "conversations.json",
        [
            {
                "id": "conv-1",
                "turns": [
                    {"role": "user", "content": "Remember blue"},
                    {"role": "user", "content": "What color?"},
                ],
                "fact_turn_index": 0,
                "probe_turn_index": 1,
                "expected_fact": "blue",
                "fact_type": "preference",
            }
        ],
    )
    _json(
        root / "experiment_manifest.json",
        [
            {
                "conversation_id": "conv-1",
                "failure_turn": 1,
                "mode": "timeout",
                "fallback_provider": "fixture-fallback",
            }
        ],
    )
    _continuity_logs(root)
    _continuity_metrics(root)


def _continuity_logs(root: Path) -> None:
    common_log = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "conversation_id": "conv-1",
        "turn_index": 1,
        "provider": "fixture-fallback",
        "model": "fixture-model",
        "failed_over": True,
        "failover_from": "fixture-primary",
        "failure_mode": "timeout",
        "latency_ms": 10,
        "time_to_first_token_ms": None,
        "error": None,
    }
    _jsonl(
        root / "baseline_log.jsonl",
        [
            {
                **common_log,
                "request_id": "request-baseline",
                "response_text": "I do not remember.",
                "system": "baseline",
            }
        ],
    )
    _jsonl(
        root / "treatment_log.jsonl",
        [
            {
                **common_log,
                "request_id": "request-treatment",
                "response_text": "blue",
                "system": "treatment",
            }
        ],
    )


def _continuity_metrics(root: Path) -> None:
    header = (
        "conversation_id",
        "system",
        "failed_over",
        "latency_ms",
        "queue_wait_ms",
        "preserved",
        "expected_fact",
        "reasoning",
        "provider",
        "turn_count",
        "concurrency",
    )
    _csv(
        root / "raw_metrics.csv",
        header,
        [
            (
                "conv-1",
                "baseline",
                True,
                10,
                1,
                True,
                "blue",
                "preserved",
                "fixture",
                2,
                1,
            ),
            (
                "conv-1",
                "treatment",
                True,
                10,
                1,
                True,
                "blue",
                "preserved",
                "fixture-fallback",
                2,
                1,
            ),
        ],
    )
