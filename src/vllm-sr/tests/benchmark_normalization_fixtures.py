"""Minimal source-native exports for benchmark normalizer contract tests."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from benchmark_normalization_agentic_fixtures import _ace, _continuity, _twin

MODELS = ("model-a", "model-b")


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: Iterable[object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _routerarena(root: Path) -> None:
    rows = []
    for index, model in enumerate(MODELS):
        rows.append(
            {
                "global index": "case-1",
                "prompt": "Which model should answer this?",
                "prediction": model,
                "generated_result": {
                    "generated_answer": "answer",
                    "success": True,
                    "token_usage": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                    "provider": "fixture",
                    "error": None,
                },
                "accuracy": float(index == 0),
                "cost": 0.01 + index / 100,
                "for_optimality": index != 0,
            }
        )
    _json(root / "predictions.json", rows)
    _json(
        root / "robustness_predictions.json",
        [
            {
                "global index": "case-1",
                "prompt": "Select the appropriate answering model for this request.",
                "prediction": MODELS[0],
                "generated_result": None,
                "accuracy": None,
                "cost": None,
                "for_optimality": False,
            }
        ],
    )


def _coderouter(root: Path) -> None:
    _jsonl(
        root / "id_test_tasks.jsonl",
        [
            {
                "dimension": "code_generation",
                "source_split": "test",
                "split": "id_test",
                "task_id": "task-1",
            }
        ],
    )
    header = (
        "task_id",
        "split",
        "source_split",
        "dimension",
        "model",
        "score",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "cost_source",
    )
    _csv(
        root / "id_test_results_long.csv",
        header,
        [
            (
                "task-1",
                "id_test",
                "test",
                "code_generation",
                model,
                index,
                0.01,
                4,
                2,
                6,
                10,
                "fixture",
            )
            for index, model in enumerate(MODELS)
        ],
    )
    _jsonl(
        root / "id_decisions.jsonl",
        [
            {
                "chosen_model": MODELS[0],
                "dimension": "code_generation",
                "matched_key": "fixture",
                "matched_mode": "source",
                "task_id": "task-1",
                "voter": "fixture-voter",
            }
        ],
    )
    _json(
        root / "models.json",
        {
            "models": [
                {
                    "model": model,
                    "provider": "fixture",
                    "tier": "test",
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                }
                for model in MODELS
            ]
        },
    )


def _llmrouter(root: Path) -> None:
    documents = []
    for index, model in enumerate(MODELS):
        documents.append(
            {
                "performance": float(index),
                "time_taken": 1.0,
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "cost": 0.01,
                "counts": {},
                "model_name": model,
                "dataset_name": "fixture",
                "split": "test",
                "demo": False,
                "extra_metrics": {},
                "data_fingerprint": "fixture-fingerprint",
                "records": [
                    {
                        "index": 0,
                        "origin_query": "question",
                        "prompt": "question",
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "cost": 0.01,
                        "score": float(index),
                        "prediction": "answer",
                        "ground_truth": "answer",
                        "raw_output": "answer",
                        "extra_fields": {},
                    }
                ],
            }
        )
    _jsonl(root / "results.jsonl", documents)


def _models(root: Path) -> None:
    _json(root / "models.json", {"models": list(MODELS)})


def _routerbench(root: Path) -> None:
    _models(root)
    header = ["sample_id", "prompt", "eval_name"]
    row: list[object] = ["case-1", "question", "fixture"]
    for index, model in enumerate(MODELS):
        header.extend((model, f"{model}|model_response", f"{model}|total_cost"))
        row.extend((float(index), "answer", 0.01 + index / 100))
    _csv(root / "data.csv", header, [row])


def _xroute(root: Path) -> None:
    header = (
        "task_name",
        "query",
        "ground_truth",
        "metric",
        "choices",
        "task_id",
        "model_name",
        "response",
        "token_num",
        "input_tokens",
        "output_tokens",
        "response_time",
        "performance",
        "embedding_id",
    )
    _csv(
        root / "routing-data.csv",
        header,
        [
            (
                "fixture",
                "question",
                "answer",
                "exact",
                "",
                "case-1",
                model,
                "answer",
                6,
                4,
                2,
                0.01,
                float(index),
                "embed-1",
            )
            for index, model in enumerate(MODELS)
        ],
    )


def _mmr(root: Path) -> None:
    _models(root)
    (root / "images").mkdir()
    (root / "images" / "one.png").write_bytes(b"fixture-image")
    header = ["question", "answer", "dataset_idx", "img_path"]
    row: list[object] = ["What is shown?", "one", "fixture-1", "images/one.png"]
    for index, model in enumerate(MODELS):
        header.extend((f"{model}_correct", f"{model}_cost"))
        row.extend((index == 0, 0.1 + index / 10))
    _csv(root / "MMR-Bench.csv", header, [row])


def _fusion(root: Path) -> None:
    header = (
        "task_name",
        "task_id",
        "task_description",
        "task_description_embedding",
        "query",
        "query_embedding",
        "ground_truth",
        "metric",
        "llm",
        "input_price",
        "output_price",
        "input_tokens_num",
        "output_tokens_num",
        "performance",
        "cost",
        "response",
        "llm_description",
    )
    _csv(
        root / "aligned.csv",
        header,
        [
            (
                "fixture",
                "case-1",
                "description",
                "[]",
                "question",
                "[]",
                "answer",
                "exact",
                model,
                1,
                2,
                4,
                2,
                float(index),
                0.01,
                "answer",
                "fixture model",
            )
            for index, model in enumerate(MODELS)
        ],
    )


def _r2(root: Path) -> None:
    budgets = (10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000)
    header = (
        "case_id",
        "query",
        "model",
        "budget_tokens",
        "score",
        "token_count",
        "split",
    )
    _csv(
        root / "curves.csv",
        header,
        [
            ("case-1", "question", model, budget, model_index, min(budget, 12), "test")
            for model_index, model in enumerate(MODELS)
            for budget in budgets
        ],
    )


BUILDERS = {
    "routerarena": _routerarena,
    "coderouterbench": _coderouter,
    "llmrouterbench": _llmrouter,
    "routerbench": _routerbench,
    "xroutebench": _xroute,
    "twinrouterbench": _twin,
    "mmr-bench": _mmr,
    "acebench": _ace,
    "continuity-bench": _continuity,
    "fusionfactory": _fusion,
    "r2-router": _r2,
}


def write_native_fixture(adapter_id: str, root: Path) -> None:
    root.mkdir()
    BUILDERS[adapter_id](root)
