# ruff: noqa: PLR2004
from __future__ import annotations

import json
from pathlib import Path

from bench.router_flow.real_eval.collect_evalscope_results import (
    collect_scores,
    first_count_value,
    first_metric_value,
    load_reference_scores,
    suite_model_labels,
    suite_models,
    write_outputs,
)


def test_first_metric_value_finds_nested_evalscope_metric() -> None:
    report = {
        "model": "flow_gpqa_d",
        "datasets": {
            "gpqa_diamond": {
                "metrics": {
                    "acc": 0.965,
                    "count": 20,
                }
            }
        },
    }

    assert first_metric_value(report, "acc") == 0.965


def test_first_metric_value_accepts_evalscope_mean_metric_alias() -> None:
    report = {
        "metrics": [
            {
                "name": "mean_acc",
                "score": 1.0,
                "categories": [{"name": ["default"], "score": 1.0}],
            }
        ]
    }

    assert first_metric_value(report, "acc") == 1.0


def test_first_metric_value_accepts_mrcr_overall_alias() -> None:
    report = {
        "metrics": [
            {"name": "overall_mrcr_score", "score": 0.9},
            {"name": "131072-262144_mrcr_score", "score": 1.0},
        ]
    }

    assert first_metric_value(report, "mrcr_score") == 0.9


def test_first_count_value_finds_evalscope_num() -> None:
    report = {
        "score": 0.9,
        "num": 20,
        "details": {"num_samples": 999},
    }

    assert first_count_value(report) == 20


def test_suite_models_defaults_to_suite_models() -> None:
    suite = {"models": {"auto": "vllm-sr/auto"}}

    assert suite_models(suite, []) == {"auto": "vllm-sr/auto"}


def test_suite_model_labels_can_override_auto_identity() -> None:
    suite = {
        "models": {"auto": "vllm-sr/auto", "glm52_native": "z-ai/glm-5.2"},
        "model_labels": {"auto": "VSR 1.0", "glm52_native": "GLM 5.2"},
    }

    assert suite_model_labels(suite, suite_models(suite, [])) == {
        "auto": "VSR 1.0",
        "glm52_native": "GLM 5.2",
    }


def test_collect_scores_aligns_evalscope_reports_with_reference_scores(
    tmp_path: Path,
) -> None:
    suite = {
        "models": {
            "auto": "vllm-sr/auto",
            "flow": "vllm-sr/flow",
        },
        "benchmarks": [
            {
                "id": "gpqa_d",
                "public_name": "GPQA-D",
                "dataset": "gpqa_diamond",
                "metric": "mean_acc",
                "tier": "core",
                "default_run": True,
            }
        ],
    }
    output_root = tmp_path / "evalscope"
    report_dir = output_root / "gpqa_d" / "flow" / "reports" / "flow_gpqa_d"
    report_dir.mkdir(parents=True)
    (report_dir / "gpqa_diamond.json").write_text(
        json.dumps({"metrics": [{"name": "mean_acc", "score": 0.965}], "num": 20})
    )
    reference_path = tmp_path / "public_reference_scores.json"
    reference_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark": "GPQA-D",
                        "Fugu": 95.5,
                        "Fugu Ultra": 95.5,
                    }
                ]
            }
        )
    )

    collected = collect_scores(
        suite=suite,
        output_root=output_root,
        references=load_reference_scores(reference_path),
        requested_models=["auto", "flow"],
        requested_benchmarks=[],
        include_heavy=False,
    )

    row = collected["benchmarks"][0]
    assert row["models"]["flow"]["score"] == 96.5
    assert row["models"]["flow"]["raw_score"] == 0.965
    assert row["models"]["flow"]["num"] == 20
    assert row["comparison"]["flow"]["beats_fugu_ultra"] is True
    assert row["models"]["auto"]["status"] == "missing_report"
    assert collected["model_labels"]["auto"] == "VSR 1.0 Pro"
    assert collected["missing"] == [{"benchmark": "gpqa_d", "model": "auto"}]


def test_collect_scores_compares_vsr_against_glm52_reference(
    tmp_path: Path,
) -> None:
    suite = {
        "models": {"auto": "vllm-sr/auto"},
        "model_labels": {"auto": "VSR 1.0"},
        "benchmarks": [
            {
                "id": "hle_text",
                "public_name": "Humanity's Last Exam",
                "dataset": "hle",
                "metric": "acc",
                "tier": "core",
                "default_run": True,
            }
        ],
    }
    output_root = tmp_path / "evalscope"
    report_dir = output_root / "hle_text" / "auto" / "reports" / "auto_hle"
    report_dir.mkdir(parents=True)
    (report_dir / "hle.json").write_text(
        json.dumps({"metrics": [{"name": "acc", "score": 0.415}], "num": 24})
    )
    reference_path = tmp_path / "public_reference_scores.json"
    reference_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark": "Humanity's Last Exam",
                        "GLM 5.2": 40.5,
                        "Fugu": 48.5,
                        "Fugu Ultra": 50.0,
                    }
                ]
            }
        )
    )

    collected = collect_scores(
        suite=suite,
        output_root=output_root,
        references=load_reference_scores(reference_path),
        requested_models=[],
        requested_benchmarks=[],
        include_heavy=False,
    )

    comparison = collected["benchmarks"][0]["comparison"]["auto"]
    assert comparison["delta_vs_glm_5_2"] == 1.0
    assert comparison["beats_glm_5_2"] is True
    assert comparison["beats_fugu"] is False
    assert collected["model_labels"]["auto"] == "VSR 1.0"


def test_write_outputs_creates_blog_ready_artifacts(tmp_path: Path) -> None:
    collected = {
        "score_unit": "0_to_100",
        "models": {"flow": "vllm-sr/flow"},
        "benchmarks": [
            {
                "id": "gpqa_d",
                "benchmark": "GPQA-D",
                "dataset": "gpqa_diamond",
                "metric": "acc",
                "tier": "core",
                "models": {
                    "flow": {
                        "score": 96.5,
                        "raw_score": 0.965,
                        "num": 20,
                        "metric": "acc",
                        "report_path": "reports/flow_gpqa_d/gpqa_diamond.json",
                        "status": "ok",
                    }
                },
                "reference": {"Fugu": 95.5, "Fugu Ultra": 95.5},
                "comparison": {},
            }
        ],
        "averages": {
            "models": {"flow": 96.5},
            "reference": {"GLM 5.2": None, "Fugu": 95.5, "Fugu Ultra": 95.5},
        },
        "missing": [],
        "source": "test",
    }

    write_outputs(tmp_path, collected)

    assert (tmp_path / "evalscope_scores.json").exists()
    assert (
        "| GPQA-D | 96.5 |  | 95.5 | 95.5 |"
        in (tmp_path / "benchmark_table.md").read_text()
    )
    assert (tmp_path / "overall_bars.svg").exists()
    assert (tmp_path / "benchmark_bars.svg").exists()
