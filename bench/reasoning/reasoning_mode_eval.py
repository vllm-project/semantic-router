"""
Reasoning Mode Evaluation Benchmark

Issue #42: [v0.1]Bench: Reasoning mode evaluation

Acceptance Criteria:
Compare standard vs. reasoning mode using:
- Response correctness on MMLU(-Pro) and non-MMLU test sets
- Token usage (completion_tokens/prompt_tokens ratio)
- Response time per output token

This module provides a dedicated benchmark for comparing reasoning modes with
comprehensive metrics including token efficiency and throughput analysis.
"""

from __future__ import annotations

import json
import random
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from .canonical_patch import (
    VSR_CANONICAL_PATCH_JSON,
    VSR_CANONICAL_PATCH_YAML,
    generate_vsr_canonical_patch,
)
from .cli_args import parse_args
from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import Question
from .plotting import generate_comparison_plots
from .reporting import generate_markdown_report
from .router_reason_bench_multi_dataset import (
    build_extra_body_for_model,
    call_model,
    compare_free_form_answers,
    extract_answer,
    get_dataset_optimal_tokens,
)

BINARY_OPTIONS = (("Yes", "No"), ("No", "Yes"))


@dataclass
class ReasoningModeMetrics:
    """Metrics for a single reasoning mode evaluation."""

    mode_name: str
    total_questions: int = 0
    correct_answers: int = 0
    failed_queries: int = 0

    accuracy: float = 0.0
    avg_response_time: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    token_usage_ratio: float = 0.0
    time_per_output_token: float = 0.0

    response_times: list[float] = field(default_factory=list)
    completion_token_counts: list[int] = field(default_factory=list)
    prompt_token_counts: list[int] = field(default_factory=list)

    def compute_derived_metrics(self) -> None:
        """Compute derived metrics from raw data."""
        if self.total_questions > 0:
            self.accuracy = self.correct_answers / self.total_questions

        if self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)

        if self.prompt_token_counts:
            self.avg_prompt_tokens = sum(self.prompt_token_counts) / len(
                self.prompt_token_counts
            )

        if self.completion_token_counts:
            self.avg_completion_tokens = sum(self.completion_token_counts) / len(
                self.completion_token_counts
            )

        if self.avg_prompt_tokens > 0:
            self.token_usage_ratio = self.avg_completion_tokens / self.avg_prompt_tokens

        if self.avg_completion_tokens > 0:
            self.time_per_output_token = (
                self.avg_response_time * 1000
            ) / self.avg_completion_tokens

        self.avg_total_tokens = self.avg_prompt_tokens + self.avg_completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode_name": self.mode_name,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "failed_queries": self.failed_queries,
            "accuracy": round(self.accuracy, 4),
            "avg_response_time_sec": round(self.avg_response_time, 3),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens": round(self.avg_completion_tokens, 1),
            "avg_total_tokens": round(self.avg_total_tokens, 1),
            "token_usage_ratio": round(self.token_usage_ratio, 4),
            "time_per_output_token_ms": round(self.time_per_output_token, 3),
        }


@dataclass
class ReasoningModeComparison:
    """Comparison between standard and reasoning modes."""

    dataset_name: str
    model_name: str
    timestamp: str
    standard_mode: ReasoningModeMetrics | None = None
    reasoning_mode: ReasoningModeMetrics | None = None
    category_results: dict[str, dict[str, ReasoningModeMetrics]] = field(
        default_factory=dict
    )

    def get_improvement_summary(self) -> dict[str, Any]:
        """Calculate improvement of reasoning mode over standard mode."""
        if not self.standard_mode or not self.reasoning_mode:
            return {}

        return {
            "accuracy_delta": round(
                self.reasoning_mode.accuracy - self.standard_mode.accuracy, 4
            ),
            "accuracy_improvement_pct": round(
                (
                    (self.reasoning_mode.accuracy - self.standard_mode.accuracy)
                    / max(self.standard_mode.accuracy, 0.001)
                )
                * 100,
                2,
            ),
            "token_usage_ratio_delta": round(
                self.reasoning_mode.token_usage_ratio
                - self.standard_mode.token_usage_ratio,
                4,
            ),
            "time_per_output_token_delta_ms": round(
                self.reasoning_mode.time_per_output_token
                - self.standard_mode.time_per_output_token,
                3,
            ),
            "response_time_delta_sec": round(
                self.reasoning_mode.avg_response_time
                - self.standard_mode.avg_response_time,
                3,
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "timestamp": self.timestamp,
            "standard_mode": (
                self.standard_mode.to_dict() if self.standard_mode else None
            ),
            "reasoning_mode": (
                self.reasoning_mode.to_dict() if self.reasoning_mode else None
            ),
            "improvement_summary": self.get_improvement_summary(),
            "category_breakdown": {},
        }

        for category, modes in self.category_results.items():
            result["category_breakdown"][category] = {
                mode_name: metrics.to_dict() for mode_name, metrics in modes.items()
            }

        return result


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_available_models(endpoint: str, api_key: str = "") -> list[str]:
    """Get available models from the target endpoint."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "1234", timeout=30.0)
    try:
        models = client.models.list()
    except Exception as exc:
        print(f"Error fetching models: {exc}")
        return []
    return [model.id for model in models.data]


def process_question(
    client: OpenAI,
    model: str,
    question: Question,
    dataset: Any,
    max_tokens: int,
    temperature: float,
    reasoning_enabled: bool,
) -> dict[str, Any]:
    """Process a single question with the specified reasoning mode."""
    extra_body = build_extra_body_for_model(model, reasoning_enabled)
    prompt = dataset.format_prompt(question, "plain")

    start_time = time.time()
    response_text, success, prompt_tokens, completion_tokens, total_tokens = call_model(
        client, model, prompt, max_tokens, temperature, extra_body=extra_body
    )
    end_time = time.time()

    predicted_answer = extract_answer(response_text, question) if success else None
    return {
        "question_id": question.question_id,
        "category": question.category,
        "correct_answer": question.correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": _is_correct_answer(question, predicted_answer),
        "success": success,
        "response_time": end_time - start_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_enabled": reasoning_enabled,
    }


def _is_correct_answer(question: Question, predicted_answer: str | None) -> bool:
    if not predicted_answer:
        return False

    if hasattr(question, "options") and question.options:
        if _is_binary_question(question):
            return predicted_answer == question.correct_answer
        if predicted_answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return False
        if isinstance(question.correct_answer, str):
            return predicted_answer == question.correct_answer
        if isinstance(question.correct_answer, int):
            predicted_idx = ord(predicted_answer) - ord("A")
            return predicted_idx == question.correct_answer
        return False

    return compare_free_form_answers(predicted_answer, question.correct_answer)


def _is_binary_question(question: Question) -> bool:
    return tuple(question.options) in BINARY_OPTIONS


def evaluate_mode(
    questions: list[Question],
    dataset: Any,
    model: str,
    endpoint: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_enabled: bool,
    concurrent_requests: int,
    mode_name: str,
) -> tuple[ReasoningModeMetrics, pd.DataFrame]:
    """Evaluate a single reasoning mode across all questions."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "1234", timeout=300.0)
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(
                process_question,
                client,
                model,
                question,
                dataset,
                max_tokens,
                temperature,
                reasoning_enabled,
            )
            for question in questions
        ]
        try:
            for future in tqdm(
                futures, total=len(futures), desc=f"Evaluating {mode_name}"
            ):
                results.append(future.result())
        except KeyboardInterrupt:
            print(f"\n⚠️  {mode_name} evaluation interrupted. Saving partial results...")
            _cancel_futures(futures)
            _append_completed_results(futures, results)

    metrics = _metrics_from_results(results, mode_name)
    return metrics, pd.DataFrame(results)


def _cancel_futures(futures: list[Future[dict[str, Any]]]) -> None:
    for future in futures:
        future.cancel()


def _append_completed_results(
    futures: list[Future[dict[str, Any]]], results: list[dict[str, Any]]
) -> None:
    for future in futures:
        if future.done() and not future.cancelled():
            with suppress(Exception):
                results.append(future.result())


def _metrics_from_results(
    results: list[dict[str, Any]], mode_name: str
) -> ReasoningModeMetrics:
    metrics = ReasoningModeMetrics(mode_name=mode_name)
    metrics.total_questions = len(results)

    successful = [result for result in results if result["success"]]
    metrics.correct_answers = sum(1 for result in successful if result["is_correct"])
    metrics.failed_queries = len(results) - len(successful)
    _append_successful_metrics(metrics, successful)
    metrics.compute_derived_metrics()
    return metrics


def _append_successful_metrics(
    metrics: ReasoningModeMetrics, successful_rows: list[Any]
) -> None:
    for row in successful_rows:
        if row["response_time"] is not None:
            metrics.response_times.append(float(row["response_time"]))
        if row["prompt_tokens"] is not None:
            metrics.prompt_token_counts.append(int(row["prompt_tokens"]))
        if row["completion_tokens"] is not None:
            metrics.completion_token_counts.append(int(row["completion_tokens"]))


def evaluate_dataset(
    dataset_name: str,
    model: str,
    endpoint: str,
    api_key: str,
    samples_per_category: int,
    categories: list[str] | None,
    max_tokens: int | None,
    temperature: float,
    concurrent_requests: int,
    seed: int,
) -> tuple[ReasoningModeComparison, pd.DataFrame, pd.DataFrame]:
    """Evaluate both standard and reasoning modes on a dataset."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    dataset = DatasetFactory.create_dataset(dataset_name)
    questions, dataset_info = dataset.load_dataset(
        categories=categories,
        samples_per_category=samples_per_category,
        seed=seed,
    )
    print(
        f"Loaded {len(questions)} questions across {len(dataset_info.categories)} categories"
    )

    effective_max_tokens = max_tokens or get_dataset_optimal_tokens(dataset_info, model)
    print(f"Using max_tokens: {effective_max_tokens}")

    comparison = ReasoningModeComparison(
        dataset_name=dataset_info.name,
        model_name=model,
        timestamp=datetime.now().isoformat(),
    )
    comparison.standard_mode, standard_df = _run_dataset_mode(
        questions=questions,
        dataset=dataset,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        reasoning_enabled=False,
        concurrent_requests=concurrent_requests,
        mode_name="standard",
        label="📊 Standard Mode (reasoning=False)",
    )
    comparison.reasoning_mode, reasoning_df = _run_dataset_mode(
        questions=questions,
        dataset=dataset,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        reasoning_enabled=True,
        concurrent_requests=concurrent_requests,
        mode_name="reasoning",
        label="🧠 Reasoning Mode (reasoning=True)",
    )
    comparison.category_results = _category_breakdown(
        questions, dataset_info.categories, standard_df, reasoning_df
    )
    return comparison, standard_df, reasoning_df


def _run_dataset_mode(
    questions: list[Question],
    dataset: Any,
    model: str,
    endpoint: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_enabled: bool,
    concurrent_requests: int,
    mode_name: str,
    label: str,
) -> tuple[ReasoningModeMetrics, pd.DataFrame]:
    print(f"\n{label}")
    return evaluate_mode(
        questions=questions,
        dataset=dataset,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled,
        concurrent_requests=concurrent_requests,
        mode_name=mode_name,
    )


def _category_breakdown(
    questions: list[Question],
    categories: list[str],
    standard_df: pd.DataFrame,
    reasoning_df: pd.DataFrame,
) -> dict[str, dict[str, ReasoningModeMetrics]]:
    category_results: dict[str, dict[str, ReasoningModeMetrics]] = {}
    present_categories = {question.category for question in questions}
    for category in categories:
        if category not in present_categories:
            continue
        category_results[category] = {
            "standard": _metrics_from_frame(
                standard_df[standard_df["category"] == category], "standard"
            ),
            "reasoning": _metrics_from_frame(
                reasoning_df[reasoning_df["category"] == category], "reasoning"
            ),
        }
    return category_results


def _metrics_from_frame(frame: pd.DataFrame, mode_name: str) -> ReasoningModeMetrics:
    metrics = ReasoningModeMetrics(mode_name=mode_name)
    metrics.total_questions = len(frame)
    metrics.correct_answers = int(frame["is_correct"].sum())
    metrics.failed_queries = int((~frame["success"]).sum())
    _append_successful_metrics(metrics, frame[frame["success"]].to_dict("records"))
    metrics.compute_derived_metrics()
    return metrics


def save_results(
    comparisons: list[ReasoningModeComparison],
    all_standard_dfs: list[pd.DataFrame],
    all_reasoning_dfs: list[pd.DataFrame],
    output_dir: Path,
    model_name: str = "evaluated-model",
    reasoning_family: str | None = None,
) -> None:
    """Save all results to files, including the canonical vSR patch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vsr_config = generate_vsr_canonical_patch(comparisons, model_name, reasoning_family)

    with open(output_dir / VSR_CANONICAL_PATCH_JSON, "w") as handle:
        json.dump(vsr_config, handle, indent=2, cls=NumpyEncoder)

    with open(output_dir / VSR_CANONICAL_PATCH_YAML, "w") as handle:
        yaml.dump(
            vsr_config["suggested_vsr_patch"],
            handle,
            default_flow_style=False,
            sort_keys=False,
        )

    print("\n📝 vSR canonical patch generated:")
    print(f"   - JSON: {output_dir / VSR_CANONICAL_PATCH_JSON}")
    print(f"   - YAML: {output_dir / VSR_CANONICAL_PATCH_YAML}")
    print(f"\n{vsr_config['recommendation']}")
    if "manual_follow_up" in vsr_config:
        print(f"\nFollow-up required: {vsr_config['manual_follow_up']}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "issue": "42",
        "title": "Reasoning Mode Evaluation",
        "comparisons": [comparison.to_dict() for comparison in comparisons],
        "vsr_canonical_patch_recommendation": vsr_config,
    }
    with open(output_dir / "reasoning_mode_eval_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, cls=NumpyEncoder)

    for comparison, standard_df, reasoning_df in zip(
        comparisons, all_standard_dfs, all_reasoning_dfs, strict=True
    ):
        _save_dataset_csvs(output_dir, comparison, standard_df, reasoning_df)

    print(f"\n✅ Results saved to {output_dir}")


def _save_dataset_csvs(
    output_dir: Path,
    comparison: ReasoningModeComparison,
    standard_df: pd.DataFrame,
    reasoning_df: pd.DataFrame,
) -> None:
    dataset_dir = output_dir / comparison.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    standard_export = standard_df.copy()
    reasoning_export = reasoning_df.copy()
    standard_export["mode"] = "standard"
    reasoning_export["mode"] = "reasoning"

    combined_df = pd.concat([standard_export, reasoning_export], ignore_index=True)
    combined_df.to_csv(dataset_dir / "detailed_results.csv", index=False)
    standard_export.to_csv(dataset_dir / "standard_mode_results.csv", index=False)
    reasoning_export.to_csv(dataset_dir / "reasoning_mode_results.csv", index=False)


def _resolve_model(args: Any) -> str | None:
    if args.model:
        return args.model

    print("Fetching available models from endpoint...")
    models = get_available_models(args.endpoint, args.api_key)
    if not models:
        print("❌ No models available. Please specify --model")
        return None

    model = models[0]
    print(f"Using model: {model}")
    return model


def _print_benchmark_header(args: Any, model: str) -> None:
    print("\n" + "=" * 60)
    print("🧠 REASONING MODE EVALUATION BENCHMARK")
    print("=" * 60)
    print("Issue #42: Standard vs Reasoning Mode Comparison")
    print(f"Model: {model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Samples per category: {args.samples_per_category}")
    print(f"Endpoint: {args.endpoint}")
    print("=" * 60 + "\n")


def _print_dataset_summary(
    dataset_name: str, comparison: ReasoningModeComparison
) -> None:
    print(f"\n📊 {dataset_name} Summary:")
    print(f"  Standard Accuracy:  {comparison.standard_mode.accuracy:.4f}")
    print(f"  Reasoning Accuracy: {comparison.reasoning_mode.accuracy:.4f}")
    print(
        "  Accuracy Delta:     "
        f"{comparison.reasoning_mode.accuracy - comparison.standard_mode.accuracy:+.4f}"
    )
    print(f"  Token Ratio (Std):  {comparison.standard_mode.token_usage_ratio:.4f}")
    print(f"  Token Ratio (Reas): {comparison.reasoning_mode.token_usage_ratio:.4f}")
    print(
        f"  Time/Token (Std):   {comparison.standard_mode.time_per_output_token:.2f} ms"
    )
    print(
        f"  Time/Token (Reas):  {comparison.reasoning_mode.time_per_output_token:.2f} ms"
    )


def _evaluate_requested_datasets(
    args: Any, model: str
) -> tuple[list[ReasoningModeComparison], list[pd.DataFrame], list[pd.DataFrame]]:
    comparisons: list[ReasoningModeComparison] = []
    all_standard_dfs: list[pd.DataFrame] = []
    all_reasoning_dfs: list[pd.DataFrame] = []

    for dataset_name in args.datasets:
        try:
            comparison, standard_df, reasoning_df = evaluate_dataset(
                dataset_name=dataset_name,
                model=model,
                endpoint=args.endpoint,
                api_key=args.api_key,
                samples_per_category=args.samples_per_category,
                categories=args.categories,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrent_requests=args.concurrent_requests,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"❌ Error evaluating {dataset_name}: {exc}")
            traceback.print_exc()
            continue

        comparisons.append(comparison)
        all_standard_dfs.append(standard_df)
        all_reasoning_dfs.append(reasoning_df)
        _print_dataset_summary(dataset_name, comparison)

    return comparisons, all_standard_dfs, all_reasoning_dfs


def main() -> None:
    """Main entry point for reasoning mode evaluation."""
    args = parse_args()
    if args.list_datasets:
        list_available_datasets()
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    model = _resolve_model(args)
    if model is None:
        return

    _print_benchmark_header(args, model)
    comparisons, all_standard_dfs, all_reasoning_dfs = _evaluate_requested_datasets(
        args, model
    )
    if not comparisons:
        print("❌ No successful evaluations. Exiting.")
        return

    output_dir = Path(args.output_dir)
    save_results(
        comparisons,
        all_standard_dfs,
        all_reasoning_dfs,
        output_dir,
        model_name=model,
        reasoning_family=args.reasoning_family,
    )

    if args.generate_plots:
        print("\n📈 Generating comparison plots...")
        for comparison in comparisons:
            generate_comparison_plots(comparison, output_dir / "plots")

    if args.generate_report:
        print("\n📝 Generating markdown report...")
        generate_markdown_report(comparisons, output_dir)

    print("\n" + "=" * 60)
    print("✅ REASONING MODE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
