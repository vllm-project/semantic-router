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

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from openai import OpenAI
from tqdm import tqdm

from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import DatasetInfo, Question
from .router_reason_bench_multi_dataset import (
    build_extra_body_for_model,
    call_model,
    extract_answer,
    compare_free_form_answers,
    get_dataset_optimal_tokens,
)


@dataclass
class ReasoningModeMetrics:
    """Metrics for a single reasoning mode evaluation."""

    mode_name: str
    total_questions: int = 0
    correct_answers: int = 0
    failed_queries: int = 0

    # Core metrics
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0

    # Issue #42 specific metrics
    token_usage_ratio: float = 0.0  # completion_tokens / prompt_tokens
    time_per_output_token: float = 0.0  # response_time / completion_tokens (ms)

    # Distribution stats
    response_times: List[float] = field(default_factory=list)
    completion_token_counts: List[int] = field(default_factory=list)
    prompt_token_counts: List[int] = field(default_factory=list)

    def compute_derived_metrics(self):
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

        # Issue #42 specific: Token usage ratio
        if self.avg_prompt_tokens > 0:
            self.token_usage_ratio = self.avg_completion_tokens / self.avg_prompt_tokens

        # Issue #42 specific: Time per output token (in milliseconds)
        if self.avg_completion_tokens > 0:
            self.time_per_output_token = (
                self.avg_response_time * 1000
            ) / self.avg_completion_tokens

        self.avg_total_tokens = self.avg_prompt_tokens + self.avg_completion_tokens

    def to_dict(self) -> Dict[str, Any]:
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
            # Issue #42 key metrics
            "token_usage_ratio": round(self.token_usage_ratio, 4),
            "time_per_output_token_ms": round(self.time_per_output_token, 3),
        }


@dataclass
class ReasoningModeComparison:
    """Comparison between standard and reasoning modes."""

    dataset_name: str
    model_name: str
    timestamp: str

    standard_mode: ReasoningModeMetrics = None
    reasoning_mode: ReasoningModeMetrics = None

    # Category-level breakdown
    category_results: Dict[str, Dict[str, ReasoningModeMetrics]] = field(
        default_factory=dict
    )

    def get_improvement_summary(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reasoning Mode Evaluation Benchmark (Issue #42)"
    )

    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mmlu", "gpqa"],
        help="Datasets to evaluate (supports MMLU and non-MMLU). Use --list-datasets to see all.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    # Endpoint configuration
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="vLLM endpoint URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "1234")
        ),
        help="API key for endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to evaluate (if not specified, fetches from endpoint)",
    )

    # Sampling options
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=10,
        help="Number of questions to sample per category",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to evaluate",
    )

    # Execution options
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: dataset-optimal)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reasoning_mode_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate markdown report",
    )
    parser.add_argument(
        "--reasoning-family",
        type=str,
        default=None,
        help="Reasoning family identifier (e.g., 'deepseek', 'qwen3', 'gpt-oss') for vSR config generation",
    )

    return parser.parse_args()


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    """Get available models from endpoint."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "1234", timeout=30.0)
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def process_question(
    client: OpenAI,
    model: str,
    question: Question,
    dataset: Any,
    max_tokens: int,
    temperature: float,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    """Process a single question with specified reasoning mode."""
    # Build extra_body for reasoning control
    extra_body = build_extra_body_for_model(model, reasoning_enabled)

    # Format prompt (plain for both modes - reasoning is controlled via extra_body)
    prompt = dataset.format_prompt(question, "plain")

    start_time = time.time()
    response_text, success, prompt_tokens, completion_tokens, total_tokens = call_model(
        client, model, prompt, max_tokens, temperature, extra_body=extra_body
    )
    end_time = time.time()

    # Extract and evaluate answer
    predicted_answer = extract_answer(response_text, question) if success else None
    is_correct = False

    if predicted_answer:
        if hasattr(question, "options") and question.options:
            if len(question.options) == 2 and set(question.options) == {"Yes", "No"}:
                is_correct = predicted_answer == question.correct_answer
            elif predicted_answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if isinstance(question.correct_answer, str):
                    is_correct = predicted_answer == question.correct_answer
                elif isinstance(question.correct_answer, int):
                    predicted_idx = ord(predicted_answer) - ord("A")
                    is_correct = predicted_idx == question.correct_answer
        else:
            is_correct = compare_free_form_answers(
                predicted_answer, question.correct_answer
            )

    return {
        "question_id": question.question_id,
        "category": question.category,
        "correct_answer": question.correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "success": success,
        "response_time": end_time - start_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_enabled": reasoning_enabled,
    }


def evaluate_mode(
    questions: List[Question],
    dataset: Any,
    model: str,
    endpoint: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_enabled: bool,
    concurrent_requests: int,
    mode_name: str,
) -> Tuple[ReasoningModeMetrics, pd.DataFrame]:
    """Evaluate a single reasoning mode across all questions."""

    client = OpenAI(base_url=endpoint, api_key=api_key or "1234", timeout=300.0)

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(
                process_question,
                client,
                model,
                q,
                dataset,
                max_tokens,
                temperature,
                reasoning_enabled,
            )
            for q in questions
        ]

        try:
            for future in tqdm(
                futures, total=len(futures), desc=f"Evaluating {mode_name}"
            ):
                results.append(future.result())
        except KeyboardInterrupt:
            print(f"\n⚠️  {mode_name} evaluation interrupted. Saving partial results...")
            for future in futures:
                future.cancel()
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception:
                        pass

    # Create DataFrame
    df = pd.DataFrame(results)

    # Compute metrics
    metrics = ReasoningModeMetrics(mode_name=mode_name)
    metrics.total_questions = len(results)

    successful = [r for r in results if r["success"]]
    metrics.correct_answers = sum(1 for r in successful if r["is_correct"])
    metrics.failed_queries = len(results) - len(successful)

    for r in successful:
        if r["response_time"] is not None:
            metrics.response_times.append(r["response_time"])
        if r["prompt_tokens"] is not None:
            metrics.prompt_token_counts.append(r["prompt_tokens"])
        if r["completion_tokens"] is not None:
            metrics.completion_token_counts.append(r["completion_tokens"])

    metrics.compute_derived_metrics()

    return metrics, df


def evaluate_dataset(
    dataset_name: str,
    model: str,
    endpoint: str,
    api_key: str,
    samples_per_category: int,
    categories: Optional[List[str]],
    max_tokens: Optional[int],
    temperature: float,
    concurrent_requests: int,
    seed: int,
) -> ReasoningModeComparison:
    """Evaluate both standard and reasoning modes on a dataset."""

    print(f"\n{'='*60}")
    print(f"Evaluating Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = DatasetFactory.create_dataset(dataset_name)
    questions, dataset_info = dataset.load_dataset(
        categories=categories,
        samples_per_category=samples_per_category,
        seed=seed,
    )

    print(
        f"Loaded {len(questions)} questions across {len(dataset_info.categories)} categories"
    )

    # Determine optimal max_tokens
    effective_max_tokens = max_tokens or get_dataset_optimal_tokens(dataset_info, model)
    print(f"Using max_tokens: {effective_max_tokens}")

    # Create comparison object
    comparison = ReasoningModeComparison(
        dataset_name=dataset_info.name,
        model_name=model,
        timestamp=datetime.now().isoformat(),
    )

    # Evaluate Standard Mode (reasoning OFF)
    print(f"\n📊 Standard Mode (reasoning=False)")
    standard_metrics, standard_df = evaluate_mode(
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
    )
    comparison.standard_mode = standard_metrics

    # Evaluate Reasoning Mode (reasoning ON)
    print(f"\n🧠 Reasoning Mode (reasoning=True)")
    reasoning_metrics, reasoning_df = evaluate_mode(
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
    )
    comparison.reasoning_mode = reasoning_metrics

    # Compute per-category metrics
    for category in dataset_info.categories:
        cat_questions = [q for q in questions if q.category == category]
        if not cat_questions:
            continue

        # Standard mode for this category
        cat_standard = standard_df[standard_df["category"] == category]
        cat_standard_metrics = ReasoningModeMetrics(mode_name="standard")
        cat_standard_metrics.total_questions = len(cat_standard)
        cat_standard_metrics.correct_answers = cat_standard["is_correct"].sum()
        cat_standard_metrics.failed_queries = (~cat_standard["success"]).sum()

        for _, row in cat_standard[cat_standard["success"]].iterrows():
            if row["response_time"] is not None:
                cat_standard_metrics.response_times.append(row["response_time"])
            if row["prompt_tokens"] is not None:
                cat_standard_metrics.prompt_token_counts.append(row["prompt_tokens"])
            if row["completion_tokens"] is not None:
                cat_standard_metrics.completion_token_counts.append(
                    row["completion_tokens"]
                )
        cat_standard_metrics.compute_derived_metrics()

        # Reasoning mode for this category
        cat_reasoning = reasoning_df[reasoning_df["category"] == category]
        cat_reasoning_metrics = ReasoningModeMetrics(mode_name="reasoning")
        cat_reasoning_metrics.total_questions = len(cat_reasoning)
        cat_reasoning_metrics.correct_answers = cat_reasoning["is_correct"].sum()
        cat_reasoning_metrics.failed_queries = (~cat_reasoning["success"]).sum()

        for _, row in cat_reasoning[cat_reasoning["success"]].iterrows():
            if row["response_time"] is not None:
                cat_reasoning_metrics.response_times.append(row["response_time"])
            if row["prompt_tokens"] is not None:
                cat_reasoning_metrics.prompt_token_counts.append(row["prompt_tokens"])
            if row["completion_tokens"] is not None:
                cat_reasoning_metrics.completion_token_counts.append(
                    row["completion_tokens"]
                )
        cat_reasoning_metrics.compute_derived_metrics()

        comparison.category_results[category] = {
            "standard": cat_standard_metrics,
            "reasoning": cat_reasoning_metrics,
        }

    return comparison, standard_df, reasoning_df


def generate_comparison_plots(
    comparison: ReasoningModeComparison,
    output_dir: Path,
):
    """Generate visualization plots comparing standard vs reasoning modes."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Overall Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ("accuracy", "Accuracy", "higher is better"),
        (
            "token_usage_ratio",
            "Token Usage Ratio\n(completion/prompt)",
            "efficiency metric",
        ),
        ("time_per_output_token_ms", "Time per Output Token (ms)", "lower is better"),
        ("avg_response_time_sec", "Avg Response Time (s)", "lower is better"),
    ]

    colors = {"standard": "#3498db", "reasoning": "#e74c3c"}

    for ax, (metric_key, metric_label, note) in zip(axes.flatten(), metrics_to_plot):
        std_val = getattr(
            comparison.standard_mode,
            metric_key.replace("_ms", "").replace("_sec", ""),
            0,
        )
        reas_val = getattr(
            comparison.reasoning_mode,
            metric_key.replace("_ms", "").replace("_sec", ""),
            0,
        )

        # Handle special cases for display
        if "_ms" in metric_key:
            std_val = comparison.standard_mode.time_per_output_token
            reas_val = comparison.reasoning_mode.time_per_output_token
        elif "_sec" in metric_key:
            std_val = comparison.standard_mode.avg_response_time
            reas_val = comparison.reasoning_mode.avg_response_time

        bars = ax.bar(
            ["Standard", "Reasoning"],
            [std_val, reas_val],
            color=[colors["standard"], colors["reasoning"]],
            edgecolor="white",
            linewidth=2,
        )

        # Add value labels on bars
        for bar, val in zip(bars, [std_val, reas_val]):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_title(f"{metric_label}\n({note})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")

    fig.suptitle(
        f"Standard vs Reasoning Mode: {comparison.dataset_name}\nModel: {comparison.model_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{comparison.dataset_name}_overall_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Per-Category Accuracy Comparison
    if comparison.category_results:
        categories = list(comparison.category_results.keys())
        std_accs = [
            comparison.category_results[c]["standard"].accuracy for c in categories
        ]
        reas_accs = [
            comparison.category_results[c]["reasoning"].accuracy for c in categories
        ]

        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, std_accs, width, label="Standard", color=colors["standard"]
        )
        bars2 = ax.bar(
            x + width / 2,
            reas_accs,
            width,
            label="Reasoning",
            color=colors["reasoning"],
        )

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"Per-Category Accuracy: Standard vs Reasoning\n{comparison.dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{comparison.dataset_name}_category_accuracy.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 3. Token Efficiency Plot
    if comparison.category_results:
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))

        std_ratios = [
            comparison.category_results[c]["standard"].token_usage_ratio
            for c in categories
        ]
        reas_ratios = [
            comparison.category_results[c]["reasoning"].token_usage_ratio
            for c in categories
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(
            x - width / 2, std_ratios, width, label="Standard", color=colors["standard"]
        )
        ax.bar(
            x + width / 2,
            reas_ratios,
            width,
            label="Reasoning",
            color=colors["reasoning"],
        )

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Token Usage Ratio (completion/prompt)", fontsize=12)
        ax.set_title(
            f"Token Usage Ratio by Category\n{comparison.dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{comparison.dataset_name}_token_usage_ratio.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Time per Output Token Plot
    if comparison.category_results:
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))

        std_times = [
            comparison.category_results[c]["standard"].time_per_output_token
            for c in categories
        ]
        reas_times = [
            comparison.category_results[c]["reasoning"].time_per_output_token
            for c in categories
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(
            x - width / 2, std_times, width, label="Standard", color=colors["standard"]
        )
        ax.bar(
            x + width / 2,
            reas_times,
            width,
            label="Reasoning",
            color=colors["reasoning"],
        )

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel("Time per Output Token (ms)", fontsize=12)
        ax.set_title(
            f"Response Time per Output Token by Category\n{comparison.dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{comparison.dataset_name}_time_per_token.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print(f"  ✅ Plots saved to {output_dir}")


def generate_markdown_report(
    comparisons: List[ReasoningModeComparison],
    output_dir: Path,
):
    """Generate comprehensive markdown report."""

    report_path = output_dir / "REASONING_MODE_EVALUATION_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Reasoning Mode Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(
            "This report compares **Standard Mode** (reasoning OFF) vs **Reasoning Mode** (reasoning ON) "
        )
        f.write("across the following key metrics as specified in Issue #42:\n\n")
        f.write(
            "1. **Response Correctness (Accuracy)** - Percentage of correct answers\n"
        )
        f.write("2. **Token Usage Ratio** - `completion_tokens / prompt_tokens`\n")
        f.write(
            "3. **Response Time per Output Token** - Throughput efficiency metric\n\n"
        )

        # Overall summary table
        f.write("### Overall Results Summary\n\n")
        f.write(
            "| Dataset | Mode | Accuracy | Token Ratio | Time/Token (ms) | Avg Response (s) |\n"
        )
        f.write(
            "|---------|------|----------|-------------|-----------------|------------------|\n"
        )

        for comp in comparisons:
            std = comp.standard_mode
            reas = comp.reasoning_mode
            f.write(
                f"| {comp.dataset_name} | Standard | {std.accuracy:.4f} | {std.token_usage_ratio:.4f} | {std.time_per_output_token:.2f} | {std.avg_response_time:.2f} |\n"
            )
            f.write(
                f"| | Reasoning | {reas.accuracy:.4f} | {reas.token_usage_ratio:.4f} | {reas.time_per_output_token:.2f} | {reas.avg_response_time:.2f} |\n"
            )

        f.write("\n")

        # Improvement summary
        f.write("### Improvement Summary (Reasoning over Standard)\n\n")
        f.write(
            "| Dataset | Accuracy Δ | Accuracy % | Token Ratio Δ | Time/Token Δ (ms) |\n"
        )
        f.write(
            "|---------|------------|------------|---------------|-------------------|\n"
        )

        for comp in comparisons:
            imp = comp.get_improvement_summary()
            acc_emoji = (
                "✅"
                if imp.get("accuracy_delta", 0) > 0
                else "⬇️" if imp.get("accuracy_delta", 0) < 0 else "➖"
            )
            f.write(
                f"| {comp.dataset_name} | {acc_emoji} {imp.get('accuracy_delta', 0):+.4f} | {imp.get('accuracy_improvement_pct', 0):+.2f}% | {imp.get('token_usage_ratio_delta', 0):+.4f} | {imp.get('time_per_output_token_delta_ms', 0):+.2f} |\n"
            )

        f.write("\n---\n\n")

        # Detailed results per dataset
        for comp in comparisons:
            f.write(f"## {comp.dataset_name}\n\n")
            f.write(f"**Model:** {comp.model_name}\n\n")
            f.write(f"**Timestamp:** {comp.timestamp}\n\n")

            f.write("### Mode Comparison\n\n")
            f.write("| Metric | Standard | Reasoning | Delta |\n")
            f.write("|--------|----------|-----------|-------|\n")

            std = comp.standard_mode
            reas = comp.reasoning_mode

            f.write(
                f"| Accuracy | {std.accuracy:.4f} | {reas.accuracy:.4f} | {reas.accuracy - std.accuracy:+.4f} |\n"
            )
            f.write(
                f"| Token Usage Ratio | {std.token_usage_ratio:.4f} | {reas.token_usage_ratio:.4f} | {reas.token_usage_ratio - std.token_usage_ratio:+.4f} |\n"
            )
            f.write(
                f"| Time/Token (ms) | {std.time_per_output_token:.2f} | {reas.time_per_output_token:.2f} | {reas.time_per_output_token - std.time_per_output_token:+.2f} |\n"
            )
            f.write(
                f"| Avg Response (s) | {std.avg_response_time:.2f} | {reas.avg_response_time:.2f} | {reas.avg_response_time - std.avg_response_time:+.2f} |\n"
            )
            f.write(
                f"| Avg Completion Tokens | {std.avg_completion_tokens:.1f} | {reas.avg_completion_tokens:.1f} | {reas.avg_completion_tokens - std.avg_completion_tokens:+.1f} |\n"
            )
            f.write(
                f"| Avg Prompt Tokens | {std.avg_prompt_tokens:.1f} | {reas.avg_prompt_tokens:.1f} | {reas.avg_prompt_tokens - std.avg_prompt_tokens:+.1f} |\n"
            )

            f.write("\n")

            # Category breakdown
            if comp.category_results:
                f.write("### Per-Category Results\n\n")
                f.write(
                    "| Category | Std Acc | Reas Acc | Δ Acc | Std Token Ratio | Reas Token Ratio |\n"
                )
                f.write(
                    "|----------|---------|----------|-------|-----------------|------------------|\n"
                )

                for cat, modes in sorted(comp.category_results.items()):
                    std_cat = modes["standard"]
                    reas_cat = modes["reasoning"]
                    f.write(
                        f"| {cat} | {std_cat.accuracy:.4f} | {reas_cat.accuracy:.4f} | {reas_cat.accuracy - std_cat.accuracy:+.4f} | {std_cat.token_usage_ratio:.4f} | {reas_cat.token_usage_ratio:.4f} |\n"
                    )

                f.write("\n")

            f.write("---\n\n")

        # Methodology section
        f.write("## Methodology\n\n")
        f.write("### Evaluation Modes\n\n")
        f.write(
            "- **Standard Mode**: Plain prompt with `reasoning=False` in `extra_body`\n"
        )
        f.write(
            "- **Reasoning Mode**: Plain prompt with `reasoning=True` in `extra_body`\n\n"
        )

        f.write("### Key Metrics (Issue #42 Requirements)\n\n")
        f.write(
            "1. **Response Correctness**: Accuracy = correct_answers / total_questions\n"
        )
        f.write("2. **Token Usage Ratio**: completion_tokens / prompt_tokens\n")
        f.write(
            "3. **Time per Output Token**: (response_time × 1000) / completion_tokens (ms)\n\n"
        )

        f.write("### Model-Specific Reasoning Control\n\n")
        f.write("```python\n")
        f.write('# DeepSeek V3.1: {"chat_template_kwargs": {"thinking": True/False}}\n')
        f.write('# Qwen3: {"chat_template_kwargs": {"enable_thinking": True/False}}\n')
        f.write('# GPT-OSS: {"reasoning_effort": "high"/"low"}\n')
        f.write("```\n\n")

    print(f"  ✅ Report saved to {report_path}")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def generate_vsr_model_config(
    comparisons: List[ReasoningModeComparison],
    model_name: str,
    reasoning_family: str = None,
) -> Dict[str, Any]:
    """
    Generate vSR (vLLM Semantic Router) model configuration based on evaluation results.

    This function analyzes the reasoning mode evaluation results and generates
    a recommended model configuration for semantic-router's config.yaml.

    Args:
        comparisons: List of ReasoningModeComparison results
        model_name: Name of the model being evaluated
        reasoning_family: Reasoning family identifier (e.g., "deepseek", "qwen3", "gpt-oss")
                         If None, will be inferred from extra_body parameters

    Returns:
        Dictionary containing vSR model configuration
    """
    # Aggregate metrics across all datasets
    total_std_accuracy = sum(c.standard_mode.accuracy for c in comparisons) / len(
        comparisons
    )
    total_reas_accuracy = sum(c.reasoning_mode.accuracy for c in comparisons) / len(
        comparisons
    )

    avg_token_ratio_std = sum(
        c.standard_mode.token_usage_ratio for c in comparisons
    ) / len(comparisons)
    avg_token_ratio_reas = sum(
        c.reasoning_mode.token_usage_ratio for c in comparisons
    ) / len(comparisons)

    avg_time_per_token_std = sum(
        c.standard_mode.time_per_output_token for c in comparisons
    ) / len(comparisons)
    avg_time_per_token_reas = sum(
        c.reasoning_mode.time_per_output_token for c in comparisons
    ) / len(comparisons)

    # Determine reasoning family if not provided
    if not reasoning_family:
        # Try to infer from the first comparison's reasoning mode parameters
        # This is a placeholder - in practice, this should be passed explicitly
        reasoning_family = "auto"  # Default placeholder

    # Calculate improvement percentages
    accuracy_improvement = (
        ((total_reas_accuracy - total_std_accuracy) / total_std_accuracy * 100)
        if total_std_accuracy > 0
        else 0
    )
    token_overhead = (
        ((avg_token_ratio_reas - avg_token_ratio_std) / avg_token_ratio_std * 100)
        if avg_token_ratio_std > 0
        else 0
    )
    latency_overhead = (
        (
            (avg_time_per_token_reas - avg_time_per_token_std)
            / avg_time_per_token_std
            * 100
        )
        if avg_time_per_token_std > 0
        else 0
    )

    # Generate recommendation
    recommendation = {
        "model_name": model_name,
        "reasoning_family": reasoning_family,
        "performance_analysis": {
            "standard_mode": {
                "avg_accuracy": round(total_std_accuracy, 4),
                "avg_token_usage_ratio": round(avg_token_ratio_std, 4),
                "avg_time_per_output_token_ms": round(avg_time_per_token_std, 2),
            },
            "reasoning_mode": {
                "avg_accuracy": round(total_reas_accuracy, 4),
                "avg_token_usage_ratio": round(avg_token_ratio_reas, 4),
                "avg_time_per_output_token_ms": round(avg_time_per_token_reas, 2),
            },
            "improvements": {
                "accuracy_change_percent": round(accuracy_improvement, 2),
                "token_overhead_percent": round(token_overhead, 2),
                "latency_overhead_percent": round(latency_overhead, 2),
            },
        },
        "recommendation": _generate_recommendation_text(
            accuracy_improvement, token_overhead, latency_overhead
        ),
        "suggested_vsr_config": {
            "model_config": {
                model_name: {
                    "reasoning_family": (
                        reasoning_family
                        if reasoning_family != "auto"
                        else "# REPLACE: qwen3, deepseek, or gpt-oss"
                    ),
                    "# Note": "Add this to your config.yaml model_config section",
                }
            }
        },
    }

    return recommendation


def _generate_recommendation_text(
    accuracy_imp: float, token_overhead: float, latency_overhead: float
) -> str:
    """Generate human-readable recommendation based on metrics."""

    recommendations = []

    if accuracy_imp > 5:
        recommendations.append(
            f"✅ Reasoning mode shows {accuracy_imp:.1f}% accuracy improvement. "
            "Recommended for accuracy-critical applications."
        )
    elif accuracy_imp > 0:
        recommendations.append(
            f"⚖️ Reasoning mode shows modest {accuracy_imp:.1f}% accuracy improvement."
        )
    else:
        recommendations.append(
            f"⚠️ Reasoning mode shows {abs(accuracy_imp):.1f}% accuracy degradation. "
            "Standard mode may be preferable."
        )

    if token_overhead > 50:
        recommendations.append(
            f"💰 Reasoning mode has {token_overhead:.1f}% higher token usage. "
            "Consider cost implications."
        )
    elif token_overhead > 20:
        recommendations.append(
            f"📊 Reasoning mode has {token_overhead:.1f}% moderate token overhead."
        )

    if latency_overhead > 50:
        recommendations.append(
            f"⏱️ Reasoning mode has {latency_overhead:.1f}% higher latency per token. "
            "May impact real-time applications."
        )
    elif latency_overhead > 20:
        recommendations.append(
            f"⏱️ Reasoning mode has {latency_overhead:.1f}% moderate latency overhead."
        )

    if accuracy_imp > 5 and token_overhead < 30 and latency_overhead < 30:
        recommendations.append(
            "🎯 Overall: Reasoning mode offers good accuracy improvement with acceptable overhead. Recommended for production use."
        )
    elif accuracy_imp < 0:
        recommendations.append(
            "🎯 Overall: Standard mode is recommended based on these results."
        )
    else:
        recommendations.append(
            "🎯 Overall: Consider enabling reasoning mode selectively for complex queries or high-priority categories."
        )

    return "\n".join(recommendations)


def save_results(
    comparisons: List[ReasoningModeComparison],
    all_standard_dfs: List[pd.DataFrame],
    all_reasoning_dfs: List[pd.DataFrame],
    output_dir: Path,
    model_name: str = "evaluated-model",
    reasoning_family: str = None,
):
    """Save all results to files including vSR model configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate vSR model configuration
    vsr_config = generate_vsr_model_config(comparisons, model_name, reasoning_family)

    # Save vSR config recommendation
    with open(output_dir / "vsr_model_config_recommendation.json", "w") as f:
        json.dump(vsr_config, f, indent=2, cls=NumpyEncoder)

    # Save vSR config as YAML snippet
    import yaml

    with open(output_dir / "vsr_model_config.yaml", "w") as f:
        yaml.dump(
            vsr_config["suggested_vsr_config"],
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"\n📝 vSR Model Config generated:")
    print(f"   - JSON: {output_dir / 'vsr_model_config_recommendation.json'}")
    print(f"   - YAML: {output_dir / 'vsr_model_config.yaml'}")
    print(f"\n{vsr_config['recommendation']}")

    # Save JSON summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "issue": "42",
        "title": "Reasoning Mode Evaluation",
        "comparisons": [c.to_dict() for c in comparisons],
        "vsr_config_recommendation": vsr_config,
    }

    with open(output_dir / "reasoning_mode_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Save detailed CSVs
    for comp, std_df, reas_df in zip(comparisons, all_standard_dfs, all_reasoning_dfs):
        dataset_dir = output_dir / comp.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        std_df["mode"] = "standard"
        reas_df["mode"] = "reasoning"

        combined_df = pd.concat([std_df, reas_df], ignore_index=True)
        combined_df.to_csv(dataset_dir / "detailed_results.csv", index=False)

        std_df.to_csv(dataset_dir / "standard_mode_results.csv", index=False)
        reas_df.to_csv(dataset_dir / "reasoning_mode_results.csv", index=False)

    print(f"\n✅ Results saved to {output_dir}")


def main():
    """Main entry point for reasoning mode evaluation."""
    args = parse_args()

    # List datasets if requested
    if args.list_datasets:
        list_available_datasets()
        return

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get model
    model = args.model
    if not model:
        print("Fetching available models from endpoint...")
        models = get_available_models(args.endpoint, args.api_key)
        if models:
            model = models[0]
            print(f"Using model: {model}")
        else:
            print("❌ No models available. Please specify --model")
            return

    print("\n" + "=" * 60)
    print("🧠 REASONING MODE EVALUATION BENCHMARK")
    print("=" * 60)
    print(f"Issue #42: Standard vs Reasoning Mode Comparison")
    print(f"Model: {model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Samples per category: {args.samples_per_category}")
    print(f"Endpoint: {args.endpoint}")
    print("=" * 60 + "\n")

    # Run evaluations
    comparisons = []
    all_standard_dfs = []
    all_reasoning_dfs = []

    for dataset_name in args.datasets:
        try:
            comparison, std_df, reas_df = evaluate_dataset(
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
            comparisons.append(comparison)
            all_standard_dfs.append(std_df)
            all_reasoning_dfs.append(reas_df)

            # Print summary for this dataset
            print(f"\n📊 {dataset_name} Summary:")
            print(f"  Standard Accuracy:  {comparison.standard_mode.accuracy:.4f}")
            print(f"  Reasoning Accuracy: {comparison.reasoning_mode.accuracy:.4f}")
            print(
                f"  Accuracy Delta:     {comparison.reasoning_mode.accuracy - comparison.standard_mode.accuracy:+.4f}"
            )
            print(
                f"  Token Ratio (Std):  {comparison.standard_mode.token_usage_ratio:.4f}"
            )
            print(
                f"  Token Ratio (Reas): {comparison.reasoning_mode.token_usage_ratio:.4f}"
            )
            print(
                f"  Time/Token (Std):   {comparison.standard_mode.time_per_output_token:.2f} ms"
            )
            print(
                f"  Time/Token (Reas):  {comparison.reasoning_mode.time_per_output_token:.2f} ms"
            )

        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not comparisons:
        print("❌ No successful evaluations. Exiting.")
        return

    # Save results
    output_dir = Path(args.output_dir)
    save_results(
        comparisons,
        all_standard_dfs,
        all_reasoning_dfs,
        output_dir,
        model_name=model,
        reasoning_family=args.reasoning_family,
    )

    # Generate plots
    if args.generate_plots:
        print("\n📈 Generating comparison plots...")
        for comp in comparisons:
            generate_comparison_plots(comp, output_dir / "plots")

    # Generate report
    if args.generate_report:
        print("\n📝 Generating markdown report...")
        generate_markdown_report(comparisons, output_dir)

    print("\n" + "=" * 60)
    print("✅ REASONING MODE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
