from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def generate_markdown_report(comparisons: list[Any], output_dir: Path) -> None:
    """Generate a comprehensive markdown report."""
    report_path = output_dir / "REASONING_MODE_EVALUATION_REPORT.md"
    with open(report_path, "w") as report_file:
        _write_header(report_file)
        _write_overall_summary(report_file, comparisons)
        _write_improvement_summary(report_file, comparisons)
        for comparison in comparisons:
            _write_dataset_section(report_file, comparison)
        _write_methodology(report_file)
    print(f"  ✅ Report saved to {report_path}")


def _write_header(report_file: Any) -> None:
    report_file.write("# Reasoning Mode Evaluation Report\n\n")
    report_file.write(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    report_file.write("---\n\n")
    report_file.write("## Executive Summary\n\n")
    report_file.write(
        "This report compares **Standard Mode** (reasoning OFF) vs **Reasoning Mode** "
        "(reasoning ON) across the Issue #42 metrics.\n\n"
    )
    report_file.write(
        "1. **Response Correctness (Accuracy)** - Percentage of correct answers\n"
    )
    report_file.write(
        "2. **Token Usage Ratio** - `completion_tokens / prompt_tokens`\n"
    )
    report_file.write(
        "3. **Response Time per Output Token** - Throughput efficiency metric\n\n"
    )


def _write_overall_summary(report_file: Any, comparisons: list[Any]) -> None:
    report_file.write("### Overall Results Summary\n\n")
    report_file.write(
        "| Dataset | Mode | Accuracy | Token Ratio | Time/Token (ms) | Avg Response (s) |\n"
    )
    report_file.write(
        "|---------|------|----------|-------------|-----------------|------------------|\n"
    )
    for comparison in comparisons:
        _write_mode_row(
            report_file, comparison.dataset_name, "Standard", comparison.standard_mode
        )
        _write_mode_row(report_file, "", "Reasoning", comparison.reasoning_mode)
    report_file.write("\n")


def _write_mode_row(
    report_file: Any, dataset_name: str, label: str, metrics: Any
) -> None:
    report_file.write(
        f"| {dataset_name} | {label} | {metrics.accuracy:.4f} | {metrics.token_usage_ratio:.4f} | "
        f"{metrics.time_per_output_token:.2f} | {metrics.avg_response_time:.2f} |\n"
    )


def _write_improvement_summary(report_file: Any, comparisons: list[Any]) -> None:
    report_file.write("### Improvement Summary (Reasoning over Standard)\n\n")
    report_file.write(
        "| Dataset | Accuracy Delta | Accuracy % | Token Ratio Delta | Time/Token Delta (ms) |\n"
    )
    report_file.write(
        "|---------|----------------|------------|-------------------|-----------------------|\n"
    )
    for comparison in comparisons:
        improvement = comparison.get_improvement_summary()
        report_file.write(
            f"| {comparison.dataset_name} | {improvement.get('accuracy_delta', 0):+.4f} | "
            f"{improvement.get('accuracy_improvement_pct', 0):+.2f}% | "
            f"{improvement.get('token_usage_ratio_delta', 0):+.4f} | "
            f"{improvement.get('time_per_output_token_delta_ms', 0):+.2f} |\n"
        )
    report_file.write("\n---\n\n")


def _write_dataset_section(report_file: Any, comparison: Any) -> None:
    report_file.write(f"## {comparison.dataset_name}\n\n")
    report_file.write(f"**Model:** {comparison.model_name}\n\n")
    report_file.write(f"**Timestamp:** {comparison.timestamp}\n\n")
    _write_dataset_metrics(report_file, comparison)
    if comparison.category_results:
        _write_category_breakdown(report_file, comparison)
    report_file.write("---\n\n")


def _write_dataset_metrics(report_file: Any, comparison: Any) -> None:
    std = comparison.standard_mode
    reasoning = comparison.reasoning_mode
    rows = [
        ("Accuracy", std.accuracy, reasoning.accuracy),
        ("Token Usage Ratio", std.token_usage_ratio, reasoning.token_usage_ratio),
        ("Time/Token (ms)", std.time_per_output_token, reasoning.time_per_output_token),
        ("Avg Response (s)", std.avg_response_time, reasoning.avg_response_time),
        (
            "Avg Completion Tokens",
            std.avg_completion_tokens,
            reasoning.avg_completion_tokens,
        ),
        ("Avg Prompt Tokens", std.avg_prompt_tokens, reasoning.avg_prompt_tokens),
    ]

    report_file.write("### Mode Comparison\n\n")
    report_file.write("| Metric | Standard | Reasoning | Delta |\n")
    report_file.write("|--------|----------|-----------|-------|\n")
    for label, standard_value, reasoning_value in rows:
        report_file.write(
            f"| {label} | {standard_value:.4f} | {reasoning_value:.4f} | "
            f"{reasoning_value - standard_value:+.4f} |\n"
        )
    report_file.write("\n")


def _write_category_breakdown(report_file: Any, comparison: Any) -> None:
    report_file.write("### Per-Category Results\n\n")
    report_file.write(
        "| Category | Std Acc | Reas Acc | Delta Acc | Std Token Ratio | Reas Token Ratio |\n"
    )
    report_file.write(
        "|----------|---------|----------|-----------|-----------------|------------------|\n"
    )
    for category, modes in sorted(comparison.category_results.items()):
        standard_metrics = modes["standard"]
        reasoning_metrics = modes["reasoning"]
        report_file.write(
            f"| {category} | {standard_metrics.accuracy:.4f} | {reasoning_metrics.accuracy:.4f} | "
            f"{reasoning_metrics.accuracy - standard_metrics.accuracy:+.4f} | "
            f"{standard_metrics.token_usage_ratio:.4f} | {reasoning_metrics.token_usage_ratio:.4f} |\n"
        )
    report_file.write("\n")


def _write_methodology(report_file: Any) -> None:
    report_file.write("## Methodology\n\n")
    report_file.write("### Evaluation Modes\n\n")
    report_file.write(
        "- **Standard Mode**: Plain prompt with `reasoning=False` in `extra_body`\n"
    )
    report_file.write(
        "- **Reasoning Mode**: Plain prompt with `reasoning=True` in `extra_body`\n\n"
    )
    report_file.write("### Key Metrics (Issue #42 Requirements)\n\n")
    report_file.write(
        "1. **Response Correctness**: Accuracy = correct_answers / total_questions\n"
    )
    report_file.write("2. **Token Usage Ratio**: completion_tokens / prompt_tokens\n")
    report_file.write(
        "3. **Time per Output Token**: (response_time x 1000) / completion_tokens (ms)\n\n"
    )
    report_file.write("### Model-Specific Reasoning Control\n\n")
    report_file.write("```python\n")
    report_file.write(
        '# DeepSeek V3.1: {"chat_template_kwargs": {"thinking": True/False}}\n'
    )
    report_file.write(
        '# Qwen3: {"chat_template_kwargs": {"enable_thinking": True/False}}\n'
    )
    report_file.write('# GPT-OSS: {"reasoning_effort": "high"/"low"}\n')
    report_file.write("```\n\n")
