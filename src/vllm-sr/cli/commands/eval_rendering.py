"""Terminal rendering for evaluation and Router Learning results."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from cli.terminal import echo, fields, heading, output_width, success


def render_eval_summary(summary: str) -> None:
    """Render an eval summary without changing its schema-tolerant contents."""
    success("Evaluation complete")
    echo()
    heading("Result")
    _render_summary_lines(summary)


def _render_summary_lines(summary: str) -> None:
    """Render summary text as wrapped fields, sections, and bullets."""
    if summary.lstrip().startswith(("{", "[")):
        _render_indented_json(summary)
        return

    pending_fields: list[tuple[str, str]] = []

    def flush_fields() -> None:
        if pending_fields:
            fields(tuple(pending_fields))
            pending_fields.clear()

    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            flush_fields()
            fields((("•", line[2:]),))
            continue
        if line.endswith(":"):
            flush_fields()
            heading(line[:-1].replace("_", " ").title(), indent=2)
            continue
        if ": " in line:
            label, value = line.split(": ", 1)
            pending_fields.append((label.replace("_", " ").title(), value))
            continue
        flush_fields()
        fields((("Response", line),))

    flush_fields()


def _render_indented_json(summary: str) -> None:
    """Keep fallback JSON readable without exceeding a narrow terminal."""
    width = output_width()
    for source_line in summary.splitlines():
        content = source_line.lstrip()
        indent = min(2 + len(source_line) - len(content), max(0, width - 1))
        prefix = " " * indent
        lines = textwrap.wrap(
            content,
            width=max(1, width - indent),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for line in lines:
            echo(f"{prefix}{line}")


def render_recipe_learning_artifact(
    artifact: dict[str, Any], output_dir: Path | None = None
) -> None:
    """Render the human-facing recipe analysis with wrapped terminal fields."""
    metrics = artifact["metrics"]["overall"]
    findings = artifact["findings"]

    success("Router Learning recipe analysis complete")
    echo()
    heading("Summary")
    fields(
        (
            ("Records", artifact["metrics"]["records"]),
            ("Learning coverage", f"{metrics['learning_coverage']:.2f}"),
            ("Outcome coverage", f"{metrics['outcome_coverage']:.2f}"),
            ("Switch rate", f"{metrics['switch_rate']:.2f}"),
            ("Cache preservation", f"{metrics['cache_preservation']:.2f}"),
            ("Cost savings", f"{metrics['cost_savings']:.6f}"),
            ("Findings", len(findings)),
            ("Candidate recipes", len(artifact["candidate_recipes"])),
        )
    )

    if findings:
        echo()
        heading("Top findings")
        fields(
            (
                str(index),
                f"[{item['severity']}] {item['decision']}: "
                f"{item['type']} - {item['message']}",
            )
            for index, item in enumerate(findings[:5], 1)
        )

    if output_dir is not None:
        echo()
        heading("Artifacts")
        fields((("Directory", output_dir),))
