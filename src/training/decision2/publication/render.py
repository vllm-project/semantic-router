"""Deterministic, dependency-free model-card tables and SVG figures."""

from __future__ import annotations

from html import escape
from typing import Any

from benchmark.generate import FINAL_FAMILIES

COLORS = {
    "decision2": "#315bff",
    "decision1": "#738299",
    "open": "#f46b50",
    "hosted": "#9b55c7",
    "other": "#43536e",
}
TYPE_NAMES = ("choice", "noul", "score")
FAMILY_LABELS = ("Constraint", "Exception", "Evidence", "Resource")
TYPE_LABELS = ("Choice", "Noul", "Score")


def ordered_models(data: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        data["models"],
        key=lambda model: (
            -model["benchmark"]["macro_family_accuracy"],
            model["label"].casefold(),
        ),
    )


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def points(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def md(value: Any) -> str:
    return (
        escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _row(values: list[Any]) -> str:
    return "| " + " | ".join(md(value) for value in values) + " |"


def render_markdown(data: dict[str, Any]) -> str:
    ordered = ordered_models(data)
    lines = [
        f"# {md(data['title'])}",
        "",
        "Scores below come from the same frozen **final** typed-decision gold file. "
        "The headline is the unweighted mean of accuracy across four final families; "
        "invalid and missing predictions count as incorrect. Rank is within the models supplied to this generator.",
        "",
        f"Gold SHA-256: `{data['gold_sha256']}` · {data['items']} items · {data['questions']} questions.",
        "",
        "## Frozen benchmark ranking",
        "",
        _row(
            [
                "Rank",
                "Model",
                "Group",
                "Size",
                "Family macro ↑",
                "Overall accuracy ↑",
                "Invalid / missing",
                "Brier ↓ (covered)",
            ]
        ),
        _row(["---:", "---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for model in ordered:
        report = model["benchmark"]
        overall = report["overall"]
        brier = overall["brier"]
        brier_n = overall.get("probability_n", 0)
        brier_text = f"{brier:.4f} (n={brier_n})" if brier is not None else "—"
        rank = 1 + sum(
            other["benchmark"]["macro_family_accuracy"]
            > report["macro_family_accuracy"]
            for other in ordered
        )
        lines.append(
            _row(
                [
                    rank,
                    model["label"],
                    model["group"],
                    model.get("size", "—"),
                    percent(report["macro_family_accuracy"]),
                    percent(overall["accuracy_all"]),
                    f"{overall['invalid_or_missing_n']}/{overall['n']}",
                    brier_text,
                ]
            )
        )
    lines.extend(
        [
            "",
            "Brier is reported only over valid predictions that include probabilities; its coverage is shown in parentheses. "
            "The frozen benchmark report does not provide a confidence interval, so no synthetic-benchmark interval is shown.",
            "",
            "## Accuracy by final family and native task type",
            "",
            _row(["Model", *FAMILY_LABELS, *TYPE_LABELS]),
            _row(["---", *(["---:"] * (len(FINAL_FAMILIES) + len(TYPE_NAMES)))]),
        ]
    )
    for model in ordered:
        report = model["benchmark"]
        lines.append(
            _row(
                [
                    model["label"],
                    *(
                        percent(report["by_family"][name]["accuracy_all"])
                        for name in FINAL_FAMILIES
                    ),
                    *(
                        percent(report["by_type"][name]["accuracy_all"])
                        for name in TYPE_NAMES
                    ),
                ]
            )
        )
    lines.extend(
        [
            "",
            "Family columns and task-type columns are two views of the same final questions; do not add them together.",
            "",
        ]
    )

    if data["pairs"]:
        lines.extend(
            [
                "## Configured comparisons",
                "",
                _row(
                    [
                        "New model",
                        "Reference",
                        "Family macro difference",
                        "Overall accuracy difference",
                    ]
                ),
                _row(["---", "---", "---:", "---:"]),
            ]
        )
        for pair in data["pairs"]:
            new, old = pair["new"], pair["old"]
            lines.append(
                _row(
                    [
                        new["label"],
                        old["label"],
                        points(
                            new["benchmark"]["macro_family_accuracy"]
                            - old["benchmark"]["macro_family_accuracy"]
                        ),
                        points(
                            new["benchmark"]["overall"]["accuracy_all"]
                            - old["benchmark"]["overall"]["accuracy_all"]
                        ),
                    ]
                )
            )
        lines.extend(
            [
                "",
                "Differences use the same frozen gold file and are descriptive point estimates; no benchmark interval is available.",
                "",
            ]
        )

    css_models = [model for model in ordered if model["css"] is not None]
    if css_models:
        lines.extend(
            [
                "## CSS human-label transfer",
                "",
                "These results use the separate CSS panel: the median over 15 evaluation tasks. "
                "Three pilot tasks are excluded; missing and invalid predictions count as misses. "
                "All CSS panel tasks are Choice classification, so this panel does not independently test Noul or Score.",
                "",
                f"CSS gold SHA-256: `{data['css_gold_sha256']}`.",
                "",
                _row(["Model", "Median task macro-F1 ↑", "Median task accuracy ↑"]),
                _row(["---", "---:", "---:"]),
            ]
        )
        for model in css_models:
            role = model["css"]["roles"]["evaluation"]
            lines.append(
                _row(
                    [
                        model["label"],
                        percent(role["median_task_macro_f1_all"]),
                        percent(role["median_task_accuracy_all"]),
                    ]
                )
            )
        lines.append("")
        paired = [pair for pair in data["pairs"] if pair["css_comparison"] is not None]
        if paired:
            lines.extend(
                [
                    "### Paired item bootstrap comparisons",
                    "",
                    _row(
                        [
                            "New model",
                            "Reference",
                            "Median macro-F1 difference (95% interval)",
                            "Median accuracy difference (95% interval)",
                        ]
                    ),
                    _row(["---", "---", "---:", "---:"]),
                ]
            )
            for pair in paired:
                metrics = pair["css_comparison"]["evaluation_median_over_15_tasks"]
                formatted = []
                for name in ("macro_f1_all", "accuracy_all"):
                    summary = metrics[name]
                    interval = summary["difference_interval95"]
                    formatted.append(
                        f"{points(summary['difference_a_minus_b'])} "
                        f"[{points(interval['low'])}, {points(interval['high'])}]"
                    )
                lines.append(
                    _row([pair["new"]["label"], pair["old"]["label"], *formatted])
                )
            lines.extend(
                [
                    "",
                    "Intervals jointly resample item IDs within each CSS evaluation task. "
                    "They apply only to these supplied CSS prediction pairs, not to the synthetic benchmark.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Provenance",
            "",
            "Use `manifest.json` for full input report and prediction SHA-256 digests. "
            "Display names, group, and size are metadata supplied in the configuration; scores and revisions come from scorer reports.",
            "",
            _row(["Model", "Scored model ID", "Revision", "Backend", "Report SHA-256"]),
            _row(["---", "---", "---", "---", "---"]),
        ]
    )
    for model in ordered:
        identity = model["benchmark"]["model"]
        lines.append(
            _row(
                [
                    model["label"],
                    identity["id"],
                    identity["revision"],
                    identity["backend"],
                    model["benchmark_file_sha256"],
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _svg_open(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 14,
    color: str = "#202b36",
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-family="system-ui, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def render_ranking_svg(data: dict[str, Any]) -> str:
    ordered = ordered_models(data)
    height = 170 + 49 * len(ordered)
    parts = _svg_open(
        1040,
        height,
        "Frozen final benchmark ranking",
        "Unweighted mean accuracy across four final families. Invalid and missing predictions count as incorrect. No confidence intervals are available.",
    )
    parts.append(
        _svg_text(32, 42, "Frozen final benchmark ranking", size=24, weight=700)
    )
    parts.append(
        _svg_text(
            32,
            69,
            "Unweighted family macro accuracy · all questions · descriptive point estimates",
            size=13,
            color="#51606b",
        )
    )
    x0, span = 340, 570
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x = x0 + tick * span
        parts.append(
            f'<line x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="{height - 58}" stroke="#e3e8ed"/>'
        )
        parts.append(
            _svg_text(
                x, 101, f"{tick * 100:.0f}%", size=11, color="#61707e", anchor="middle"
            )
        )
    for index, model in enumerate(ordered):
        score = model["benchmark"]["macro_family_accuracy"]
        y = 120 + index * 49
        color = COLORS[model["group"]]
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span}" height="25" rx="4" fill="#f1f4f7"/>'
        )
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span * score:.2f}" height="25" rx="4" fill="{color}"/>'
        )
        parts.append(_svg_text(32, y + 18, model["label"], size=15, weight=600))
        parts.append(
            _svg_text(995, y + 18, percent(score), size=14, anchor="end", weight=700)
        )
    parts.append(
        _svg_text(
            32,
            height - 25,
            f"Gold SHA-256 {data['gold_sha256'][:16]}… · {data['questions']} questions · rank among displayed models only",
            size=12,
            color="#61707e",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _cell_color(value: float) -> tuple[str, str]:
    # A single monotone palette keeps all family/type cells comparable.
    start, end = (239, 245, 243), (23, 107, 91)
    color = tuple(round(a * (1 - value) + b * value) for a, b in zip(start, end))
    fill = "#" + "".join(f"{channel:02x}" for channel in color)
    return fill, "#ffffff" if value >= 0.67 else "#17312c"


def render_matrix_svg(data: dict[str, Any]) -> str:
    ordered = ordered_models(data)
    height = 216 + 48 * len(ordered)
    parts = _svg_open(
        1110,
        height,
        "Frozen final benchmark accuracy matrix",
        "Each cell is accuracy over all questions in that final family or native task type. Family and type columns overlap. Invalid and missing answers count as incorrect.",
    )
    parts.append(
        _svg_text(32, 42, "Frozen final benchmark accuracy matrix", size=24, weight=700)
    )
    parts.append(_svg_text(32, 69, "Family slices", size=13, color="#51606b"))
    parts.append(_svg_text(790, 69, "Native task types", size=13, color="#51606b"))
    columns = [
        ("family", name, label) for name, label in zip(FINAL_FAMILIES, FAMILY_LABELS)
    ]
    columns += [("type", name, label) for name, label in zip(TYPE_NAMES, TYPE_LABELS)]
    x0, width, gap = 308, 102, 8
    for index, (_, name, label) in enumerate(columns):
        x = x0 + index * (width + gap)
        parts.append(
            _svg_text(x + width / 2, 108, label, size=12, anchor="middle", weight=600)
        )
    for row_index, model in enumerate(ordered):
        y = 126 + 48 * row_index
        parts.append(_svg_text(32, y + 23, model["label"], size=14, weight=600))
        for column_index, (kind, name, _) in enumerate(columns):
            score = model["benchmark"]["by_family" if kind == "family" else "by_type"][
                name
            ]["accuracy_all"]
            fill, ink = _cell_color(score)
            x = x0 + column_index * (width + gap)
            parts.append(
                f'<g><title>{escape(model["label"])} · {escape(name)}: {percent(score)}</title>'
                f'<rect x="{x}" y="{y}" width="{width}" height="34" rx="4" fill="{fill}"/>'
                f'{_svg_text(x + width / 2, y + 22, percent(score), size=12, color=ink, anchor="middle", weight=600)}</g>'
            )
    parts.append(
        _svg_text(
            32,
            height - 61,
            "Accuracy all (%) · missing and invalid count as misses · family/type views overlap",
            size=12,
            color="#51606b",
        )
    )
    parts.append(
        _svg_text(
            32,
            height - 38,
            f"Gold SHA-256 {data['gold_sha256'][:16]}… · {data['questions']} questions",
            size=12,
            color="#61707e",
        )
    )
    parts.append("</svg>")
    return "\n".join(part for part in parts if part) + "\n"
