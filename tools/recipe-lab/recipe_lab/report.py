"""Run report rendered only from the ledger (Sub-issue 4).

The ledger is self-sufficient: every live record carries its gates and
bench_refs, so the report replays the whole improvement history without
touching sr-bench or any run directory. A replay-consistency check asserts
that each live record's evidence is complete enough to have produced its
verdict - if the ledger ever drifts from its own verdicts, the report says
so instead of papering over it.
"""

from __future__ import annotations

import json
from typing import Any

from .ledger import Ledger

_MIN_LIVE_BENCH_REFS = 2  # a decisive live record must carry at least two bench refs

_HONESTY_FOOTER = (
    "Replayed from the recipe-lab ledger only. Metrics are descriptive; "
    "statistical significance is not established. A PASS is a recommendation, "
    "not a promotion."
)


def _cell(text: Any) -> str:
    """Markdown table cell: a literal pipe would split the column."""
    return str(text).replace("|", "\\|")


def _metrics_line(metrics: dict[str, Any]) -> str:
    # Units belong to the metric, not to a magnitude guess: a 0.4 s latency
    # and a 0.4 ratio both render as recorded, so a small value is never
    # silently rewritten as a percentage.
    if not metrics:
        return "n/a"
    return ", ".join(
        f"{key}={'n/a' if value is None else value}" for key, value in metrics.items()
    )


def _reasons_text(record: dict[str, Any]) -> str:
    """The ledger's own reasons, so an INCONCLUSIVE row keeps its cause."""
    reasons = record.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return "; ".join(str(reason) for reason in reasons) if reasons else "-"


def render_run_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# recipe-lab run report",
        "",
        "| record | round | tier | verdict | metrics | gates | bench_refs | reasons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    inconsistencies: list[str] = []
    hashes = sorted({r.get("dataset_hash") for r in records if r.get("dataset_hash")})
    for record in records:
        # None means guard results are missing; {} means the spec declared no
        # guards, which is a valid PASS shape and not an inconsistency.
        gates = record.get("gates")
        gates_text = (
            "-" if not gates else ", ".join(f"{k}={v}" for k, v in gates.items())
        )
        refs = record.get("bench_refs") or []
        lines.append(
            f"| {record.get('record_id', '-')} | {record.get('round')} | {record.get('tier')} | **{record.get('verdict')}** "
            f"| {_cell(_metrics_line(record.get('metrics') or {}))} | {_cell(gates_text)} "
            f"| {len(refs)} ref(s) | {_cell(_reasons_text(record))} |"
        )
        if record.get("tier") == "live" and record.get("verdict") != "INCONCLUSIVE":
            # replay-consistency: a decisive live record must carry complete
            # evidence (a gates dict - possibly empty - and two bench refs)
            if gates is None:
                inconsistencies.append(
                    f"round {record.get('round')}: live {record.get('verdict')} without gate results"
                )
            if len(refs) < _MIN_LIVE_BENCH_REFS:
                inconsistencies.append(
                    f"round {record.get('round')}: live {record.get('verdict')} with fewer than two bench_refs"
                )
        if record.get("coverage_warning"):
            lines.append(
                f"| - | {record.get('round')} | note | coverage_warning | evidence incomplete for some cases | - | - |"
            )
    if len(hashes) > 1:
        lines += [
            "",
            "## Cross-dataset warning",
            "",
            f"This ledger mixes {len(hashes)} dataset hashes; metrics across those",
            "rounds are NOT directly comparable:",
            "",
        ]
        lines += [f"- {h}" for h in hashes]
    if inconsistencies:
        lines += ["", "## Replay inconsistencies", ""]
        lines += [f"- {item}" for item in inconsistencies]
    lines += ["", "---", "", _HONESTY_FOOTER, ""]
    return "\n".join(lines)


def write_run_report(ledger_path: str, output_path: str) -> str:
    ledger = Ledger(ledger_path)
    report = render_run_report(ledger.read())
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return report


def render_promotion_draft(record: dict[str, Any]) -> str:
    """A docs/records draft for a human to review before promotion.

    Promotion stays a human decision informed by the Standard holdout; this
    draft only assembles the evidence the ledger already holds.
    """
    candidate = record.get("candidate") or {}
    provenance = candidate.get("provenance") or {}
    metrics = record.get("metrics") or {}
    gates = record.get("gates") or {}
    lines = [
        "# Recipe promotion record (DRAFT - requires human review + Standard holdout)",
        "",
        f"- record: `{record.get('record_id', 'unrecorded')}`",
        f"- candidate ref: `{candidate.get('ref')}` ({candidate.get('kind')})",
        f"- proposer: {provenance.get('proposer')} insight_refs={provenance.get('insight_refs')}",
        f"- dataset: {record.get('dataset_hash')}",
        f"- live metrics: {json.dumps(metrics, ensure_ascii=False)}",
        f"- guard gates: {json.dumps(gates, ensure_ascii=False) or 'none declared'}",
        f"- bench evidence: {', '.join(record.get('bench_refs') or [])}",
        f"- ledger reasons: {record.get('reasons')}",
        "",
        "## Human checklist",
        "",
        "- [ ] Standard holdout executed and clean (never tuned on)",
        "- [ ] Guard datasets reviewed; no PII / jailbreak regression",
        "- [ ] Previous recipe preserved for recovery (`vllm-sr config plan/apply`)",
        "- [ ] Promotion approved by a workgroup owner",
        "",
    ]
    return "\n".join(lines)
