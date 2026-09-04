"""Build deterministic public reports from normalized evidence."""

from __future__ import annotations

from html import escape

from cli.evaluation.architecture_feedback import architecture_recommendations
from cli.evaluation.contracts import RunManifest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import track_evidence_level
from cli.evaluation.metric_core import aggregate_track_coverage
from cli.evaluation.metrics import coverage
from cli.evaluation.reporting import (
    EvaluationArtifact,
    EvaluationCostAmount,
    EvaluationCostLedgers,
    EvaluationGate,
    EvaluationMetric,
    EvaluationProvenance,
    EvaluationReport,
    EvaluationReportSummary,
    EvaluationRun,
    EvaluationTrackReport,
)


def _value(metrics: list[EvaluationMetric], metric_id: str) -> float | None:
    metric = next((row for row in metrics if row.id == metric_id), None)
    return metric.value if metric else None


def _sum_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def select_report_metrics(
    manifest: RunManifest, metrics: list[EvaluationMetric]
) -> list[EvaluationMetric]:
    """Project computed metrics onto the immutable run track selection."""

    return [row for row in metrics if row.track_id in manifest.track_ids]


def build_costs(records: list[ExecutionRecord]) -> EvaluationCostLedgers:
    return EvaluationCostLedgers(
        runtime=EvaluationCostAmount(
            amount=_sum_optional([row.runtime_cost for row in records]),
            currency="USD",
            input_tokens=sum(row.input_tokens or 0 for row in records),
            output_tokens=sum(row.output_tokens or 0 for row in records),
        ),
        evaluation_overhead=EvaluationCostAmount(
            amount=_sum_optional([row.evaluation_cost for row in records]),
            currency="USD",
        ),
        capacity_tco=EvaluationCostAmount(
            amount=_sum_optional([row.capacity_tco for row in records]),
            currency="USD",
            gpu_seconds=_sum_optional([row.gpu_seconds for row in records]),
            energy_kwh=_sum_optional([row.energy_kwh for row in records]),
        ),
    )


def _verdict(gates: list[EvaluationGate]) -> str:
    required = [gate for gate in gates if gate.disposition == "required"]
    if any(gate.verdict == "fail" for gate in required):
        return "fail"
    if any(gate.verdict == "unavailable" for gate in required):
        return "unavailable"
    return "pass"


def _track_reports(
    manifest: RunManifest,
    records: list[ExecutionRecord],
    metrics: list[EvaluationMetric],
    gates: list[EvaluationGate],
    totals: dict[str, int],
) -> tuple[EvaluationTrackReport, ...]:
    reports: list[EvaluationTrackReport] = []
    for track_id in manifest.track_ids:
        track_records = [row for row in records if row.track_id == track_id]
        available = [row for row in track_records if row.status != "unavailable"]
        track_metrics = tuple(row for row in metrics if row.track_id == track_id)
        track_gates = tuple(row for row in gates if row.track_id == track_id)
        track_coverage = coverage(track_records, totals.get(track_id, 0))
        if available:
            status = "completed"
            failures = sum(row.status == "failed" for row in available)
            summary = f"Collected {len(available)} evidence records"
            if failures:
                summary += (
                    f"; {failures} executions failed and remain in the denominator."
                )
            else:
                summary += "."
        else:
            status = "unavailable"
            reasons = sorted({row.error for row in track_records if row.error})
            summary = reasons[0] if reasons else "No qualified evidence was produced."
        reports.append(
            EvaluationTrackReport(
                track_id=track_id,
                status=status,
                evidence_level=track_evidence_level(
                    manifest.mode, track_id, track_records
                ),
                summary=summary,
                coverage=track_coverage,
                metrics=track_metrics,
                gates=track_gates,
            )
        )
    return tuple(reports)


def build_report(
    *,
    manifest: RunManifest,
    run: EvaluationRun,
    records: list[ExecutionRecord],
    metrics: list[EvaluationMetric],
    gates: list[EvaluationGate],
    provenance: EvaluationProvenance,
    artifacts: tuple[EvaluationArtifact, ...],
    total_cases: int,
    multimodal_cases: int,
) -> EvaluationReport:
    metrics = select_report_metrics(manifest, metrics)
    selected_gates = list(gates)
    costs = build_costs(records)
    totals = dict.fromkeys(manifest.track_ids, total_cases)
    if "multimodal" in totals:
        totals["multimodal"] = multimodal_cases
    overall_coverage = aggregate_track_coverage(records, totals)
    quality = _value(metrics, "joint.realized_quality")
    if quality is None:
        quality = _value(metrics, "routing.accuracy")
    if quality is None:
        quality = _value(metrics, "model_pool.oracle_quality")
    latency = _value(metrics, "joint.latency_p95_ms")
    if latency is None:
        latency = _value(metrics, "capacity.latency_p95_ms")
    if latency is None:
        latency = _value(metrics, "routing.latency_p95_ms")
    promotion_summary_available = run.evidence_level != "E0"
    verdict = _verdict(selected_gates)
    unavailable = [gate for gate in selected_gates if gate.verdict == "unavailable"]
    failed = [gate for gate in selected_gates if gate.verdict == "fail"]
    gate_recommendations = [
        f"Resolve {gate.id} ({gate.name}): {gate.rationale or 'inspect evidence.'}"
        for gate in failed + unavailable
    ]
    architecture_findings = (
        []
        if run.evidence_level == "E0"
        else list(architecture_recommendations(metrics, selected_gates))
    )
    if run.evidence_level == "E0":
        gate_recommendations.insert(
            0,
            "E0 diagnostic only: validate the harness, then collect qualified evidence before inferring a recipe, pool, or runtime architecture change.",
        )
    recommendations = list(dict.fromkeys(gate_recommendations + architecture_findings))
    if not recommendations:
        recommendations = [
            "All applicable local gates passed; validate on the target runtime before promotion."
        ]
    return EvaluationReport(
        run=run,
        summary=EvaluationReportSummary(
            verdict=verdict,
            quality_score=quality if promotion_summary_available else None,
            latency_p95_ms=latency if promotion_summary_available else None,
            runtime_cost=(
                costs.runtime.amount if promotion_summary_available else None
            ),
            capacity_tco=(
                costs.capacity_tco.amount if promotion_summary_available else None
            ),
            coverage=overall_coverage,
            passed_gates=sum(gate.verdict == "pass" for gate in selected_gates),
            failed_gates=len(failed),
            unavailable_gates=len(unavailable),
        ),
        tracks=_track_reports(manifest, records, metrics, selected_gates, totals),
        metrics=tuple(metrics),
        gates=tuple(selected_gates),
        costs=costs,
        recommendations=tuple(recommendations),
        provenance=provenance,
        artifacts=artifacts,
    )


def _markdown_tracks(report: EvaluationReport) -> list[str]:
    lines = [
        "## Track coverage",
        "",
        "| Track | Status | Evidence | Coverage | Summary |",
        "|---|---|---|---:|---|",
    ]
    lines.extend(
        f"| {track.track_id} | {track.status} | {track.evidence_level} | "
        f"{track.coverage.evaluated}/{track.coverage.total} | {track.summary} |"
        for track in report.tracks
    )
    return lines


def _markdown_metrics(report: EvaluationReport) -> list[str]:
    lines = [
        "## Metrics",
        "",
        "| Metric | Track | Value | 95% CI | Baseline / delta | Unit | Samples |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for metric in report.metrics:
        value = "unavailable" if metric.value is None else f"{metric.value:.6g}"
        interval = (
            "-"
            if metric.confidence_interval is None
            else f"[{metric.confidence_interval[0]:.6g}, {metric.confidence_interval[1]:.6g}]"
        )
        comparison = "-"
        if metric.baseline_value is not None:
            delta = "-" if metric.delta is None else f"{metric.delta:+.6g}"
            comparison = f"{metric.baseline_value:.6g} / {delta}"
        lines.append(
            f"| {metric.name} | {metric.track_id or '-'} | {value} | {interval} | "
            f"{comparison} | {metric.unit} | {metric.sample_count or 0} |"
        )
    return lines


def _markdown_gates(report: EvaluationReport) -> list[str]:
    lines = [
        "## Gates",
        "",
        "| Gate | Disposition / verdict | Evidence | N / coverage | Owner | Rationale |",
        "|---|---|---|---:|---|---|",
    ]
    for gate in report.gates:
        gate_coverage = "-"
        if gate.coverage is not None:
            gate_coverage = (
                f"{gate.sample_count or 0} / "
                f"{gate.coverage.evaluated}/{gate.coverage.total}"
            )
        lines.append(
            f"| {gate.id} {gate.name} | {gate.disposition} / {gate.verdict} | "
            f"{gate.evidence_level or '-'}: {', '.join(gate.evidence_refs)} | "
            f"{gate_coverage} | {gate.owner or '-'} | {gate.rationale or ''} |"
        )
    return lines


def _markdown_costs(report: EvaluationReport) -> list[str]:
    lines = [
        "## Cost ledgers",
        "",
        "| Ledger | Amount (USD) | Tokens | GPU seconds | Energy (kWh) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, ledger in (
        ("runtime", report.costs.runtime),
        ("evaluation overhead", report.costs.evaluation_overhead),
        ("capacity TCO", report.costs.capacity_tco),
    ):
        amount = "unavailable" if ledger.amount is None else f"{ledger.amount:.6g}"
        lines.append(
            f"| {name} | {amount} | {(ledger.input_tokens or 0) + (ledger.output_tokens or 0)} | "
            f"{ledger.gpu_seconds or 0:.6g} | {ledger.energy_kwh or 0:.6g} |"
        )
    return lines


def _markdown_provenance(report: EvaluationReport) -> list[str]:
    lines = [
        "## Provenance",
        "",
        f"- Code revision: `{report.provenance.code_revision or 'unavailable'}`",
        f"- Workload: `{report.provenance.workload_snapshot_digest or 'unavailable'}`",
        f"- Policy: `{report.provenance.policy_snapshot_digest or 'unavailable'}`",
        f"- Binding: `{report.provenance.binding_snapshot_digest or 'unavailable'}`",
        f"- Pool: `{report.provenance.pool_snapshot_digest or 'unavailable'}`",
        f"- Environment: `{report.provenance.environment_snapshot_digest or 'unavailable'}`",
        f"- Benchmark revisions: `{report.provenance.benchmark_revisions or {}}`",
        "",
        "## Public artifacts",
        "",
    ]
    lines.extend(
        f"- `{artifact.name}` — `{artifact.digest or 'unavailable'}`"
        for artifact in report.artifacts
    )
    return lines


def render_markdown(report: EvaluationReport) -> str:
    gate_contract = report.gates[0].contract_version if report.gates else "unavailable"
    lines = [
        f"# Evaluation report: {report.run.name}",
        "",
        f"- Run: `{report.run.id}`",
        f"- Verdict: **{report.summary.verdict}**",
        f"- Mode / evidence: `{report.run.mode}` / `{report.run.evidence_level}`",
        f"- Change profile: `{report.run.change_profile}`",
        f"- Gate contract: `{gate_contract}`",
        f"- Coverage: {report.summary.coverage.evaluated}/{report.summary.coverage.total}",
        "",
    ]
    for section in (
        _markdown_tracks(report),
        _markdown_metrics(report),
        _markdown_gates(report),
        _markdown_costs(report),
        ["## Recommendations", "", *(f"- {row}" for row in report.recommendations)],
        _markdown_provenance(report),
    ):
        lines.extend(section)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_html(report: EvaluationReport) -> str:
    metric_rows = "".join(
        "<tr>"
        f"<td>{escape(metric.name)}</td><td>{escape(metric.track_id or '-')}</td>"
        f"<td>{'unavailable' if metric.value is None else f'{metric.value:.6g}'}</td>"
        f"<td>{escape(metric.unit)}</td>"
        "</tr>"
        for metric in report.metrics
    )
    gate_rows = "".join(
        "<tr>"
        f"<td>{escape(gate.id)} {escape(gate.name)}</td>"
        f"<td>{escape(gate.disposition)} / {escape(gate.verdict)}</td>"
        f"<td>{escape(gate.evidence_level or '-')}</td>"
        f"<td>{escape(', '.join(gate.evidence_refs))}</td>"
        f"<td>{escape(gate.rationale or '')}</td>"
        "</tr>"
        for gate in report.gates
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Evaluation {escape(report.run.id)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#182230}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #d0d5dd;padding:.5rem;text-align:left}</style></head><body>"
        f"<h1>Evaluation report: {escape(report.run.name)}</h1>"
        f"<p>Run <code>{escape(report.run.id)}</code> — verdict <strong>{escape(report.summary.verdict)}</strong></p>"
        f"<p>Mode <code>{escape(report.run.mode)}</code>; evidence <code>{escape(report.run.evidence_level)}</code>; "
        f"change profile <code>{escape(report.run.change_profile)}</code>.</p>"
        "<h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Track</th><th>Value</th><th>Unit</th>"
        f"</tr></thead><tbody>{metric_rows}</tbody></table>"
        "<h2>Gates</h2><table><thead><tr><th>Gate</th><th>Disposition / verdict</th>"
        "<th>Evidence level</th><th>Evidence refs</th><th>Rationale</th>"
        f"</tr></thead><tbody>{gate_rows}</tbody></table></body></html>\n"
    )
