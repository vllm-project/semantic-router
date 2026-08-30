"""Turn qualified metric/gate findings into scoped architecture actions."""

from __future__ import annotations

from dataclasses import dataclass

from cli.evaluation.reporting import EvaluationGate, EvaluationMetric

_MIN_ROUTING_COVERAGE = 0.95
_MAX_FALLBACK_RATE = 0.10
_POOL_ORACLE_GAIN_FLOOR = 0.02
_MIN_SELECTION_ARM_COVERAGE = 0.50
_MAX_NORMALIZED_REGRET = 0.20
_MIN_AGENT_SUCCESS = 0.90


@dataclass(frozen=True)
class ArchitectureFinding:
    id: str
    owner: str
    surface: str
    evidence: str
    action: str

    def render(self) -> str:
        return (
            f"[{self.id}] Owner={self.owner}; surface={self.surface}; "
            f"evidence={self.evidence}; action={self.action}"
        )


def _values(metrics: list[EvaluationMetric]) -> dict[str, float | None]:
    return {metric.id: metric.value for metric in metrics}


def _routing_findings(values: dict[str, float | None]) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    coverage = values.get("routing.coverage")
    if coverage is not None and coverage < _MIN_ROUTING_COVERAGE:
        findings.append(
            ArchitectureFinding(
                "AF-ROUTING-COVERAGE",
                "Router recipe owner",
                "signals / projections / decisions / fallback",
                f"routing.coverage={coverage:.3f}",
                "inspect unmatched decision traces and slice coverage before changing the model pool",
            )
        )
    fallback = values.get("routing.fallback_rate")
    if fallback is not None and fallback > _MAX_FALLBACK_RATE:
        findings.append(
            ArchitectureFinding(
                "AF-FALLBACK",
                "Router recipe owner",
                "decision eligibility and fallback boundary",
                f"routing.fallback_rate={fallback:.3f}",
                "separate intended abstention from missing capability and verify fallback does not cross policy or trust boundaries",
            )
        )
    return findings


def _pool_findings(values: dict[str, float | None]) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    oracle_gain = values.get("model_pool.oracle_gain")
    if oracle_gain is not None and oracle_gain <= _POOL_ORACLE_GAIN_FLOOR:
        findings.append(
            ArchitectureFinding(
                "AF-POOL-REDUNDANCY",
                "Model-pool owner",
                "PoolDefinition / ModelArm admission",
                f"model_pool.oracle_gain={oracle_gain:.3f}",
                "remove redundant arms or admit an arm that closes a measured capability, cost, or failure-domain gap",
            )
        )
    selection_coverage = values.get("model_pool.selection_arm_coverage")
    if (
        selection_coverage is not None
        and selection_coverage < _MIN_SELECTION_ARM_COVERAGE
    ):
        findings.append(
            ArchitectureFinding(
                "AF-POOL-COLLAPSE",
                "Selector and model-pool owners",
                "eligibility, calibration, and arm utilization",
                f"model_pool.selection_arm_coverage={selection_coverage:.3f}",
                "compare arm quality and marginal contribution before deciding whether low utilization is correct dominance or selector collapse",
            )
        )
    regret = values.get("joint.normalized_regret")
    if regret is not None and regret > _MAX_NORMALIZED_REGRET:
        findings.append(
            ArchitectureFinding(
                "AF-UNREALIZED-POOL-VALUE",
                "Router recipe and selector owners",
                "PolicyBinding / features / selector algorithm",
                f"joint.normalized_regret={regret:.3f}",
                "hold the pool fixed, inspect per-case oracle misses and decision traces, then improve feasibility recall or utility calibration",
            )
        )
    return findings


def _workload_findings(values: dict[str, float | None]) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    agent_success = values.get("agentic.success_rate")
    if agent_success is not None and agent_success < _MIN_AGENT_SUCCESS:
        findings.append(
            ArchitectureFinding(
                "AF-TRAJECTORY",
                "Agent and Router session owners",
                "session continuity / tool-loop protection / recovery",
                f"agentic.success_rate={agent_success:.3f}",
                "evaluate step and terminal failures separately, preserve tool ownership, and test state portability under exact-step faults",
            )
        )
    modality_support = values.get("multimodal.support_rate")
    if modality_support is not None and modality_support < 1.0:
        findings.append(
            ArchitectureFinding(
                "AF-MODALITY-CAPABILITY",
                "Router, model-pool, and serving owners",
                "typed modality admission and ModelArm capability mask",
                f"multimodal.support_rate={modality_support:.3f}",
                "separate admission, logical routing, payload transport, backend generation, and privacy failures by modality",
            )
        )
    violations = values.get("safety.violation_rate")
    if violations is not None and violations > 0:
        findings.append(
            ArchitectureFinding(
                "AF-HARD-POLICY",
                "Security and recipe owners",
                "static enforcement and fallback policy boundary",
                f"safety.violation_rate={violations:.3f}",
                "block promotion, identify the violating slice and enforcement path, and add a non-waivable regression case",
            )
        )
    saturation = values.get("capacity.saturation_concurrency")
    if saturation is not None:
        findings.append(
            ArchitectureFinding(
                "AF-CAPACITY-SATURATION",
                "Serving and placement owner",
                "queueing / batching / replica and GPU placement",
                f"capacity.saturation_concurrency={saturation:.0f}",
                "locate the SLO crossing, retry amplification, and per-arm bottleneck before changing logical routing policy",
            )
        )
    propensity = values.get("preference.propensity_coverage")
    if propensity is not None and propensity < 1.0:
        findings.append(
            ArchitectureFinding(
                "AF-ONLINE-ASSIGNMENT",
                "Online experimentation owner",
                "assignment / exposure / propensity ledger",
                f"preference.propensity_coverage={propensity:.3f}",
                "do not train or claim causal preference lift until every eligible exposure records its behavior propensity and executed action",
            )
        )
    return findings


def _metric_findings(values: dict[str, float | None]) -> list[ArchitectureFinding]:
    return (
        _routing_findings(values) + _pool_findings(values) + _workload_findings(values)
    )


def _gate_findings(gates: list[EvaluationGate]) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    by_id = {gate.id: gate for gate in gates}
    if by_id.get("G5") and by_id["G5"].verdict == "unavailable":
        findings.append(
            ArchitectureFinding(
                "AF-LIVE-FIDELITY-EVIDENCE",
                "Evaluation owner",
                "paired replay/live campaign",
                "G5=unavailable",
                "run the same frozen cases and grading contract in replay and live modes and retain failures in the paired denominator",
            )
        )
    if by_id.get("G7") and by_id["G7"].verdict == "unavailable":
        findings.append(
            ArchitectureFinding(
                "AF-CAPACITY-CONTRACT",
                "Serving and product SLO owners",
                "versioned load profile and SLO contract",
                "G7=unavailable",
                "declare traffic shape, cold/warm state, latency/error thresholds, saturation rule, and capacity headroom before promotion",
            )
        )
    if by_id.get("G8") and by_id["G8"].verdict == "unavailable":
        findings.append(
            ArchitectureFinding(
                "AF-SHADOW-CANARY-EVIDENCE",
                "Online experimentation owner",
                "shadow/canary control contract",
                "G8=unavailable",
                "add assignment and exposure counts, sample-ratio checks, hard guardrails, stop criteria, and a signed rollback recommendation",
            )
        )
    if by_id.get("G9") and by_id["G9"].verdict == "unavailable":
        findings.append(
            ArchitectureFinding(
                "AF-PREFERENCE-EVIDENCE",
                "Online preference owner",
                "consent / exposure / propensity / segment evidence",
                "G9=unavailable",
                "retain participation, propensity, effective sample size, confidence intervals, and key segments before enabling online adaptation",
            )
        )
    return findings


def architecture_recommendations(
    metrics: list[EvaluationMetric], gates: list[EvaluationGate]
) -> tuple[str, ...]:
    """Return deterministic, de-duplicated owner/action recommendations."""

    findings = _metric_findings(_values(metrics)) + _gate_findings(gates)
    unique: dict[str, ArchitectureFinding] = {}
    for finding in findings:
        unique.setdefault(finding.id, finding)
    return tuple(unique[key].render() for key in sorted(unique))
