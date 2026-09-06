"""Built-in normalization definitions for audited benchmark adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NativeArtifactRequirement,
    NativeMetricMapping,
    NormalizedAdapterPayload,
    artifact,
    metric,
)
from cli.evaluation.benchmark_normalizer_coderouter import normalize_coderouterbench
from cli.evaluation.benchmark_normalizer_mmr import normalize_mmr_bench
from cli.evaluation.benchmark_normalizer_routerarena import normalize_routerarena
from cli.evaluation.benchmark_normalizers_agentic import (
    normalize_acebench,
    normalize_continuitybench,
)
from cli.evaluation.benchmark_normalizers_dense import (
    normalize_llmrouterbench,
    normalize_routerbench,
)
from cli.evaluation.benchmark_normalizers_matrix import (
    normalize_fusionfactory,
    normalize_r2_router,
    normalize_xroutebench,
)
from cli.evaluation.benchmark_normalizers_trajectories import normalize_twinrouterbench
from cli.evaluation.benchmark_registry import get_benchmark_registry

NativeParser = Callable[[Path, BenchmarkNormalizerDescriptor], NormalizedAdapterPayload]


@dataclass(frozen=True, slots=True)
class BenchmarkNormalizerDefinition:
    """One immutable native-export contract and its trusted parser."""

    descriptor: BenchmarkNormalizerDescriptor
    parser: NativeParser | None

    def __post_init__(self) -> None:
        has_parser = self.parser is not None
        if self.descriptor.executable != has_parser:
            state = "executable" if self.descriptor.executable else "non-executable"
            requirement = "must have" if self.descriptor.executable else "must not have"
            raise ValueError(f"{state} normalizer definition {requirement} a parser")
        if self.parser is not None and not callable(self.parser):
            raise ValueError("normalizer definition parser must be callable")


class BenchmarkNormalizerCatalog:
    """Immutable built-in lookup with fail-closed adapter contract parity."""

    __slots__ = ("_by_id",)

    def __init__(self, definitions: Iterable[BenchmarkNormalizerDefinition]):
        by_id: dict[str, BenchmarkNormalizerDefinition] = {}
        for definition in definitions:
            adapter_id = definition.descriptor.adapter_id
            if adapter_id in by_id:
                raise ValueError(
                    f"duplicate benchmark normalizer definition: {adapter_id}"
                )
            by_id[adapter_id] = definition

        adapters = get_benchmark_registry().adapters
        adapters_by_id = {adapter.id: adapter for adapter in adapters}
        if len(adapters_by_id) != len(adapters):
            raise ValueError(
                "benchmark adapter registry contains duplicate descriptors"
            )
        registered_ids = set(by_id)
        expected_ids = set(adapters_by_id)
        if registered_ids != expected_ids:
            missing = sorted(expected_ids - registered_ids)
            unexpected = sorted(registered_ids - expected_ids)
            raise ValueError(
                "benchmark normalizer descriptor parity mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for adapter_id, definition in by_id.items():
            adapter = adapters_by_id[adapter_id]
            if not set(definition.descriptor.track_ids).issubset(adapter.track_ids):
                raise ValueError(
                    "benchmark normalizer descriptor parity mismatch: "
                    f"{adapter_id} declares tracks outside its adapter contract"
                )
        self._by_id: Mapping[str, BenchmarkNormalizerDefinition] = MappingProxyType(
            by_id
        )

    @property
    def descriptors(self) -> tuple[BenchmarkNormalizerDescriptor, ...]:
        return tuple(definition.descriptor for definition in self._by_id.values())

    def require(self, adapter_id: str) -> BenchmarkNormalizerDefinition:
        try:
            return self._by_id[adapter_id]
        except KeyError as exc:
            raise ValueError(f"unknown benchmark normalizer: {adapter_id}") from exc


def _definition(
    adapter_id: str,
    parser: NativeParser,
    schema: str,
    tracks: tuple[str, ...],
    artifacts: tuple[NativeArtifactRequirement, ...],
    metrics: tuple[NativeMetricMapping, ...],
    limitations: tuple[str, ...],
) -> BenchmarkNormalizerDefinition:
    return BenchmarkNormalizerDefinition(
        descriptor=BenchmarkNormalizerDescriptor(
            adapter_id=adapter_id,
            export_schema_id=schema,
            executable=True,
            track_ids=tracks,
            required_artifacts=artifacts,
            native_metric_mappings=metrics,
            limitations=limitations,
        ),
        parser=parser,
    )


_BUILTIN_NORMALIZERS: tuple[BenchmarkNormalizerDefinition, ...] = (
    _definition(
        "routerarena",
        normalize_routerarena,
        "routerarena.predictions-and-robustness.v2",
        ("routing", "model_pool", "joint"),
        (
            artifact("predictions", "predictions.json", "application/json"),
            artifact(
                "robustness-predictions",
                "robustness_predictions.json",
                "application/json",
            ),
        ),
        (
            metric("accuracy", "outcomes.quality", "Native task-grade accuracy."),
            metric("cost", "outcomes.runtime_cost_usd", "Per-response USD."),
            metric("prediction", "decisions.selected_arm_id", "Selected model."),
            metric(
                "global index",
                "perturbations.source_case_id+perturbed_case_id",
                "Native full/robustness pairing key used by RouterArena's flip reducer.",
            ),
        ),
        (
            "Dense model-pool evidence comes only from the optimality-complete full prediction export; robustness rows contribute routing decisions and invariant pairs only.",
            "The robustness method attests RouterArena action stability under its pinned paraphrase split; it does not infer contamination or arbitrary OOD coverage.",
        ),
    ),
    BenchmarkNormalizerDefinition(
        descriptor=BenchmarkNormalizerDescriptor(
            adapter_id="routejudge-orbit",
            export_schema_id="routejudge-orbit.unavailable.v1",
            executable=False,
            track_ids=(),
            required_artifacts=(),
            native_metric_mappings=(),
            blocker=(
                "The pinned ORBIT repository does not emit RouteJudge vote/exposure "
                "records or a safe per-case export; its available serialized datasets "
                "use pickle. Offline conversion would require executing or deserializing "
                "untrusted upstream artifacts."
            ),
        ),
        parser=None,
    ),
    _definition(
        "coderouterbench",
        normalize_coderouterbench,
        "coderouterbench.id-results.v1",
        ("routing", "model_pool", "joint"),
        (
            artifact("tasks", "id_test_tasks.jsonl", "application/x-ndjson"),
            artifact("results", "id_test_results_long.csv", "text/csv", max_mib=2048),
            artifact("decisions", "id_decisions.jsonl", "application/x-ndjson"),
            artifact("models", "models.json", "application/json"),
        ),
        (
            metric("score", "outcomes.quality", "Native coding task score."),
            metric("cost_usd", "outcomes.runtime_cost_usd", "Recorded USD."),
            metric("latency_ms", "outcomes.latency_ms", "Recorded latency."),
            metric("chosen_model", "decisions.selected_arm_id", "Router choice."),
        ),
        (
            "The pinned ID task index has no prompt text, so visible cases carry task identity and dimension rather than executable coding prompts.",
            "This path excludes the sandboxed OOD agent stream and does not claim agentic evidence.",
        ),
    ),
    _definition(
        "llmrouterbench",
        normalize_llmrouterbench,
        "llmrouterbench.result-documents.v1",
        ("model_pool",),
        (artifact("results", "results.jsonl", "application/x-ndjson", max_mib=2048),),
        (
            metric("records[].score", "outcomes.quality", "Native task score."),
            metric("records[].cost", "outcomes.runtime_cost_usd", "Recorded USD."),
            metric("records[].completion_tokens", "outcomes.output_tokens", "Tokens."),
        ),
        (
            "Normalizes aligned collector result documents only; it does not replay a learned router decision.",
            "Gain-at-budget, Pareto-distance, and pricing-drift reducers are not attested.",
        ),
    ),
    BenchmarkNormalizerDefinition(
        descriptor=BenchmarkNormalizerDescriptor(
            adapter_id="routereval",
            export_schema_id="routereval.unavailable.v1",
            executable=False,
            track_ids=(),
            required_artifacts=(),
            native_metric_mappings=(),
            blocker=(
                "The pinned repository's per-case leaderboard/model-pool data is only "
                "loaded through pickle and it emits no safe JSON/CSV equivalent. The "
                "normalizer refuses code execution and unsafe deserialization."
            ),
        ),
        parser=None,
    ),
    _definition(
        "routerbench",
        normalize_routerbench,
        "routerbench.wide-csv.v1",
        ("model_pool",),
        (
            artifact("models", "models.json", "application/json"),
            artifact("wide-data", "data.csv", "text/csv", max_mib=2048),
        ),
        (
            metric("<model>", "outcomes.quality", "Native wide-table score."),
            metric("<model>|total_cost", "outcomes.runtime_cost_usd", "Recorded USD."),
        ),
        (
            "Covers the converted dense wide table only; cascade and over-generation policies require separate decision exports.",
            "AIQ and zero-router convex-hull reducers are not attested.",
        ),
    ),
    _definition(
        "xroutebench",
        normalize_xroutebench,
        "xroutebench.standardized-csv.v1",
        ("model_pool",),
        (artifact("routing-data", "routing-data.csv", "text/csv", max_mib=2048),),
        (
            metric("performance", "outcomes.quality", "Scenario task metric."),
            metric("response_time", "outcomes.latency_ms", "Response latency."),
            metric("output_tokens", "outcomes.output_tokens", "Generated tokens."),
        ),
        (
            "One standardized scenario CSV is normalized at a time; session state and personalization are not inferred.",
            "Preference, multimodal media, hidden-call costs, and weighted-frontier reducers require separate native artifacts.",
        ),
    ),
    _definition(
        "twinrouterbench",
        normalize_twinrouterbench,
        "twinrouterbench.static-summary.v1",
        ("agentic",),
        (
            artifact("question-bank", "question_bank.jsonl", "application/x-ndjson"),
            artifact("summary", "eval_summary.json", "application/json"),
        ),
        (
            metric("rows[].match", "trajectories.task_score", "Tier exact match."),
            metric("rows[].passed", "trajectories.terminal_success", "Step pass."),
            metric("rows[].pred_tier_id", "trajectories.selected_action_id", "Tier."),
        ),
        (
            "Static prefix rows remain weak-label off-policy evidence, not a dynamic SWE sandbox result.",
            "Cost savings and full-trajectory resolved/bill metrics are not attested.",
        ),
    ),
    _definition(
        "mmr-bench",
        normalize_mmr_bench,
        "mmrbench.merged-csv.v1",
        ("model_pool", "multimodal"),
        (
            artifact("models", "models.json", "application/json"),
            artifact("merged-data", "MMR-Bench.csv", "text/csv", max_mib=2048),
        ),
        (
            metric("<model>_correct", "outcomes.quality", "Native correctness."),
            metric(
                "<model>_cost",
                "outcomes.source_record_digest",
                "Native normalized cost is bound but is not attested as USD.",
            ),
            metric("img_path", "media_manifest.digest", "Hashed local media bytes."),
        ),
        (
            "Native cost is a benchmark-normalized proxy and is deliberately not converted to runtime USD.",
            "Normalized AUC, peak, and deployability/capability-mask reducers are not attested.",
        ),
    ),
    _definition(
        "acebench",
        normalize_acebench,
        "acebench.run-summary.v1",
        ("agentic",),
        (artifact("summary", "summary.json", "application/json"),),
        (
            metric("scores.overall_score", "trajectories.task_score", "Task score."),
            metric("error", "trajectories.terminal_success", "Execution status."),
            metric(
                "scores.privacy_score",
                "trajectories.source_record_digest",
                "Privacy score is bound; it is not converted into an exposure count.",
            ),
        ),
        (
            "Run summaries expose terminal task scores but not a complete tool, side-effect, or egress trajectory.",
            "Privacy score is not converted into a privacy-exposure count or safety gate.",
        ),
    ),
    _definition(
        "continuity-bench",
        normalize_continuitybench,
        "continuitybench.labeled-failover.v3",
        ("agentic",),
        (
            artifact("conversations", "conversations.json", "application/json"),
            artifact(
                "experiment-manifest",
                "experiment_manifest.json",
                "application/json",
            ),
            artifact("raw-metrics", "raw_metrics.csv", "text/csv"),
            artifact("baseline-log", "baseline_log.jsonl", "application/x-ndjson"),
            artifact("treatment-log", "treatment_log.jsonl", "application/x-ndjson"),
        ),
        (
            metric("preserved", "trajectories.terminal_success", "Context preserved."),
            metric(
                "failure_turn+mode",
                "faults.sequence+kind",
                "Pinned source fallback-selection manifest.",
            ),
            metric(
                "failed_over+failover_from+failure_mode",
                "faults.failover_labeled",
                "Source-labeled provider fallback bound to the manifest; not a real fault receipt.",
            ),
            metric(
                "latency_ms",
                "trajectories.source_record_digest",
                "Latency remains source-bound and is not misreported as task score.",
            ),
        ),
        (
            "The native proxy selects a fallback and labels timeout, rate-limit, or API-error modes; it does not execute real timeout, HTTP error, retry-after, network, or partial-stream faults.",
            "Latency and concurrency columns remain source-bound; this path does not claim capacity qualification.",
        ),
    ),
    _definition(
        "fusionfactory",
        normalize_fusionfactory,
        "fusionfactory.aligned-csv.v1",
        ("model_pool",),
        (artifact("aligned-results", "aligned.csv", "text/csv", max_mib=2048),),
        (
            metric("performance", "outcomes.quality", "Native task metric."),
            metric(
                "input_price*input_tokens_num+output_price*output_tokens_num",
                "outcomes.runtime_cost_usd",
                "USD reconstructed from recorded token counts and per-million prices.",
            ),
            metric("llm", "outcomes.arm_id", "Base or reasoning action."),
        ),
        (
            "Aligned rows do not encode fusion graph topology, synthesis policy, or hidden judge calls.",
            "Composite-quality and full-call-ledger reducers are not attested.",
        ),
    ),
    _definition(
        "r2-router",
        normalize_r2_router,
        "r2bench.model-budget-csv.v1",
        ("model_pool",),
        (artifact("curves", "curves.csv", "text/csv", max_mib=2048),),
        (
            metric("score", "outcomes.quality", "Quality at a token budget."),
            metric("token_count", "outcomes.output_tokens", "Observed output length."),
            metric("budget_tokens", "outcomes.budget_tokens", "Enforced budget."),
        ),
        (
            "Requires the fixed 15-budget long-form safe export; pickle/joblib predictors are never loaded.",
            "Deployment-curve AUC, scalarization, length-bias calibration, and concurrency capacity are not attested.",
        ),
    ),
)

_REGISTRY = BenchmarkNormalizerCatalog(_BUILTIN_NORMALIZERS)


def get_benchmark_normalizers() -> tuple[BenchmarkNormalizerDescriptor, ...]:
    return _REGISTRY.descriptors


def get_benchmark_normalizer_definition(
    adapter_id: str,
) -> BenchmarkNormalizerDefinition:
    return _REGISTRY.require(adapter_id)


def get_benchmark_normalizer(adapter_id: str) -> BenchmarkNormalizerDescriptor:
    return get_benchmark_normalizer_definition(adapter_id).descriptor
