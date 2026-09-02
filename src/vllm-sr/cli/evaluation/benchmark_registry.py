"""Pinned benchmark adapter registry derived from the semantic-routing audit.

The registry is metadata, not a claim that an external leaderboard has been
reproduced. Native source trees remain outside the repository and must pass the
pin verifier before an adapter may normalize them into a suite bundle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.reporting import EvidenceLevel, TrackID

ADAPTER_CONTRACT_VERSION = "benchmark-adapter.v1"

AdapterFamily = Literal[
    "prediction_file",
    "pairwise_preference",
    "dense_outcome_matrix",
    "scenario_session",
    "trajectory_prefix",
    "executable_agent",
    "fault_session",
    "fusion_graph",
    "model_budget_curve",
]


class BenchmarkAdapterDescriptor(StrictModel):
    schema_version: Literal[ADAPTER_CONTRACT_VERSION] = ADAPTER_CONTRACT_VERSION
    id: str
    name: str
    source_url: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_cache_key: str
    dataset_url: str | None = None
    dataset_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dataset_cache_key: str | None = None
    family: AdapterFamily
    decision_unit: str
    action_space: str
    outcome_source: str
    track_ids: tuple[TrackID, ...]
    native_metrics: tuple[str, ...]
    evidence_levels: tuple[EvidenceLevel, ...]
    required_capabilities: tuple[str, ...]
    limitations: tuple[str, ...]


class BenchmarkRegistry(StrictModel):
    schema_version: Literal[ADAPTER_CONTRACT_VERSION] = ADAPTER_CONTRACT_VERSION
    adapters: tuple[BenchmarkAdapterDescriptor, ...]


_ADAPTERS = (
    BenchmarkAdapterDescriptor(
        id="routerarena",
        name="RouterArena",
        source_url="https://github.com/RouteWorks/RouterArena.git",
        source_revision="fda4c53bcf9a979fd9c6f6bb6b713d6ab08ff43e",
        source_cache_key="routerarena",
        family="prediction_file",
        decision_unit="query",
        action_space="one model",
        outcome_source="cached or generated response plus task grader",
        track_ids=("routing", "model_pool", "joint"),
        native_metrics=(
            "accuracy",
            "cost_per_1000_queries",
            "arena_score",
            "optimality",
            "robustness",
            "router_latency",
        ),
        evidence_levels=("E3", "E4"),
        required_capabilities=(
            "blind prediction export",
            "task grader",
            "price snapshot",
            "paired perturbations",
        ),
        limitations=(
            "Does not establish that the evaluated model pool is optimal.",
            "Prediction stability alone is not semantic robustness.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="routejudge-orbit",
        name="RouteJudge / ORBIT",
        source_url="https://github.com/LAMDA-Model-Reuse/ORBIT.git",
        source_revision="494810de2605f69737e72b55baf6e60c95c6dec0",
        source_cache_key="routejudge-orbit",
        family="pairwise_preference",
        decision_unit="query plus budget and vote exposure",
        action_space="router recommendation followed by anonymous duel pair",
        outcome_source="offline qualification or anonymous user vote",
        track_ids=("routing", "joint", "preference"),
        native_metrics=(
            "elo",
            "win_rate",
            "participation",
            "head_to_head",
            "cost_preference_frontier",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "pairwise response bundle",
            "budget exposure",
            "assignment and participation ledger",
            "behavior propensity for causal claims",
        ),
        limitations=(
            "Win rate without exposure and propensity is not an online causal claim.",
            "Preference does not replace correctness, safety, or privacy grading.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="coderouterbench",
        name="Agent-as-a-Router / CodeRouterBench",
        source_url="https://github.com/LanceZPF/agent-as-a-router.git",
        source_revision="e43839edb0d5d0a9feec2f7078019406ab4d64bd",
        source_cache_key="coderouterbench",
        dataset_url="https://huggingface.co/datasets/Lance1573/CodeRouterBench",
        dataset_revision="e567d89bdd569c9c74ffc7c7118e50d15e46b886",
        dataset_cache_key="coderouterbench-dataset",
        family="dense_outcome_matrix",
        decision_unit="stream item with verified-history state",
        action_space="one coding backend",
        outcome_source="dense cached matrix plus sandboxed agentic OOD result",
        track_ids=("routing", "model_pool", "joint", "agentic"),
        native_metrics=(
            "average_performance",
            "cumulative_regret",
            "usd",
            "performance_per_usd",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "ordered stream snapshot",
            "dense arm outcomes",
            "verified feedback join",
            "sandbox result for OOD qualification",
        ),
        limitations=(
            "Sequence, warm-up, memory, artifact revision, and seed must remain frozen.",
            "Coding-only outcomes do not establish general conversational routing quality.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="llmrouterbench",
        name="LLMRouterBench",
        source_url="https://github.com/ynulihao/LLMRouterBench.git",
        source_revision="c77cb0506949d8f959e97967d2fefca0e8ff1b05",
        source_cache_key="llmrouterbench",
        family="dense_outcome_matrix",
        decision_unit="query",
        action_space="one model",
        outcome_source="large frozen dense response/outcome matrix",
        track_ids=("routing", "model_pool", "joint"),
        native_metrics=(
            "average_accuracy",
            "gain_at_budget",
            "gap_to_oracle",
            "performance_gain",
            "cost_save",
            "pareto_distance",
        ),
        evidence_levels=("E3", "E4"),
        required_capabilities=(
            "dense arm outcomes",
            "split and seed manifest",
            "price snapshot",
            "grader revisions",
        ),
        limitations=(
            "Frozen matrices age as providers and prices drift.",
            "Some judge and latency values are proxies rather than live measurements.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="routereval",
        name="RouterEval",
        source_url="https://github.com/MilkThink-Lab/RouterEval.git",
        source_revision="bf94b49cc9f8b37181715a7309e1b70ff5308942",
        source_cache_key="routereval",
        family="dense_outcome_matrix",
        decision_unit="query by sampled model pool",
        action_space="one of 3 to 1000 models",
        outcome_source="leaderboard-derived dense records",
        track_ids=("model_pool", "joint"),
        native_metrics=(
            "raw_score",
            "relative_reference",
            "relative_best",
            "predictive_entropy",
        ),
        evidence_levels=("E4",),
        required_capabilities=(
            "pool factorial manifest",
            "dense outcomes",
            "model metadata",
            "pool sampling seed",
        ),
        limitations=(
            "Research-scale pools are not necessarily deployable pools.",
            "Real USD, latency, health, and serving feasibility require separate evidence.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="routerbench",
        name="RouterBench",
        source_url="https://github.com/withmartian/routerbench.git",
        source_revision="cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c",
        source_cache_key="routerbench",
        family="dense_outcome_matrix",
        decision_unit="query",
        action_space="model, cascade, or over-generation policy",
        outcome_source="frozen dense outcomes",
        track_ids=("routing", "model_pool", "joint"),
        native_metrics=("quality", "cost", "aiq", "zero_router_convex_hull"),
        evidence_levels=("E4",),
        required_capabilities=(
            "dense outcomes",
            "price snapshot",
            "budget sweep",
            "no-information convex hull",
        ),
        limitations=(
            "Published model matrices and prices require a current snapshot before release use.",
            "It does not cover agentic, multimodal, or online preference behavior.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="xroutebench",
        name="LLMRouter / xRouteBench",
        source_url="https://github.com/ulab-uiuc/LLMRouter.git",
        source_revision="da3430baaea672743c3957457b0c76faba19876e",
        source_cache_key="xroutebench",
        dataset_url="https://huggingface.co/datasets/ulab-ai/xRouteBench",
        dataset_revision="ea4b6e1b29d9a734f55f0a637baf326bad6aa681",
        dataset_cache_key="xroutebench-dataset",
        family="scenario_session",
        decision_unit="single turn, session turn, or personalized state",
        action_space="one model",
        outcome_source="dense or cached task outcome plus offline preference/judge",
        track_ids=("routing", "model_pool", "joint", "multimodal", "preference"),
        native_metrics=(
            "task_metric",
            "tokens",
            "cost",
            "latency",
            "weighted_frontier",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "scenario state",
            "session grouping",
            "media manifest",
            "full hidden-call cost ledger",
        ),
        limitations=(
            "Caption, retrieval, and judge costs must be added to native cost accounting.",
            "Personalized simulation is not equivalent to real user preference evidence.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="twinrouterbench",
        name="TwinRouterBench",
        source_url="https://github.com/CommonstackAI/TwinRouterBench.git",
        source_revision="7cbb0deac8f697b5faa8489c309560e53d2ef088",
        source_cache_key="twinrouterbench",
        family="trajectory_prefix",
        decision_unit="agent trajectory prefix or step",
        action_space="model tier",
        outcome_source="static downgrade labels plus live SWE trajectory result",
        track_ids=("routing", "joint", "agentic"),
        native_metrics=(
            "row_pass",
            "row_exact",
            "trajectory_pass",
            "cost_save",
            "resolved",
            "bill",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "trajectory grouping",
            "prefix snapshot",
            "terminal sandbox result",
            "multi-seed live repeats",
        ),
        limitations=(
            "Static current-prefix labels are off-policy proxies for a full trajectory.",
            "Dynamic subsets must preserve selection criteria and cannot stand in for the full set silently.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="mmr-bench",
        name="MMR-Bench",
        source_url="https://github.com/Hunter-Wrynn/MMR-Bench.git",
        source_revision="83c8308427a3597213fdba298c098da887b8b01b",
        source_cache_key="mmr-bench",
        family="dense_outcome_matrix",
        decision_unit="multimodal query",
        action_space="one multimodal model",
        outcome_source="dense multimodal-model outcomes",
        track_ids=("model_pool", "joint", "multimodal"),
        native_metrics=("normalized_auc", "peak", "quality_normalized_cost"),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "media manifest",
            "typed model capability mask",
            "dense outcomes",
            "budget sweep",
        ),
        limitations=(
            "The audited implementation does not itself enforce a typed modality/capability mask.",
            "Normalized cost must not replace deployable USD and latency evidence.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="acebench",
        name="AceBench",
        source_url="https://github.com/OpenBMB/AceBench.git",
        source_revision="9a17bc2c7ee3fab9ca023036b82a81898512a001",
        source_cache_key="acebench",
        family="executable_agent",
        decision_unit="agent task or step with workspace and privacy state",
        action_space="edge/cloud assistance policy",
        outcome_source="isolated executable agent task",
        track_ids=("routing", "joint", "agentic", "safety"),
        native_metrics=(
            "task_utility",
            "pass_cubed",
            "cloud_usd",
            "tokens",
            "privacy_exposure",
        ),
        evidence_levels=("E5",),
        required_capabilities=(
            "isolated workspace",
            "tool sandbox",
            "privacy annotations",
            "egress ledger",
            "side-effect ledger",
        ),
        limitations=(
            "Privacy is a hard gate and must not be averaged into utility.",
            "The audited main path does not provide one uniform edge-FLOP calculation.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="continuity-bench",
        name="continuity-bench",
        source_url="https://github.com/Vishal-sys-code/continuity-bench.git",
        source_revision="5b7e7f82027c5b983435057ddc4d7115b7e9a97b",
        source_cache_key="continuitybench",
        family="fault_session",
        decision_unit="session failover event",
        action_space="stateless fallback or history forwarding",
        outcome_source="fixed-seed labeled-failover proxy and judge",
        track_ids=("joint", "agentic", "capacity"),
        native_metrics=(
            "context_preservation_rate",
            "continuity_loss",
            "wilson_interval",
            "latency_p50",
            "latency_p95",
        ),
        evidence_levels=("E4",),
        required_capabilities=(
            "session grouping",
            "exact-step fault manifest",
            "history state",
            "cluster-aware confidence interval",
        ),
        limitations=(
            "Labeled failure is not a real timeout, retry, streaming, or provider fault.",
            "Repeated turns within a conversation are not independent samples.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="fusionfactory",
        name="FusionFactory / LLMFusionBench",
        source_url="https://github.com/ulab-uiuc/FusionFactory.git",
        source_revision="ef62645a48b9e2167201047da047854415e2bc89",
        source_cache_key="llmfusionbench",
        family="fusion_graph",
        decision_unit="query or reasoning thought",
        action_space="model subset, topology, and synthesis policy",
        outcome_source="task metric plus judge over composite execution",
        track_ids=("model_pool", "joint", "agentic"),
        native_metrics=(
            "task_metric",
            "judge_score",
            "composite_quality",
            "full_call_cost",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "composite action graph",
            "all-call ledger",
            "topology snapshot",
            "judge calibration",
        ),
        limitations=(
            "Every hidden generation and judge call must be included in cost and latency.",
            "Grouping, normalization, and checkpoint leakage require an independent audit before causal claims.",
        ),
    ),
    BenchmarkAdapterDescriptor(
        id="r2-router",
        name="R2-Router / R2-Bench",
        source_url="https://github.com/UCF-ML-Research/R2-Router.git",
        source_revision="b0b2291aeee08feb4bedbd199ab014ec60d0004f",
        source_cache_key="r2-bench",
        family="model_budget_curve",
        decision_unit="query plus budget condition",
        action_space="model and output-token budget",
        outcome_source="dense model-by-budget quality curve",
        track_ids=("routing", "model_pool", "joint", "capacity"),
        native_metrics=(
            "area_under_deployment_curve",
            "peak_quality",
            "quality_normalized_cost",
            "scalarized_score",
        ),
        evidence_levels=("E4", "E5"),
        required_capabilities=(
            "model-budget outcome tensor",
            "budget enforcement",
            "common integration range",
            "price snapshot",
        ),
        limitations=(
            "Budget enforcement and length bias require calibration before comparing policies.",
            "Dataset/readme inventory drift must be resolved by the pinned source receipt.",
        ),
    ),
)


def get_benchmark_registry() -> BenchmarkRegistry:
    return BenchmarkRegistry(adapters=_ADAPTERS)


def get_benchmark_adapter(adapter_id: str) -> BenchmarkAdapterDescriptor:
    for adapter in _ADAPTERS:
        if adapter.id == adapter_id:
            return adapter
    raise ValueError(f"unknown benchmark adapter: {adapter_id}")
