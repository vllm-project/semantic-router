"""Shared builders for normalized-suite executor contract tests."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.canonical import canonical_json_bytes, digest_value, sha256_digest
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import ImagePart, ImageURL, Message
from cli.evaluation.contracts import (
    CapacitySLO,
    CaseGrading,
    CaseVisible,
    RunManifest,
)
from cli.evaluation.execution_contract import (
    NORMALIZED_LIVE_EXECUTOR_ID,
    NORMALIZED_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.executor_contracts import BUILTIN_NORMALIZED_SUITE_EXECUTORS
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.suite_catalog import NormalizedSuiteCatalog
from cli.evaluation.suite_contract import (
    BenchmarkSourceReceipt,
    NormalizedCapacityObservation,
    NormalizedDecision,
    NormalizedMultimodalObservation,
    NormalizedOutcome,
    NormalizedPreference,
    NormalizedSafetyObservation,
    NormalizedTrajectoryStep,
)
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    LICENSE_CONTRACT_VERSION,
    BenchmarkSuiteInstallRequest,
    NormalizedMediaEntry,
    SuiteArtifactInstall,
    SuiteArtifactRole,
)
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.target_contracts import (
    EvaluationTarget,
    EvaluationTargetArm,
    ManifestMixture,
    MixtureDecisionBinding,
)
from evaluation_contract_test_support import (
    build_routing_recipe_plan,
    default_capacity_load_protocol,
)


def _catalog(store: NormalizedSuiteStore) -> NormalizedSuiteCatalog:
    return NormalizedSuiteCatalog(
        store,
        DEFAULT_EXECUTOR_REGISTRY,
        BUILTIN_NORMALIZED_SUITE_EXECUTORS,
    )


@pytest.fixture(autouse=True)
def _trusted_source_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(descriptor: Any, _source_root: Path) -> BenchmarkSourceReceipt:
        return BenchmarkSourceReceipt(
            adapter_id=descriptor.id,
            expected_source_revision=descriptor.source_revision,
            observed_source_revision=descriptor.source_revision,
            expected_dataset_revision=descriptor.dataset_revision,
            observed_dataset_revision=descriptor.dataset_revision,
            source_clean=True,
            dataset_clean=(True if descriptor.dataset_revision else None),
            verified=True,
        )

    monkeypatch.setattr(
        "cli.evaluation.suite_store_install.require_verified_benchmark_source",
        verified,
    )
    # The executor fixtures use deliberately synthetic normalized rows; the
    # native-export derivation trust boundary has dedicated integration tests.
    monkeypatch.setattr(
        "cli.evaluation.suite_store_install.verify_registered_normalization",
        lambda *_args, **_kwargs: None,
    )


_PIXEL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)
_PRIVATE_MARKERS = (
    "PRIVATE NORMALIZED PROMPT",
    "HIDDEN EXPECTED ANSWER",
    "secret-arm-a",
    "secret-arm-b",
    "private-grader",
)


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _write_license(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": LICENSE_CONTRACT_VERSION,
                "licenses": [
                    {
                        "id": "upstream",
                        "name": "Pinned upstream fixture license",
                        "redistribution": "metadata_only",
                    }
                ],
            }
        )
    )


def _artifact(root: Path, role: SuiteArtifactRole) -> SuiteArtifactInstall:
    relative_path, media_type, _ = ARTIFACT_ROLE_LAYOUT[role]
    content = (root / relative_path).read_bytes()
    return SuiteArtifactInstall(
        role=role,
        relative_path=relative_path,
        digest="sha256:" + hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type=media_type,
    )


def _receipt(adapter_id: str) -> BenchmarkSourceReceipt:
    descriptor = get_benchmark_adapter(adapter_id)
    return BenchmarkSourceReceipt(
        adapter_id=adapter_id,
        expected_source_revision=descriptor.source_revision,
        observed_source_revision=descriptor.source_revision,
        expected_dataset_revision=descriptor.dataset_revision,
        observed_dataset_revision=descriptor.dataset_revision,
        source_clean=True,
        dataset_clean=True if descriptor.dataset_revision else None,
        verified=True,
    )


def _digest(label: str) -> str:
    return digest_value({"private_source_record": label})


def _base_bundle(
    root: Path,
    case_id: str,
    *,
    track_ids: tuple[str, ...],
    image: bool = False,
    expected_route: str = "secret-arm-a",
    expected_answer: str = "HIDDEN EXPECTED ANSWER",
) -> None:
    message = Message(
        role="user",
        content=(
            (ImagePart(image_url=ImageURL(url=_PIXEL, detail="low")),)
            if image
            else f"PRIVATE NORMALIZED PROMPT {case_id}"
        ),
    )
    _write_jsonl(
        root / "visible/cases.jsonl",
        (
            CaseVisible(
                id=case_id,
                track_ids=track_ids,
                messages=(message,),
                modality="image" if image else "text",
                trajectory_id=f"private-trajectory-{case_id}",
            ),
        ),
    )
    _write_jsonl(
        root / "grading/cases.jsonl",
        (
            CaseGrading(
                case_id=case_id,
                expected_route=expected_route,
                expected_answer=expected_answer,
                preferred_arm_id=expected_route,
                should_block=False,
            ),
        ),
    )
    _write_license(root / "metadata/licenses.json")


def _qualification_cases(
    root: Path,
    case_ids: tuple[str, ...],
    *,
    track_ids: tuple[str, ...],
) -> None:
    _write_jsonl(
        root / "visible/cases.jsonl",
        (
            CaseVisible(
                id=case_id,
                track_ids=track_ids,
                messages=(Message(role="user", content=f"PRIVATE {case_id}"),),
                trajectory_id=f"trajectory-{case_id}",
            )
            for case_id in case_ids
        ),
    )
    _write_jsonl(
        root / "grading/cases.jsonl",
        (
            CaseGrading(
                case_id=case_id,
                expected_route="secret-arm-a",
                preferred_arm_id="secret-arm-a",
                should_block=False,
            )
            for case_id in case_ids
        ),
    )
    _write_license(root / "metadata/licenses.json")


def _decision(case_id: str) -> NormalizedDecision:
    return NormalizedDecision(
        case_id=case_id,
        selected_arm_id="secret-arm-a",
        selection_status="selected",
        success=True,
        latency_ms=2.5,
        source_record_digest=_digest(f"{case_id}-decision"),
    )


def _outcomes(case_id: str) -> tuple[NormalizedOutcome, ...]:
    return (
        NormalizedOutcome(
            case_id=case_id,
            arm_id="secret-arm-a",
            success=True,
            quality=0.9,
            latency_ms=18,
            input_tokens=11,
            output_tokens=7,
            runtime_cost_usd=0.003,
            grader_id="private-grader",
            grader_revision="private-grader-v1",
            split="frozen-test",
            source_record_digest=_digest(f"{case_id}-outcome-a"),
        ),
        NormalizedOutcome(
            case_id=case_id,
            arm_id="secret-arm-b",
            success=True,
            quality=0.6,
            latency_ms=9,
            input_tokens=11,
            output_tokens=5,
            runtime_cost_usd=0.001,
            grader_id="private-grader",
            grader_revision="private-grader-v1",
            split="frozen-test",
            source_record_digest=_digest(f"{case_id}-outcome-b"),
        ),
    )


def _write_common_observations(root: Path, case_id: str) -> list[SuiteArtifactRole]:
    _write_jsonl(root / "grading/decisions.jsonl", (_decision(case_id),))
    _write_jsonl(root / "grading/outcomes.jsonl", _outcomes(case_id))
    return ["decisions", "outcomes"]


def _suite_request(
    root: Path,
    *,
    adapter_id: str,
    suite_id: str,
    case_id: str,
    tracks: tuple[str, ...],
    optional_roles: Iterable[SuiteArtifactRole],
    case_count: int = 1,
) -> BenchmarkSuiteInstallRequest:
    descriptor = get_benchmark_adapter(adapter_id)
    roles: tuple[SuiteArtifactRole, ...] = (
        "visible_cases",
        "grading_cases",
        *tuple(optional_roles),
        "license_manifest",
    )
    return BenchmarkSuiteInstallRequest(
        id=suite_id,
        name=f"Normalized {descriptor.name} integration suite",
        adapter_id=adapter_id,
        source_receipt=_receipt(adapter_id),
        decision_unit=descriptor.decision_unit,
        action_space=descriptor.action_space,
        track_ids=tracks,  # type: ignore[arg-type]
        normalization_origin="user_provided_import",
        split_protocol="fixed composite integration split",
        case_count=case_count,
        arm_ids=("secret-arm-a", "secret-arm-b"),
        data_classification="restricted",
        redistribution="metadata_only",
        artifacts=tuple(_artifact(root, role) for role in roles),
        limitations=("integration-normalized evidence only",),
    )


def _install_xroute_suite(root: Path, store: NormalizedSuiteStore) -> str:
    xroute = root / "xroute"
    _base_bundle(
        xroute,
        "xroute-private-case",
        image=True,
        track_ids=("routing", "model_pool", "joint", "multimodal", "preference"),
    )
    xroute_roles = _write_common_observations(xroute, "xroute-private-case")
    _write_jsonl(
        xroute / "grading/multimodal-observations.jsonl",
        (
            NormalizedMultimodalObservation(
                case_id="xroute-private-case",
                modality="image",
                supported=True,
                quality=0.88,
                privacy_violations=0,
                source_record_digest=_digest("xroute-multimodal"),
            ),
        ),
    )
    _write_jsonl(
        xroute / "metadata/media.jsonl",
        (
            NormalizedMediaEntry(
                id="pixel",
                digest=_digest("xroute-pixel"),
                media_type="image/png",
                size_bytes=1,
                modality="image",
                license_id="upstream",
            ),
        ),
    )
    _write_jsonl(
        xroute / "grading/preferences.jsonl",
        (
            NormalizedPreference(
                case_id="xroute-private-case",
                left_action_id="secret-arm-a",
                right_action_id="secret-arm-b",
                preference="left",
                chosen_action_id="secret-arm-a",
                reward=1.0,
                exposure_probability=0.5,
                behavior_propensity=0.5,
                participant_digest=_digest("private-participant"),
                source_record_digest=_digest("xroute-preference"),
            ),
        ),
    )
    xroute_roles.extend(("multimodal_observations", "preferences", "media_manifest"))
    return store.install(
        _suite_request(
            xroute,
            adapter_id="xroutebench",
            suite_id="composite-xroute",
            case_id="xroute-private-case",
            tracks=("routing", "model_pool", "joint", "multimodal", "preference"),
            optional_roles=xroute_roles,
        ),
        xroute,
        source_root=xroute.parent,
    ).id


def _install_live_target_suite(root: Path, store: NormalizedSuiteStore) -> str:
    bundle = root / "target-live"
    case_id = "target-live-private-case"
    _base_bundle(
        bundle,
        case_id,
        track_ids=("routing", "multimodal"),
        image=True,
        expected_route="provider-strong",
        expected_answer="TARGET HIDDEN ANSWER",
    )
    _write_jsonl(bundle / "grading/decisions.jsonl", (_decision(case_id),))
    _write_jsonl(
        bundle / "grading/multimodal-observations.jsonl",
        (
            NormalizedMultimodalObservation(
                case_id=case_id,
                modality="image",
                supported=True,
                quality=0.88,
                privacy_violations=0,
                source_record_digest=_digest("target-live-multimodal"),
            ),
        ),
    )
    _write_jsonl(
        bundle / "metadata/media.jsonl",
        (
            NormalizedMediaEntry(
                id="target-pixel",
                digest=_digest("target-live-pixel"),
                media_type="image/png",
                size_bytes=1,
                modality="image",
                license_id="upstream",
            ),
        ),
    )
    return store.install(
        _suite_request(
            bundle,
            adapter_id="xroutebench",
            suite_id="target-live-xroute",
            case_id=case_id,
            tracks=("routing", "multimodal"),
            optional_roles=(
                "decisions",
                "multimodal_observations",
                "media_manifest",
            ),
        ),
        bundle,
        source_root=bundle.parent,
    ).id


def _install_ace_suite(root: Path, store: NormalizedSuiteStore) -> str:
    ace = root / "ace"
    _base_bundle(
        ace,
        "ace-private-case",
        track_ids=("routing", "joint", "agentic", "safety"),
    )
    ace_roles = _write_common_observations(ace, "ace-private-case")
    _write_jsonl(
        ace / "grading/trajectories.jsonl",
        (
            NormalizedTrajectoryStep(
                trajectory_id="private-trajectory-ace-private-case",
                step_id="private-step-0",
                sequence=0,
                case_id="ace-private-case",
                selected_action_id="secret-arm-a",
                tool_name="private-tool",
                tool_call_valid=True,
                terminal=False,
                privacy_exposures=0,
                source_record_digest=_digest("ace-step-0"),
            ),
            NormalizedTrajectoryStep(
                trajectory_id="private-trajectory-ace-private-case",
                step_id="private-step-1",
                sequence=1,
                case_id="ace-private-case",
                selected_action_id="secret-arm-a",
                terminal=True,
                terminal_success=True,
                task_score=0.95,
                privacy_exposures=0,
                source_record_digest=_digest("ace-step-1"),
            ),
        ),
    )
    _write_jsonl(
        ace / "grading/safety-observations.jsonl",
        (
            NormalizedSafetyObservation(
                case_id="ace-private-case",
                violations=0,
                blocked=False,
                source_record_digest=_digest("ace-safety"),
            ),
        ),
    )
    ace_roles.extend(("trajectories", "safety_observations"))
    return store.install(
        _suite_request(
            ace,
            adapter_id="acebench",
            suite_id="composite-ace",
            case_id="ace-private-case",
            tracks=("routing", "joint", "agentic", "safety"),
            optional_roles=ace_roles,
        ),
        ace,
        source_root=ace.parent,
    ).id


def _install_r2_suite(root: Path, store: NormalizedSuiteStore) -> str:
    r2 = root / "r2"
    _base_bundle(
        r2,
        "r2-private-case",
        track_ids=("routing", "model_pool", "joint", "capacity"),
    )
    r2_roles = _write_common_observations(r2, "r2-private-case")
    _write_jsonl(
        r2 / "grading/capacity-observations.jsonl",
        (
            NormalizedCapacityObservation(
                case_id="r2-private-case",
                concurrency=1,
                success=True,
                latency_ms=12,
                throughput_rps=8,
                runtime_cost_usd=0.002,
                capacity_tco_usd=0.003,
                gpu_seconds=0.05,
                energy_kwh=0.0002,
                elapsed_seconds=1,
                source_record_digest=_digest("r2-capacity-1"),
            ),
            NormalizedCapacityObservation(
                case_id="r2-private-case",
                concurrency=2,
                success=True,
                latency_ms=16,
                throughput_rps=14,
                runtime_cost_usd=0.003,
                capacity_tco_usd=0.004,
                gpu_seconds=0.08,
                energy_kwh=0.0003,
                elapsed_seconds=1,
                source_record_digest=_digest("r2-capacity-2"),
            ),
        ),
    )
    r2_roles.append("capacity_observations")
    return store.install(
        _suite_request(
            r2,
            adapter_id="r2-router",
            suite_id="composite-r2",
            case_id="r2-private-case",
            tracks=("routing", "model_pool", "joint", "capacity"),
            optional_roles=r2_roles,
        ),
        r2,
        source_root=r2.parent,
    ).id


def _install_composite(root: Path, store: NormalizedSuiteStore) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                _install_xroute_suite(root, store),
                _install_ace_suite(root, store),
                _install_r2_suite(root, store),
            )
        )
    )


def _manifest(
    run_id: str,
    suite_ids: tuple[str, ...],
    suite_store: NormalizedSuiteStore,
) -> RunManifest:
    revisions = {
        suite_id: suite_store.get_suite_manifest(suite_id).revision
        for suite_id in suite_ids
    }
    return RunManifest.from_semantic_fields(
        run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vllm-sr-evaluation:{run_id}")),
        name=f"Normalized evaluation {run_id}",
        description="Normalized suite contract fixture",
        mode="replay",
        target=EvaluationTarget(
            id="benchmark-source", kind="normalized-benchmark-source"
        ),
        change_profile="schema_adapter",
        gate_contract_version="evaluation-release-gates.v2",
        suite_ids=suite_ids,
        suite_revisions=revisions,
        suite_executors=dict.fromkeys(suite_ids, NORMALIZED_REPLAY_EXECUTOR_ID),
        track_ids=TRACK_IDS,
        sample_limit=100,
        concurrency=1,
        seed=19,
        created_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value(
            {"kind": "normalized-replay-policy", "suite_revisions": revisions}
        ),
        config_digest=digest_value({"normalized_suite_test": True}),
        redaction_policy="strict-no-prompts",
    )


def _target_arms() -> tuple[EvaluationTargetArm, ...]:
    return (
        EvaluationTargetArm(
            id="arm-fast",
            model="provider-fast",
            provider_model_id_digest=sha256_digest(b"provider-fast"),
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=2.0,
            capabilities=("chat",),
            modalities=("text",),
            runtime_revision="runtime-v1",
            config_digest=digest_value("fast-config"),
        ),
        EvaluationTargetArm(
            id="arm-strong",
            model="provider-strong",
            provider_model_id_digest=sha256_digest(b"provider-strong"),
            input_cost_per_million_tokens_usd=3.0,
            output_cost_per_million_tokens_usd=4.0,
            capabilities=("chat", "vision"),
            modalities=("text", "image"),
            runtime_revision="runtime-v1",
            config_digest=digest_value("strong-config"),
        ),
    )


def _target_mixture() -> ManifestMixture:
    arms = _target_arms()
    recipe_name = "target-recipe"
    recipe_digest = digest_value("target-policy-v1")
    pool_digest = model_pool_snapshot_digest(arms)
    mixture_id = mixture_target_id(recipe_name)
    aliases = ("entrypoint-a",)
    selector_policy_digest = digest_value("target-selector-policy")
    selector_digest = selector_snapshot_digest(selector_policy_digest, ())
    adaptation_digest = digest_value("target-adaptation")
    binding_digest = digest_value("target-binding-v1")
    return ManifestMixture(
        id=mixture_id,
        entrypoint_model="entrypoint-a",
        aliases=aliases,
        recipe_name=recipe_name,
        recipe_description="Normalized target recipe",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy_digest,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=(),
        fallback_arm_id="arm-fast",
        decisions=(
            MixtureDecisionBinding(
                name="default",
                algorithm="static",
                arm_ids=("arm-fast", "arm-strong"),
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy_digest,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=tuple(arm.id for arm in arms),
            fallback_arm_id="arm-fast",
            signals=(),
            projections=(),
        ),
    )


def _live_manifest(
    run_id: str,
    suite_id: str,
    suite_store: NormalizedSuiteStore,
    *,
    track_ids: tuple[str, ...] = ("routing", "multimodal"),
) -> RunManifest:
    revision = suite_store.get_suite_manifest(suite_id).revision
    capacity_selected = "capacity" in track_ids
    concurrency = 2 if capacity_selected else 1
    mixture = _target_mixture()
    return RunManifest.from_semantic_fields(
        run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vllm-sr-evaluation:{run_id}")),
        name=f"Normalized target evaluation {run_id}",
        description="Execute one installed normalized workload on this runtime.",
        mode="live",
        target=EvaluationTarget(
            id=mixture.id,
            kind="mixture-of-models",
            router_api_url="http://router:8080",
            envoy_url="http://envoy:8801",
            backend_topology_digest=digest_value("target-topology-v1"),
            mixture=mixture,
        ),
        change_profile="runtime_capacity",
        gate_contract_version="evaluation-release-gates.v2",
        suite_ids=(suite_id,),
        suite_revisions={suite_id: revision},
        suite_executors={suite_id: NORMALIZED_LIVE_EXECUTOR_ID},
        track_ids=track_ids,  # type: ignore[arg-type]
        sample_limit=1,
        concurrency=concurrency,
        capacity_slo=(
            CapacitySLO(
                required_concurrency=1,
                max_latency_p95_ms=1000,
                max_error_rate=0.5,
                min_throughput_rps=0.1,
                min_throughput_scaling_efficiency=0.01,
            )
            if capacity_selected
            else None
        ),
        capacity_load_protocol=(
            default_capacity_load_protocol(concurrency) if capacity_selected else None
        ),
        seed=19,
        created_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        code_revision="sha256:" + "1" * 64,
        policy_snapshot_digest=digest_value("target-policy-v1"),
        config_digest=digest_value("target-config-v1"),
        redaction_policy="strict-no-prompts",
    )
