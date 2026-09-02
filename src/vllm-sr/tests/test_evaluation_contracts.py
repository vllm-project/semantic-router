from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from cli.evaluation.canonical import digest_value, sha256_digest
from cli.evaluation.catalog import EvaluationCatalog, get_catalog
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contracts import RunManifest
from cli.evaluation.executor_contracts import BUILTIN_EXECUTOR_CONTRACTS
from cli.evaluation.manifest_identity import (
    manifest_semantic_digest,
    mixture_snapshot_digest,
    mixture_target_id,
    model_pool_snapshot_digest,
    routing_recipe_plan_digest,
    seal_manifest_fields,
    selector_snapshot_digest,
)
from cli.evaluation.routing_recipe_plan import (
    RoutingRecipeInputSpec,
    RoutingRecipeProjectionSpec,
)
from cli.evaluation.target_capabilities import DEFAULT_TARGET_REGISTRY
from cli.evaluation.target_contracts import (
    CatalogMixture,
    EvaluationTarget,
    EvaluationTargetArm,
    ManifestMixture,
    MixtureDecisionBinding,
    SupportModelIdentity,
)
from cli.evaluation.worker_report import WorkerEvent, WorkerReportDraft
from evaluation_contract_test_support import build_routing_recipe_plan
from evaluation_schema_test_support import contract_schemas
from pydantic import ValidationError


def _golden(name: str) -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _mixture(arms: tuple[EvaluationTargetArm, ...]) -> ManifestMixture:
    recipe_name = "contract-recipe"
    recipe_digest = digest_value("contract-mixture-policy")
    pool_digest = model_pool_snapshot_digest(arms)
    mixture_id = mixture_target_id(recipe_name)
    aliases = ("entrypoint-contract",)
    selector_policy_digest = digest_value("contract-selector-policy")
    selector_digest = selector_snapshot_digest(selector_policy_digest, ())
    adaptation_digest = digest_value("contract-adaptation")
    binding_digest = digest_value("contract-mixture-binding")
    fallback_arm_id = arms[0].id
    return ManifestMixture(
        id=mixture_id,
        entrypoint_model="entrypoint-contract",
        aliases=aliases,
        recipe_name=recipe_name,
        recipe_description="Contract test recipe",
        recipe_digest=recipe_digest,
        pool_digest=pool_digest,
        selector_policy_digest=selector_policy_digest,
        selector_digest=selector_digest,
        adaptation_digest=adaptation_digest,
        binding_digest=binding_digest,
        model_arms=arms,
        support_models=(),
        fallback_arm_id=fallback_arm_id,
        decisions=(
            MixtureDecisionBinding(
                name="default",
                algorithm="static" if len(arms) > 1 else "single",
                arm_ids=tuple(sorted(arm.id for arm in arms)),
            ),
        ),
        routing_recipe_plan=build_routing_recipe_plan(
            recipe_digest=recipe_digest,
            pool_digest=pool_digest,
            selector_policy_digest=selector_policy_digest,
            selector_digest=selector_digest,
            adaptation_digest=adaptation_digest,
            binding_digest=binding_digest,
            arm_ids=tuple(sorted({arm.id for arm in arms})),
            fallback_arm_id=fallback_arm_id,
            signals=(),
            projections=(),
        ),
    )


def _assert_catalog_evidence_contract(
    catalog: EvaluationCatalog, report: WorkerReportDraft
) -> None:
    fixture = next(target for target in catalog.targets if target.id == "fixture")
    assert tuple(target.id for target in catalog.targets) == (
        "fixture",
        "benchmark-source",
    )
    assert fixture.evidence_level == "E0"
    assert report.run.evidence_level == "E0"
    assert tuple(suite.id for suite in catalog.suites) == (
        "evaluation-smoke",
        "live-mom-core",
        "live-agent-tasks",
        "live-fault-recovery",
        "live-multimodal",
        "live-hard-policy",
        "live-production-experiment",
        "live-capacity",
    )
    assert {suite.id: suite.evidence_level for suite in catalog.suites} == {
        "evaluation-smoke": "E0",
        "live-mom-core": "E0",
        "live-agent-tasks": "E5",
        "live-fault-recovery": "E5",
        "live-multimodal": "E0",
        "live-hard-policy": "E4",
        "live-production-experiment": "E5",
        "live-capacity": "E0",
    }
    assert all(suite.methods for suite in catalog.suites)
    expected_slots = tuple(f"G{index}" for index in range(2, 10))
    for profile in catalog.change_profiles:
        assert tuple(slot.gate_id for slot in profile.campaign_slots) == expected_slots
    g4_slots = {
        profile.id: next(
            slot for slot in profile.campaign_slots if slot.gate_id == "G4"
        )
        for profile in catalog.change_profiles
    }
    assert all(slot.mode == "live" for slot in g4_slots.values())
    assert all(slot.minimum_evidence_level == "E4" for slot in g4_slots.values())
    assert all(
        slot.accepted_executor_ids == ("normalized-suite-live.v1",)
        for slot in g4_slots.values()
    )
    _assert_fidelity_slots(catalog)


def _assert_fidelity_slots(catalog: EvaluationCatalog) -> None:
    g5_slots = {
        profile.id: next(
            slot for slot in profile.campaign_slots if slot.gate_id == "G5"
        )
        for profile in catalog.change_profiles
    }
    assert all(slot.mode == "live" for slot in g5_slots.values())
    assert g5_slots["agent_multimodal"].track_id == "multimodal"
    assert g5_slots["agent_multimodal"].minimum_evidence_level == "E4"
    assert g5_slots["agent_multimodal"].accepted_executor_ids == (
        "normalized-suite-live.v1",
    )
    assert all(
        slot.track_id == "joint"
        and slot.minimum_evidence_level == "E5"
        and slot.accepted_executor_ids == ("live-runtime.v1",)
        for profile, slot in g5_slots.items()
        if profile != "agent_multimodal"
    )


def test_cross_language_golden_contracts_parse_strictly() -> None:
    catalog_payload = _golden("catalog.json")
    manifest_payload = _golden("manifest.json")
    catalog = EvaluationCatalog.model_validate(catalog_payload)
    manifest = RunManifest.model_validate(manifest_payload)
    live_manifest = RunManifest.model_validate(_golden("live-manifest.json"))
    report = WorkerReportDraft.model_validate(_golden("worker-report-draft.json"))

    for model, payload in (
        (EvaluationCatalog, catalog_payload),
        (RunManifest, manifest_payload),
    ):
        without_version = dict(payload)
        without_version.pop("schema_version")
        with pytest.raises(ValidationError, match="schema_version"):
            model.model_validate(without_version)

    assert catalog.schema_version == manifest.schema_version == report.schema_version
    assert catalog.schema_version == SCHEMA_VERSION == "evaluation.v1"
    assert tuple(track.id for track in catalog.tracks) == TRACK_IDS
    assert report.run.id == manifest.run_id
    assert report.run.client_request_id == report.run.id
    assert report.run.name == manifest.name
    assert report.run.description == manifest.description
    assert manifest_semantic_digest(manifest) == manifest.manifest_digest
    assert manifest_semantic_digest(live_manifest) == live_manifest.manifest_digest
    assert live_manifest.target.mixture is not None
    assert live_manifest.target.mixture.model_arms[0].model == "public-fast"
    assert live_manifest.target.router_api_key is None
    assert live_manifest.target.envoy_api_key is not None
    assert live_manifest.target.envoy_api_key.env == "VLLM_SR_ENVOY_EVAL_API_KEY"
    assert (
        live_manifest.target.mixture.model_arms[0].input_cost_per_million_tokens_usd
        == 0.0
    )
    assert (
        live_manifest.target.mixture.model_arms[0].output_cost_per_million_tokens_usd
        == 1.0
    )
    assert (
        live_manifest.target.mixture.model_arms[1].input_cost_per_million_tokens_usd
        == 1e-7
    )
    assert (
        math.copysign(
            1.0,
            live_manifest.target.mixture.model_arms[
                1
            ].output_cost_per_million_tokens_usd,
        )
        == -1.0
    )
    assert live_manifest.target.mixture.model_arms[1].model == "公共-strong-模型"
    assert live_manifest.created_at.microsecond == 123456
    assert report.summary.verdict == "unavailable"
    assert report.summary.failed_gates == 0
    assert {gate.id: gate.verdict for gate in report.gates}["G8"] == "not_applicable"
    assert {gate.id: gate.verdict for gate in report.gates}["G9"] == "not_applicable"
    _assert_catalog_evidence_contract(catalog, report)
    assert get_catalog(generated_at=False).model_dump(
        mode="json", exclude_none=True
    ) == _golden("catalog.json")


def test_generated_json_schema_matches_versioned_golden_digests() -> None:
    golden = _golden("schema-digests.json")
    actual = {name: digest_value(schema) for name, schema in contract_schemas().items()}
    assert golden["schema_version"] == SCHEMA_VERSION
    assert actual == golden["digests"]


def test_manifest_is_strict_and_has_no_literal_secret_surface() -> None:
    payload = _golden("manifest.json")
    missing_digest = dict(payload)
    missing_digest.pop("manifest_digest")
    with pytest.raises(ValidationError, match="manifest_digest"):
        RunManifest.model_validate(missing_digest)
    with pytest.raises(ValidationError, match="manifest_digest"):
        RunManifest.model_validate({**payload, "manifest_digest": "sha256:caller"})

    payload["api_key"] = "literal-secret"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunManifest.model_validate(payload)

    target = dict(payload["target"])
    target["router_api_url"] = "https://user:secret@example.com/router"
    with pytest.raises(ValidationError, match="credentials"):
        EvaluationTarget.model_validate(target)

    for key_field in ("router_api_key", "envoy_api_key"):
        with pytest.raises(ValidationError):
            EvaluationTarget.model_validate(
                {
                    "id": "runtime",
                    "kind": "runtime",
                    key_field: "literal-secret",
                }
            )
        with pytest.raises(ValidationError, match="uppercase environment variable"):
            EvaluationTarget.model_validate(
                {
                    "id": "runtime",
                    "kind": "runtime",
                    key_field: {"env": "literal-secret"},
                }
            )

    target_with_refs = EvaluationTarget.model_validate(
        {
            "id": "runtime",
            "kind": "runtime",
            "router_api_url": "http://router:8080",
            "envoy_url": "http://envoy:8801",
            "router_api_key": {"env": "ROUTER_EVAL_API_KEY"},
            "envoy_api_key": {"env": "ENVOY_EVAL_API_KEY"},
        }
    )
    assert target_with_refs.router_api_key is not None
    assert target_with_refs.router_api_key.env == "ROUTER_EVAL_API_KEY"
    assert target_with_refs.envoy_api_key is not None
    assert target_with_refs.envoy_api_key.env == "ENVOY_EVAL_API_KEY"


def test_manifest_identity_collections_are_canonical_and_unambiguous() -> None:
    payload = _golden("manifest.json")
    parsed = RunManifest.model_validate(payload)
    assert parsed.run_id == "00000000-0000-4000-8000-000000000001"
    assert parsed.suite_ids == tuple(payload["suite_ids"])
    assert parsed.track_ids == tuple(payload["track_ids"])

    for run_id in (
        "fixture-run",
        "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        parsed.run_id.replace("-", ""),
    ):
        with pytest.raises(ValidationError, match="canonical UUID"):
            RunManifest.model_validate({**payload, "run_id": run_id})
    with pytest.raises(ValidationError, match="suite ids must be unique"):
        RunManifest.model_validate(
            {**payload, "suite_ids": ["evaluation-smoke", "evaluation-smoke"]}
        )
    with pytest.raises(ValidationError, match="track ids must be unique"):
        RunManifest.model_validate({**payload, "track_ids": ["routing", "routing"]})

    mixed = {
        **payload,
        "suite_ids": ["suite-a", "suite-b"],
        "suite_revisions": {"suite-a": "revision-a", "suite-b": "revision-b"},
        "suite_executors": {
            "suite-a": "fixture-replay.v1",
            "suite-b": "normalized-suite-replay.v1",
        },
    }
    with pytest.raises(ValidationError, match="cannot mix executor"):
        RunManifest.model_validate(mixed)

    reversed_tracks = {**payload, "track_ids": list(reversed(payload["track_ids"]))}
    with pytest.raises(ValidationError, match="canonical catalog order"):
        RunManifest.model_validate(reversed_tracks)

    live = _golden("live-manifest.json")
    reversed_suites = {**live, "suite_ids": list(reversed(live["suite_ids"]))}
    with pytest.raises(ValidationError, match="canonical catalog order"):
        RunManifest.model_validate(reversed_suites)

    installed = {
        **payload,
        "suite_ids": ["installed-z", "installed-a"],
        "suite_revisions": {"installed-z": "v1", "installed-a": "v1"},
        "suite_executors": {
            "installed-z": "normalized-suite-replay.v1",
            "installed-a": "normalized-suite-replay.v1",
        },
    }
    with pytest.raises(ValidationError, match="lexical canonical order"):
        RunManifest.model_validate(installed)

    for field, value, error in (
        ("name", " padded ", "run name"),
        ("description", " padded ", "run description"),
    ):
        with pytest.raises(ValidationError, match=error):
            RunManifest.model_validate({**payload, field: value})

    report = _golden("worker-report-draft.json")
    report["run"] = {
        **report["run"],
        "client_request_id": "00000000-0000-4000-8000-000000000099",
    }
    with pytest.raises(ValidationError, match="must equal the run id"):
        WorkerReportDraft.model_validate(report)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "Tampered evaluation name"),
        ("description", "Tampered evaluation description"),
        ("suite_executors", {"evaluation-smoke": "fixture-replay.v2"}),
    ),
)
def test_manifest_semantic_digest_rejects_field_tampering(
    field: str, value: object
) -> None:
    payload = _golden("manifest.json")
    payload[field] = value

    with pytest.raises(ValidationError, match="manifest_digest does not match"):
        RunManifest.model_validate(payload)


def test_live_manifest_semantic_digest_binds_every_capacity_slo_bound() -> None:
    payload = _golden("live-manifest.json")
    frozen_slo = dict(payload["capacity_slo"])
    tampered_values = {
        "required_concurrency": (2 if frozen_slo["required_concurrency"] == 1 else 1),
        "max_latency_p95_ms": frozen_slo["max_latency_p95_ms"] * 1.1,
        "max_error_rate": (
            frozen_slo["max_error_rate"] / 2
            if frozen_slo["max_error_rate"] > 0
            else 0.001
        ),
        "min_throughput_rps": frozen_slo["min_throughput_rps"] * 1.1,
        "min_throughput_scaling_efficiency": frozen_slo[
            "min_throughput_scaling_efficiency"
        ]
        / 2,
    }
    for field, tampered_value in tampered_values.items():
        tampered_slo = {**frozen_slo, field: tampered_value}
        with pytest.raises(ValidationError, match="manifest_digest does not match"):
            RunManifest.model_validate({**payload, "capacity_slo": tampered_slo})


def test_live_manifest_semantic_digest_binds_every_capacity_protocol_field() -> None:
    payload = _golden("live-manifest.json")
    frozen = dict(payload["capacity_load_protocol"])
    tampered_values = {
        "kind": "other",
        "concurrency_levels": [1, 4],
        "warmup_request_multiplier": frozen["warmup_request_multiplier"] + 1,
        "measurement_requests_per_repetition": (
            frozen["measurement_requests_per_repetition"] + 1
        ),
        "repetitions_per_level": frozen["repetitions_per_level"] + 1,
        "confidence_level": 0.9,
        "max_throughput_cv": frozen["max_throughput_cv"] / 2,
        "max_latency_p95_cv": frozen["max_latency_p95_cv"] / 2,
    }
    for field, tampered_value in tampered_values.items():
        tampered = {**frozen, field: tampered_value}
        assert (
            manifest_semantic_digest({**payload, "capacity_load_protocol": tampered})
            != payload["manifest_digest"]
        )


def test_routing_recipe_plan_is_required_strict_and_fully_bound() -> None:
    arms = (
        EvaluationTargetArm(
            id="fast",
            model="org/fast",
            provider_model_id_digest=sha256_digest(b"org/fast"),
            input_cost_per_million_tokens_usd=0.1,
            output_cost_per_million_tokens_usd=0.2,
        ),
        EvaluationTargetArm(
            id="strong",
            model="org/strong",
            provider_model_id_digest=sha256_digest(b"org/strong"),
            input_cost_per_million_tokens_usd=0.3,
            output_cost_per_million_tokens_usd=0.4,
        ),
    )
    mixture = _mixture(arms)
    payload = mixture.model_dump(mode="json", exclude_none=True)

    missing = copy.deepcopy(payload)
    missing.pop("routing_recipe_plan")
    with pytest.raises(ValidationError, match="routing_recipe_plan"):
        ManifestMixture.model_validate(missing)
    public_missing = mixture.public_summary().model_dump(mode="json", exclude_none=True)
    public_missing.pop("routing_recipe_plan")
    with pytest.raises(ValidationError, match="routing_recipe_plan"):
        CatalogMixture.model_validate(public_missing)

    extra = copy.deepcopy(payload)
    extra["routing_recipe_plan"]["outcome"] = "forged"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ManifestMixture.model_validate(extra)

    stale_digest = copy.deepcopy(payload)
    stale_digest["routing_recipe_plan"]["arm_ids"] = ["strong", "fast"]
    stale_digest["routing_recipe_plan"]["plan_digest"] = sha256_digest(b"stale-plan")
    with pytest.raises(ValidationError, match="does not bind its canonical body"):
        ManifestMixture.model_validate(stale_digest)

    detached_target = copy.deepcopy(payload)
    detached_target["routing_recipe_plan"]["target_snapshot_digest"] = sha256_digest(
        b"detached-target"
    )
    detached_target["routing_recipe_plan"]["plan_digest"] = routing_recipe_plan_digest(
        detached_target["routing_recipe_plan"]
    )
    with pytest.raises(
        ValidationError, match="does not bind its immutable component digests"
    ):
        ManifestMixture.model_validate(detached_target)

    truncated_top_k = copy.deepcopy(payload)
    truncated_top_k["routing_recipe_plan"]["top_k"] = [1]
    truncated_top_k["routing_recipe_plan"]["plan_digest"] = routing_recipe_plan_digest(
        truncated_top_k["routing_recipe_plan"]
    )
    with pytest.raises(ValidationError, match="frozen pool top-k schedule"):
        ManifestMixture.model_validate(truncated_top_k)

    nonnumeric_signal = copy.deepcopy(payload)
    nonnumeric_signal["routing_recipe_plan"]["signals"] = [
        {"id": "context:turns", "value_kind": "none"}
    ]
    nonnumeric_signal["routing_recipe_plan"]["plan_digest"] = (
        routing_recipe_plan_digest(nonnumeric_signal["routing_recipe_plan"])
    )
    with pytest.raises(ValidationError, match="signals must be numeric"):
        ManifestMixture.model_validate(nonnumeric_signal)


def test_routing_recipe_plan_manifest_identity_is_canonical_and_complete() -> None:
    payload = _golden("live-manifest.json")
    parsed = RunManifest.model_validate(payload)
    assert parsed.target.mixture is not None
    plan = parsed.target.mixture.routing_recipe_plan
    assert plan.plan_digest == (
        "sha256:1f3a6ccdafe32e7b2cf84b077431596c845a0c4c5b77c8da35d1bbf487c1c24c"
    )
    assert plan.target_snapshot_digest == (
        "sha256:5b8d499933f180ca9877c2cfc99bb718403c3e8feaa08227e38c4e8b9907bb9e"
    )

    permuted = copy.deepcopy(payload)
    permuted_plan = permuted["target"]["mixture"]["routing_recipe_plan"]
    permuted_plan["arm_ids"].reverse()
    permuted_plan["signals"].reverse()
    assert routing_recipe_plan_digest(permuted_plan) == plan.plan_digest
    assert manifest_semantic_digest(permuted) == payload["manifest_digest"]
    assert RunManifest.model_validate(permuted).manifest_digest == (
        payload["manifest_digest"]
    )
    assert mixture_snapshot_digest(permuted["target"]["mixture"]) != (
        mixture_snapshot_digest(payload["target"]["mixture"])
    )

    changed = copy.deepcopy(payload)
    changed_plan = changed["target"]["mixture"]["routing_recipe_plan"]
    changed_plan["signals"].append(
        RoutingRecipeInputSpec(id="context:turns", value_kind="numeric").model_dump(
            mode="json"
        )
    )
    changed_plan["projections"].append(
        RoutingRecipeProjectionSpec(
            id="projection:quality",
            value_kind="probability",
            outcome_binding="selected_is_oracle",
        ).model_dump(mode="json")
    )
    changed_plan["plan_digest"] = routing_recipe_plan_digest(changed_plan)
    assert manifest_semantic_digest(changed) != payload["manifest_digest"]


def test_manifest_target_shape_is_exact_for_each_execution_mode() -> None:
    replay = _golden("manifest.json")
    replay_target = dict(replay["target"])
    replay_target["backend_topology_digest"] = sha256_digest(b"unexpected")
    tampered = seal_manifest_fields(
        {
            **{key: value for key, value in replay.items() if key != "manifest_digest"},
            "target": replay_target,
        }
    )
    fixture_executor = next(
        executor
        for executor in BUILTIN_EXECUTOR_CONTRACTS
        if executor.id == "fixture-replay.v1"
    )
    with pytest.raises(ValueError, match="runtime connectivity"):
        DEFAULT_TARGET_REGISTRY.resolve(
            RunManifest.model_validate(tampered), fixture_executor
        )

    live = _golden("live-manifest.json")
    live_target = dict(live["target"])
    live_target["router_api_key"] = {
        "schema_version": SCHEMA_VERSION,
        "env": "ROUTER_EVAL_API_KEY",
    }
    live_with_router_credential = seal_manifest_fields(
        {
            **{key: value for key, value in live.items() if key != "manifest_digest"},
            "target": live_target,
        }
    )
    live_executor = next(
        executor
        for executor in BUILTIN_EXECUTOR_CONTRACTS
        if executor.id == "live-runtime.v1"
    )
    secret = "server-owned-router-evaluation-secret"
    authenticated_live = RunManifest.model_validate(live_with_router_credential)
    resolved = DEFAULT_TARGET_REGISTRY.resolve(authenticated_live, live_executor)
    assert resolved.execution_profile == "brokered-runtime"
    assert authenticated_live.target.router_api_key is not None
    assert authenticated_live.target.router_api_key.env == "ROUTER_EVAL_API_KEY"
    serialized = authenticated_live.model_dump_json(exclude_none=True)
    assert "ROUTER_EVAL_API_KEY" in serialized
    assert secret not in serialized


def test_worker_event_payload_is_event_specific_and_scalar_only() -> None:
    track = WorkerEvent(
        type="track", message="Track complete", payload={"record_count": 4}
    )
    completed = WorkerEvent(
        type="completed", message="Run complete", payload={"verdict": "pass"}
    )
    assert track.payload is not None and track.payload.record_count == 4
    assert completed.payload is not None and completed.payload.verdict == "pass"

    invalid_events = (
        {"type": "track", "message": "Track complete"},
        {
            "type": "track",
            "message": "Track complete",
            "payload": {"record_count": -1},
        },
        {
            "type": "completed",
            "message": "Run complete",
            "payload": {"verdict": "maybe"},
        },
        {
            "type": "completed",
            "message": "Run complete",
            "payload": {"verdict": "not_applicable"},
        },
        {
            "type": "completed",
            "message": "Run complete",
            "payload": {"verdict": "waived"},
        },
        {
            "type": "failed",
            "message": "Run failed",
            "payload": {"record_count": 1},
        },
    )
    for event in invalid_events:
        with pytest.raises(ValidationError):
            WorkerEvent.model_validate(event)


@pytest.mark.parametrize(
    "revision", ("main", "latest", "unavailable", "deadbeef", "commit-abc123")
)
def test_manifest_requires_an_immutable_full_source_revision(revision: str) -> None:
    payload = _golden("manifest.json")
    payload["code_revision"] = revision

    with pytest.raises(ValidationError, match="code_revision"):
        RunManifest.model_validate(payload)


def test_target_model_arms_are_server_owned_strict_identity_and_pricing() -> None:
    arm = {
        "id": "reasoning",
        "model": "org/reasoning-model",
        "provider_model_id_digest": sha256_digest(b"private/provider-id"),
        "input_cost_per_million_tokens_usd": 1.25,
        "output_cost_per_million_tokens_usd": 4.5,
        "capabilities": ["chat", "reasoning"],
        "modalities": ["text"],
        "context_window_tokens": 32768,
        "parameter_size": "70B",
        "runtime_revision": "runtime-v1",
        "config_digest": sha256_digest(b"config"),
    }
    parsed = EvaluationTargetArm.model_validate(arm)
    assert parsed.input_cost_per_million_tokens_usd == 1.25
    assert parsed.output_cost_per_million_tokens_usd == 4.5
    assert parsed.provider_model_id_digest == sha256_digest(b"private/provider-id")
    assert parsed.modalities == ("text",)

    for forbidden_field in ("provider_model_id", "endpoint", "api_key", "secret"):
        unsafe = {**arm, forbidden_field: "https://private.example.test"}
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationTargetArm.model_validate(unsafe)

    for invalid_price in (-0.01, float("inf"), float("nan"), "1.25"):
        invalid = {**arm, "input_cost_per_million_tokens_usd": invalid_price}
        with pytest.raises(ValidationError):
            EvaluationTargetArm.model_validate(invalid)


def test_target_model_arm_identity_and_model_are_unique() -> None:
    first = EvaluationTargetArm(
        id="fast",
        model="org/fast",
        provider_model_id_digest=sha256_digest(b"org/private-fast"),
        input_cost_per_million_tokens_usd=0.1,
        output_cost_per_million_tokens_usd=0.2,
    )
    duplicate_id = first.model_copy(update={"model": "org/other"})
    duplicate_model = first.model_copy(update={"id": "other"})

    with pytest.raises(ValidationError, match="arm ids must be unique"):
        _mixture((first, duplicate_id))
    with pytest.raises(ValidationError, match="arm models must be unique"):
        _mixture((first, duplicate_model))
    colliding_selector = first.model_copy(update={"id": "other", "model": first.id})
    with pytest.raises(ValidationError, match="ids and models must be unambiguous"):
        _mixture((first, colliding_selector))


def test_selector_support_identity_is_strict_and_digest_bound() -> None:
    arm = EvaluationTargetArm(
        id="fast",
        model="org/fast",
        provider_model_id_digest=sha256_digest(b"org/private-fast"),
        input_cost_per_million_tokens_usd=0.1,
        output_cost_per_million_tokens_usd=0.2,
    )
    mixture = _mixture((arm,))
    support = SupportModelIdentity(
        model="org/selector",
        provider_model_id_digest=sha256_digest(b"private/selector-v1"),
        config_digest=sha256_digest(b"selector-config-v1"),
        runtime_revision="runtime-v1",
        backend_topology_digest=sha256_digest(b"private-selector-endpoint"),
    )
    payload = mixture.model_dump(mode="python")
    payload["support_models"] = (support,)
    payload["selector_digest"] = selector_snapshot_digest(
        mixture.selector_policy_digest, (support,)
    )
    payload["routing_recipe_plan"] = build_routing_recipe_plan(
        recipe_digest=mixture.recipe_digest,
        pool_digest=mixture.pool_digest,
        selector_policy_digest=mixture.selector_policy_digest,
        selector_digest=payload["selector_digest"],
        adaptation_digest=mixture.adaptation_digest,
        binding_digest=mixture.binding_digest,
        arm_ids=tuple(arm.id for arm in mixture.model_arms),
        fallback_arm_id=mixture.fallback_arm_id,
        signals=mixture.routing_recipe_plan.signals,
        projections=mixture.routing_recipe_plan.projections,
    )
    parsed = ManifestMixture.model_validate(payload)
    assert parsed.support_models == (support,)

    payload["selector_digest"] = sha256_digest(b"unbound-selector")
    with pytest.raises(ValidationError, match="must bind policy and support models"):
        ManifestMixture.model_validate(payload)

    unsafe = {
        **support.model_dump(mode="python"),
        "endpoint": "https://private.example.test",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SupportModelIdentity.model_validate(unsafe)


def test_mixture_target_identity_supports_explicit_deployment_scope() -> None:
    mixture = _mixture(
        (
            EvaluationTargetArm(
                id="contract-arm",
                model="contract/model",
                provider_model_id_digest=digest_value("contract/provider-model"),
                modalities=("text",),
                capabilities=("chat", "text"),
                context_window_tokens=4096,
                input_cost_per_million_tokens_usd=0.1,
                output_cost_per_million_tokens_usd=0.2,
                config_digest=digest_value("contract-arm"),
            ),
        )
    )
    standalone = EvaluationTarget(
        id=mixture.id,
        kind="mixture-of-models",
        mixture=mixture,
    )
    scoped = standalone.model_copy(update={"id": f"candidate--{mixture.id}"})

    assert EvaluationTarget.model_validate(scoped.model_dump()) == scoped
    with pytest.raises(ValidationError, match="server-owned subject id"):
        EvaluationTarget.model_validate(
            {
                **standalone.model_dump(mode="json"),
                "id": "candidate--unrelated-mixture",
            }
        )
