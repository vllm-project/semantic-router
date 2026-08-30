from __future__ import annotations

import json
import math
from importlib.resources import files

import pytest
from cli.evaluation.canonical import digest_value, sha256_digest
from cli.evaluation.catalog import EvaluationCatalog, get_catalog
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contracts import (
    ArtifactRef,
    EvaluationTarget,
    EvaluationTargetArm,
    RunManifest,
    WorkloadSnapshot,
)
from cli.evaluation.reporting import EvaluationReport
from cli.evaluation.schemas import contract_schemas
from pydantic import ValidationError


def _golden(name: str) -> dict[str, object]:
    path = files("cli.evaluation").joinpath("golden", name)
    return json.loads(path.read_text(encoding="utf-8"))


def test_cross_language_golden_contracts_parse_strictly() -> None:
    catalog = EvaluationCatalog.model_validate(_golden("catalog.json"))
    manifest = RunManifest.model_validate(_golden("manifest.json"))
    live_manifest = RunManifest.model_validate(_golden("live-manifest.json"))
    report = EvaluationReport.model_validate(_golden("report.json"))

    assert catalog.schema_version == manifest.schema_version == report.schema_version
    assert catalog.schema_version == SCHEMA_VERSION == "evaluation.v1"
    assert tuple(track.id for track in catalog.tracks) == TRACK_IDS
    assert report.run.id == manifest.run_id
    assert live_manifest.target.model_arms[0].model == "public-fast"
    assert live_manifest.target.router_api_key is None
    assert live_manifest.target.envoy_api_key is not None
    assert live_manifest.target.envoy_api_key.env == "VLLM_SR_ENVOY_EVAL_API_KEY"
    assert live_manifest.target.model_arms[0].input_cost_per_million_tokens_usd == 0.0
    assert live_manifest.target.model_arms[0].output_cost_per_million_tokens_usd == 1.0
    assert live_manifest.target.model_arms[1].input_cost_per_million_tokens_usd == 1e-7
    assert (
        math.copysign(
            1.0,
            live_manifest.target.model_arms[1].output_cost_per_million_tokens_usd,
        )
        == -1.0
    )
    assert live_manifest.target.model_arms[1].model == "公共-strong-模型"
    assert live_manifest.created_at.microsecond == 123456
    assert report.summary.verdict == "unavailable"
    assert report.summary.failed_gates == 0
    assert {gate.id: gate.verdict for gate in report.gates}["G8"] == "not_applicable"
    assert {gate.id: gate.verdict for gate in report.gates}["G9"] == "not_applicable"
    runtime = next(target for target in catalog.targets if target.id == "runtime")
    fixture = next(target for target in catalog.targets if target.id == "fixture")
    assert runtime.track_ids == ()
    assert runtime.healthy is False
    assert runtime.evidence_level is None
    assert fixture.evidence_level == "E0"
    assert report.run.evidence_level == "E0"
    assert tuple(suite.id for suite in catalog.suites) == (
        "evaluation-smoke",
        "live-routing-core",
        "live-model-pool",
        "live-joint",
        "live-multimodal",
        "live-capacity",
    )
    assert all(suite.evidence_level == "E0" for suite in catalog.suites)
    assert runtime.labels == {
        "capabilities": "manifest-dependent",
        "credentials": "environment-only",
        "direct_arms": "unavailable",
        "model_arms": "server-owned",
    }
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
            "router_api_key": {"env": "ROUTER_EVAL_API_KEY"},
            "envoy_api_key": {"env": "ENVOY_EVAL_API_KEY"},
        }
    )
    assert target_with_refs.router_api_key is not None
    assert target_with_refs.router_api_key.env == "ROUTER_EVAL_API_KEY"
    assert target_with_refs.envoy_api_key is not None
    assert target_with_refs.envoy_api_key.env == "ENVOY_EVAL_API_KEY"


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
        EvaluationTarget(id="runtime", kind="runtime", model_arms=(first, duplicate_id))
    with pytest.raises(ValidationError, match="arm models must be unique"):
        EvaluationTarget(
            id="runtime", kind="runtime", model_arms=(first, duplicate_model)
        )


def test_runtime_catalog_tracks_are_capability_dependent() -> None:
    arm = EvaluationTargetArm(
        id="fast",
        model="org/fast",
        provider_model_id_digest=sha256_digest(b"org/private-fast"),
        input_cost_per_million_tokens_usd=0.1,
        output_cost_per_million_tokens_usd=0.2,
    )
    cases = (
        ({}, (), False),
        ({"router_api_url": "http://router:8080"}, ("routing",), True),
        ({"envoy_url": "http://envoy:8801"}, ("capacity",), True),
        (
            {"envoy_url": "http://envoy:8801", "model_arms": (arm,)},
            ("capacity",),
            True,
        ),
        (
            {
                "router_api_url": "http://router:8080",
                "envoy_url": "http://envoy:8801",
                "model_arms": (
                    arm,
                    arm.model_copy(
                        update={
                            "id": "vision",
                            "model": "org/vision",
                            "modalities": ("text", "image"),
                        }
                    ),
                ),
            },
            ("routing", "multimodal", "capacity"),
            True,
        ),
    )
    for capabilities, expected_tracks, expected_health in cases:
        catalog = get_catalog(generated_at=False, **capabilities)
        runtime = next(target for target in catalog.targets if target.id == "runtime")
        assert runtime.track_ids == expected_tracks
        assert runtime.healthy is expected_health
        assert not {"agentic", "preference", "safety"} & set(runtime.track_ids)


def test_live_manifest_accepts_capability_subsets_but_not_an_empty_target() -> None:
    payload = _golden("manifest.json")
    payload.update(
        {
            "mode": "live",
            "target": {
                "schema_version": SCHEMA_VERSION,
                "id": "runtime",
                "kind": "runtime",
                "router_api_url": "http://router:8080",
                "backend_topology_digest": sha256_digest(b"backend-topology"),
                "model_arms": [
                    {
                        "id": "fast",
                        "model": "org/fast",
                        "provider_model_id_digest": sha256_digest(b"org/private-fast"),
                        "input_cost_per_million_tokens_usd": 0.1,
                        "output_cost_per_million_tokens_usd": 0.2,
                    }
                ],
            },
        }
    )
    assert RunManifest.model_validate(payload).target.envoy_url is None
    assert RunManifest.model_validate(payload).target.model_arms[0].id == "fast"
    missing_topology = dict(payload)
    missing_topology["target"] = {
        key: value
        for key, value in dict(payload["target"]).items()
        if key != "backend_topology_digest"
    }
    with pytest.raises(ValidationError, match="backend_topology_digest"):
        RunManifest.model_validate(missing_topology)
    payload["target"] = {
        "schema_version": SCHEMA_VERSION,
        "id": "runtime",
        "kind": "runtime",
    }
    with pytest.raises(ValidationError, match="router_api_url or envoy_url"):
        RunManifest.model_validate(payload)


def test_visible_and_grading_case_artifacts_must_be_physically_separate() -> None:
    ref = ArtifactRef(
        digest="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=10,
    )
    with pytest.raises(ValidationError, match="separate artifacts"):
        WorkloadSnapshot(id="hidden-label-check", visible_cases=ref, grading_cases=ref)


def test_canonical_digest_is_key_order_independent() -> None:
    assert digest_value({"b": 2, "a": [3, 1]}) == digest_value({"a": [3, 1], "b": 2})
