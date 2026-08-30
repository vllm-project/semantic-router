"""Strict, versioned input contracts for an evaluation run."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.gate_contract import GATE_CONTRACT_VERSION, ChangeProfile

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MAX_SUITE_REVISION_LENGTH = 160


class StrictModel(BaseModel):
    """Base contract that rejects silent schema drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(
            "must be a portable identifier (letters, digits, '.', '_' or '-')"
        )
    return value


class ArtifactRef(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(ge=0)


class SecretRef(StrictModel):
    """Credential reference; literal credentials are intentionally unsupported."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    env: str

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: str) -> str:
        if not _ENV_RE.fullmatch(value):
            raise ValueError(
                "secret env must be an uppercase environment variable name"
            )
        return value


class TextPart(StrictModel):
    type: Literal["text"] = "text"
    text: str


class ImageURL(StrictModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImagePart(StrictModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[ContentPart, ...]
    name: str | None = None
    tool_call_id: str | None = None


class CaseVisible(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    messages: tuple[Message, ...] = Field(min_length=1)
    modality: Literal["text", "image", "document", "audio", "video"] = "text"
    tags: tuple[str, ...] = ()
    trajectory_id: str | None = None

    _id = field_validator("id")(_validate_id)


class CaseGrading(StrictModel):
    """Hidden labels loaded only after policy/model execution."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    expected_route: str | None = None
    expected_answer: str | None = None
    preferred_arm_id: str | None = None
    expected_tools: tuple[str, ...] = ()
    should_block: bool | None = None
    weight: float = Field(default=1.0, gt=0)

    _case_id = field_validator("case_id")(_validate_id)


class VisibleCaseSet(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    cases: tuple[CaseVisible, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> VisibleCaseSet:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("visible case ids must be unique")
        return self


class GradingCaseSet(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    cases: tuple[CaseGrading, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> GradingCaseSet:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("grading case ids must be unique")
        return self


class WorkloadSnapshot(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    visible_cases: ArtifactRef
    grading_cases: ArtifactRef

    _id = field_validator("id")(_validate_id)

    @model_validator(mode="after")
    def physically_separated(self) -> WorkloadSnapshot:
        if self.visible_cases.digest == self.grading_cases.digest:
            raise ValueError("visible and grading cases must be separate artifacts")
        return self


class HTTPServiceEndpoint(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    url: str
    api_key: SecretRef | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "endpoint URL cannot contain credentials, query, or fragment"
            )
        return value.rstrip("/")


class ModelArm(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    model: str = Field(min_length=1)
    endpoint: HTTPServiceEndpoint | None = None
    input_cost_per_million_tokens_usd: float = Field(default=0, ge=0)
    output_cost_per_million_tokens_usd: float = Field(default=0, ge=0)

    _id = field_validator("id")(_validate_id)


class PoolDefinition(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    arm_ids: tuple[str, ...] = ()

    _id = field_validator("id")(_validate_id)

    @field_validator("arm_ids")
    @classmethod
    def unique_arms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("pool arm ids must be unique")
        return value


class PolicySnapshot(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    entrypoint_model: str = Field(min_length=1)
    recipe_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _id = field_validator("id")(_validate_id)


class BindingSnapshot(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    policy_id: str
    pool_id: str

    _id = field_validator("id")(_validate_id)


class RunEnvironment(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    target_id: str
    platform: str = Field(min_length=1)
    hardware_class: str = Field(min_length=1)
    backend_topology_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    route_eval: HTTPServiceEndpoint | None = None
    routed_chat: HTTPServiceEndpoint | None = None
    replay: HTTPServiceEndpoint | None = None
    currency: Literal["USD"] = "USD"

    _id = field_validator("id")(_validate_id)


class EvaluationTargetArm(StrictModel):
    """Server-owned public model identity and pricing, never connectivity."""

    id: str
    model: str = Field(min_length=1, max_length=512)
    provider_model_id_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_cost_per_million_tokens_usd: float = Field(
        ge=0, strict=True, allow_inf_nan=False
    )
    output_cost_per_million_tokens_usd: float = Field(
        ge=0, strict=True, allow_inf_nan=False
    )
    capabilities: tuple[str, ...] = ()
    modalities: tuple[Literal["text", "image", "document", "audio", "video"], ...] = ()
    context_window_tokens: int | None = Field(default=None, gt=0)
    parameter_size: str | None = Field(default=None, min_length=1, max_length=64)
    runtime_revision: str | None = Field(default=None, min_length=1, max_length=160)
    config_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    _id = field_validator("id")(_validate_id)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("capabilities", "modalities")
    @classmethod
    def unique_capability_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("model arm capability values must be unique")
        return value


class EvaluationTarget(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    kind: str = Field(min_length=1, max_length=64)
    router_api_url: str | None = None
    envoy_url: str | None = None
    router_api_key: SecretRef | None = None
    envoy_api_key: SecretRef | None = None
    backend_topology_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    model_arms: tuple[EvaluationTargetArm, ...] = ()

    _id = field_validator("id")(_validate_id)

    @field_validator("router_api_url", "envoy_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("target URL must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "target URL cannot contain credentials, query, or fragment"
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def unique_model_arms(self) -> EvaluationTarget:
        arm_ids = [arm.id for arm in self.model_arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("evaluation target arm ids must be unique")
        models = [arm.model for arm in self.model_arms]
        if len(models) != len(set(models)):
            raise ValueError("evaluation target arm models must be unique")
        return self


class RunManifest(StrictModel):
    """Fixed public worker manifest shared with the Dashboard backend."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str
    mode: Literal["replay", "live"]
    target: EvaluationTarget
    change_profile: ChangeProfile
    gate_contract_version: Literal[GATE_CONTRACT_VERSION]
    suite_ids: tuple[str, ...] = Field(min_length=1)
    suite_revisions: dict[str, str] = Field(min_length=1)
    track_ids: tuple[str, ...] = Field(min_length=1)
    sample_limit: int = Field(gt=0, le=100000)
    concurrency: int = Field(ge=1, le=128)
    seed: int = Field(ge=0, le=2**32 - 1)
    baseline_run_id: str | None = None
    created_at: datetime
    code_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
    policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    redaction_policy: str = Field(min_length=1, max_length=160)

    _run_id = field_validator("run_id")(_validate_id)

    @field_validator("track_ids")
    @classmethod
    def validate_tracks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(TRACK_IDS))
        if unknown:
            raise ValueError(f"unknown track ids: {', '.join(unknown)}")
        if len(value) != len(set(value)):
            raise ValueError("track ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> RunManifest:
        if set(self.suite_revisions) != set(self.suite_ids) or any(
            not revision.strip() or len(revision) > _MAX_SUITE_REVISION_LENGTH
            for revision in self.suite_revisions.values()
        ):
            raise ValueError(
                "suite_revisions must contain one non-empty immutable identity per suite_id"
            )
        if self.mode == "replay" and self.target.id != "fixture":
            raise ValueError("replay mode requires the catalog target 'fixture'")
        if self.mode == "live":
            if self.target.id != "runtime":
                raise ValueError("live mode requires the catalog target 'runtime'")
            if self.target.router_api_url is None and self.target.envoy_url is None:
                raise ValueError(
                    "live runtime target requires router_api_url or envoy_url"
                )
            if self.target.backend_topology_digest is None:
                raise ValueError("live runtime target requires backend_topology_digest")
        return self


class ExecutorMetadata(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    track_id: str
    executor_id: str
    mode: Literal["replay", "live"]

    @field_validator("track_id")
    @classmethod
    def validate_track(cls, value: str) -> str:
        if value not in TRACK_IDS:
            raise ValueError("unknown track id")
        return value


class ResolvedRunSnapshot(StrictModel):
    """Content-addressed run graph resolved from the fixed public manifest."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workload: WorkloadSnapshot
    policy: PolicySnapshot
    binding: BindingSnapshot
    pool: PoolDefinition
    arms: tuple[EvaluationTargetArm, ...]
    environment: RunEnvironment
    fixture_ref: ArtifactRef | None = None
    discovered_entrypoints: tuple[str, ...] = ()
    executors: tuple[ExecutorMetadata, ...]
