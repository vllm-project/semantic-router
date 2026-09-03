"""Immutable evaluation target, model-pool, and runtime connectivity contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import SecretRef, StrictModel
from cli.evaluation.contract_validation import (
    is_portable_id,
    is_subject_target_id,
    validate_http_origin,
    validate_portable_id,
)
from cli.evaluation.manifest_identity import (
    mixture_target_id,
    model_pool_snapshot_digest,
    routing_recipe_target_snapshot_digest,
    selector_snapshot_digest,
)
from cli.evaluation.routing_recipe_plan import RoutingRecipePlan, routing_recipe_top_k

_MAX_MIXTURE_ALIAS_BYTES = 512


class HTTPServiceEndpoint(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    url: str
    api_key: SecretRef | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return validate_http_origin(value, label="endpoint URL")


class ModelArm(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    model: str = Field(min_length=1)
    endpoint: HTTPServiceEndpoint | None = None
    input_cost_per_million_tokens_usd: float = Field(default=0, ge=0)
    output_cost_per_million_tokens_usd: float = Field(default=0, ge=0)

    _id = field_validator("id")(validate_portable_id)


class PoolDefinition(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    arm_ids: tuple[str, ...] = ()

    _id = field_validator("id")(validate_portable_id)

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

    _id = field_validator("id")(validate_portable_id)


class BindingSnapshot(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    policy_id: str
    pool_id: str

    _id = field_validator("id")(validate_portable_id)


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
    agent_task_ledger: HTTPServiceEndpoint | None = None
    fault_recovery_ledger: HTTPServiceEndpoint | None = None
    hard_policy_ledger: HTTPServiceEndpoint | None = None
    production_experiment_ledger: HTTPServiceEndpoint | None = None
    replay: HTTPServiceEndpoint | None = None
    currency: Literal["USD"] = "USD"

    _id = field_validator("id")(validate_portable_id)


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

    _id = field_validator("id")(validate_portable_id)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("model must be non-empty and already trimmed")
        return value

    @field_validator("capabilities", "modalities")
    @classmethod
    def unique_capability_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("model arm capability values must be unique")
        if any(not item or item.strip() != item for item in value):
            raise ValueError(
                "model arm capability values must be non-empty and already trimmed"
            )
        return value


class SupportModelIdentity(StrictModel):
    """Server-frozen executable identity for a selector-only model."""

    model: str = Field(min_length=1, max_length=512)
    provider_model_id_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_revision: str | None = Field(default=None, min_length=1, max_length=160)
    backend_topology_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("support model must already be trimmed")
        return value

    @field_validator("runtime_revision")
    @classmethod
    def validate_runtime_revision(cls, value: str | None) -> str | None:
        if value is not None and value.strip() != value:
            raise ValueError("support runtime revision must already be trimmed")
        return value


class MixtureDecisionBinding(StrictModel):
    """Frozen candidate boundary for one decision in a selected Recipe."""

    name: str = Field(min_length=1, max_length=160)
    algorithm: str = Field(min_length=1, max_length=160)
    arm_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("name", "algorithm")
    @classmethod
    def validate_trimmed_value(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("mixture decision values must already be trimmed")
        return value

    @field_validator("arm_ids")
    @classmethod
    def validate_arm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("mixture decision arm ids must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("mixture decision arm ids must use lexical order")
        for arm_id in value:
            validate_portable_id(arm_id)
        return value


class CatalogMixture(StrictModel):
    """Connectivity-free public summary of one frozen Mixture-of-Models."""

    id: str
    entrypoint_model: str = Field(min_length=1, max_length=512)
    aliases: tuple[str, ...] = Field(min_length=1)
    recipe_name: str = Field(min_length=1, max_length=160)
    recipe_description: str = Field(max_length=4000)
    recipe_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pool_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selector_policy_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selector_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adaptation_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    binding_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_arms: tuple[EvaluationTargetArm, ...] = Field(min_length=1)
    support_models: tuple[SupportModelIdentity, ...] = ()
    fallback_arm_id: str | None = None
    decisions: tuple[MixtureDecisionBinding, ...] = ()
    routing_recipe_plan: RoutingRecipePlan

    _id = field_validator("id")(validate_portable_id)

    @field_validator(
        "entrypoint_model",
        "recipe_name",
        "recipe_description",
    )
    @classmethod
    def validate_trimmed_text(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("mixture text fields must already be trimmed")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_model_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("mixture model names must be unique")
        if any(
            not item
            or item.strip() != item
            or len(item.encode("utf-8")) > _MAX_MIXTURE_ALIAS_BYTES
            for item in value
        ):
            raise ValueError("mixture model names must be non-empty trimmed values")
        return value

    @model_validator(mode="after")
    def validate_frozen_subject(self) -> CatalogMixture:
        self._validate_primary_identity()
        arm_ids, owners = self._validate_model_arms()
        self._validate_support_models(owners)
        self._validate_decisions(arm_ids)
        self._validate_snapshot_digests()
        self._validate_routing_recipe_plan(arm_ids)
        return self

    def _validate_primary_identity(self) -> None:
        if self.id != mixture_target_id(self.recipe_name):
            raise ValueError("mixture id must bind its recipe name")
        if self.entrypoint_model not in self.aliases:
            raise ValueError("mixture aliases must include the selected entrypoint")

    def _validate_model_arms(self) -> tuple[list[str], dict[str, str]]:
        arm_ids = [arm.id for arm in self.model_arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("mixture arm ids must be unique")
        arm_models = [arm.model for arm in self.model_arms]
        if len(arm_models) != len(set(arm_models)):
            raise ValueError("mixture arm models must be unique")
        owners: dict[str, str] = {}
        for arm in self.model_arms:
            for selector in {arm.id, arm.model}:
                owner = owners.setdefault(selector, arm.id)
                if owner != arm.id:
                    raise ValueError("mixture arm ids and models must be unambiguous")
        if arm_models != sorted(arm_models):
            raise ValueError("mixture model arms must be ordered by logical model")
        return arm_ids, owners

    def _validate_support_models(self, arm_owners: dict[str, str]) -> None:
        support_names = tuple(model.model for model in self.support_models)
        if len(support_names) != len(set(support_names)):
            raise ValueError("mixture support model identities must be unique")
        if set(support_names).intersection(arm_owners):
            raise ValueError(
                "mixture support models must remain outside the model pool"
            )
        if support_names != tuple(sorted(support_names)):
            raise ValueError("mixture support models must use lexical order")

    def _validate_decisions(self, arm_ids: list[str]) -> None:
        if self.fallback_arm_id is not None and self.fallback_arm_id not in arm_ids:
            raise ValueError("mixture fallback arm must belong to the model pool")
        decision_names = [decision.name for decision in self.decisions]
        if len(decision_names) != len(set(decision_names)):
            raise ValueError("mixture decision names must be unique")
        declared = set(arm_ids)
        if any(
            not set(decision.arm_ids).issubset(declared) for decision in self.decisions
        ):
            raise ValueError("mixture decisions may reference only declared arms")
        if any(not is_portable_id(decision.algorithm) for decision in self.decisions):
            raise ValueError("mixture decision algorithms must be portable identifiers")

    def _validate_snapshot_digests(self) -> None:
        if self.pool_digest != model_pool_snapshot_digest(self.model_arms):
            raise ValueError("mixture pool digest must bind its model arms")
        if self.selector_digest != selector_snapshot_digest(
            self.selector_policy_digest, self.support_models
        ):
            raise ValueError(
                "mixture selector digest must bind policy and support models"
            )

    def _validate_routing_recipe_plan(self, arm_ids: list[str]) -> None:
        plan = self.routing_recipe_plan
        expected_target_digest = routing_recipe_target_snapshot_digest(self)
        if plan.target_snapshot_digest != expected_target_digest:
            raise ValueError(
                "mixture routing recipe plan does not bind its immutable component digests"
            )
        if tuple(sorted(plan.arm_ids)) != tuple(sorted(arm_ids)):
            raise ValueError(
                "mixture routing recipe plan does not bind its frozen model pool"
            )
        if plan.fallback_arm_id != self.fallback_arm_id:
            raise ValueError(
                "mixture routing recipe plan does not bind its frozen fallback arm"
            )
        if plan.top_k != routing_recipe_top_k(len(arm_ids)):
            raise ValueError(
                "mixture routing recipe plan does not use the frozen pool top-k schedule"
            )
        if any(signal.value_kind != "numeric" for signal in plan.signals):
            raise ValueError("mixture routing recipe signals must be numeric")
        if any(
            projection.value_kind != "probability"
            or projection.outcome_binding != "selected_is_oracle"
            for projection in plan.projections
        ):
            raise ValueError(
                "mixture routing recipe projections must bind oracle-selection probability"
            )


class ManifestMixture(CatalogMixture):
    """Server-sealed execution subject embedded in a live run manifest."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION

    def public_summary(self) -> CatalogMixture:
        return CatalogMixture.model_validate(
            self.model_dump(mode="python", exclude={"schema_version"})
        )


class EvaluationTarget(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    id: str
    kind: str = Field(min_length=1, max_length=64)
    router_api_url: str | None = None
    envoy_url: str | None = None
    router_api_key: SecretRef | None = None
    envoy_api_key: SecretRef | None = None
    agent_task_ledger: HTTPServiceEndpoint | None = None
    fault_recovery_ledger: HTTPServiceEndpoint | None = None
    hard_policy_ledger: HTTPServiceEndpoint | None = None
    production_experiment_ledger: HTTPServiceEndpoint | None = None
    backend_topology_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    mixture: ManifestMixture | None = None

    _id = field_validator("id")(validate_portable_id)

    @field_validator("router_api_url", "envoy_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_origin(value, label="target URL")

    def credential_refs(self) -> tuple[SecretRef, ...]:
        """Return credential identities without resolving their values."""

        endpoint_refs = (
            endpoint.api_key
            for endpoint in (
                self.agent_task_ledger,
                self.fault_recovery_ledger,
                self.hard_policy_ledger,
                self.production_experiment_ledger,
            )
            if endpoint is not None
        )
        return tuple(
            ref
            for ref in (self.router_api_key, self.envoy_api_key, *endpoint_refs)
            if ref is not None
        )

    @model_validator(mode="after")
    def validate_runtime_connectivity(self) -> EvaluationTarget:
        if self.mixture is not None and (
            not is_subject_target_id(self.id, self.mixture.id)
            or self.kind != "mixture-of-models"
        ):
            raise ValueError("mixture target must bind its server-owned subject id")
        if self.router_api_key is not None and self.router_api_url is None:
            raise ValueError("router_api_key requires router_api_url")
        if self.envoy_api_key is not None and self.envoy_url is None:
            raise ValueError("envoy_api_key requires envoy_url")
        credential_envs = [ref.env for ref in self.credential_refs()]
        if len(credential_envs) != len(set(credential_envs)):
            raise ValueError(
                "evaluation credentials require distinct environment variables"
            )
        return self
