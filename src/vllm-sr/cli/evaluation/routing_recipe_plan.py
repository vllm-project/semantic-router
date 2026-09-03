"""Strict decision-time plan contract for one frozen routing Recipe."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from cli.evaluation.manifest_identity import (
    routing_recipe_plan_digest,
)

ROUTING_RECIPE_PLAN_CONTRACT_VERSION = "routing-recipe-plan.v1"

_MAX_ROUTING_RECIPE_ITEMS = 128
_MAX_ROUTING_RECIPE_ARMS = 64
_MAX_RUNTIME_INPUT_ID_LENGTH = 128
_SIGNAL_PART_COUNT = 2
_KB_METRIC_PART_COUNT = 3
_ROUTING_RECIPE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPPORTED_SIGNAL_TYPES = frozenset(
    {
        "authz",
        "classifier",
        "complexity",
        "context",
        "conversation",
        "domain",
        "embedding",
        "event",
        "fact_check",
        "jailbreak",
        "kb",
        "keyword",
        "language",
        "metadata",
        "modality",
        "pii",
        "preference",
        "reask",
        "structure",
        "user_feedback",
    }
)

StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]


class _RoutingRecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _valid_runtime_input_id(value: str, *, projection: bool) -> bool:
    if not value or value.strip() != value or len(value) > _MAX_RUNTIME_INPUT_ID_LENGTH:
        return False
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(
        _ROUTING_RECIPE_ID.fullmatch(part) is None for part in parts
    ):
        return False
    signal_type = parts[0]
    if signal_type != signal_type.lower():
        return False
    if projection:
        return signal_type == "projection" and len(parts) == _SIGNAL_PART_COUNT
    if signal_type == "projection":
        return False
    if signal_type == "kb_metric":
        return len(parts) == _KB_METRIC_PART_COUNT
    if signal_type not in _SUPPORTED_SIGNAL_TYPES:
        return False
    return len(parts) == _SIGNAL_PART_COUNT or signal_type == "classifier"


class RoutingRecipeInputSpec(_RoutingRecipeModel):
    id: str
    value_kind: Literal["numeric", "none"]

    @field_validator("id")
    @classmethod
    def validate_runtime_id(cls, value: str) -> str:
        if not _valid_runtime_input_id(value, projection=False):
            raise ValueError("routing recipe input specification is invalid")
        return value


class RoutingRecipeProjectionSpec(_RoutingRecipeModel):
    id: str
    value_kind: Literal["numeric", "probability"]
    outcome_binding: Literal["selected_pool_quality", "selected_is_oracle"]

    @field_validator("id")
    @classmethod
    def validate_runtime_id(cls, value: str) -> str:
        if not _valid_runtime_input_id(value, projection=True):
            raise ValueError("routing recipe projection specification is invalid")
        return value


class RoutingRecipePlan(_RoutingRecipeModel):
    """Every server-frozen input that may be observed at routing time."""

    contract_version: Literal[ROUTING_RECIPE_PLAN_CONTRACT_VERSION]
    plan_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    target_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    arm_ids: tuple[str, ...] = Field(max_length=_MAX_ROUTING_RECIPE_ARMS)
    fallback_arm_id: str | None = None
    signals: tuple[RoutingRecipeInputSpec, ...] = Field(
        max_length=_MAX_ROUTING_RECIPE_ITEMS
    )
    projections: tuple[RoutingRecipeProjectionSpec, ...] = Field(
        max_length=_MAX_ROUTING_RECIPE_ITEMS
    )
    top_k: tuple[StrictPositiveInt, ...] = Field(max_length=_MAX_ROUTING_RECIPE_ARMS)

    @field_validator("arm_ids")
    @classmethod
    def validate_arm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_ROUTING_RECIPE_ID.fullmatch(arm_id) is None for arm_id in value):
            raise ValueError("routing recipe arm id is invalid")
        if len(value) != len(set(value)):
            raise ValueError("routing recipe arms must be unique")
        return value

    @field_validator("signals")
    @classmethod
    def validate_signal_ids(
        cls, value: tuple[RoutingRecipeInputSpec, ...]
    ) -> tuple[RoutingRecipeInputSpec, ...]:
        if len({spec.id for spec in value}) != len(value):
            raise ValueError("routing recipe input specifications must be unique")
        return value

    @field_validator("projections")
    @classmethod
    def validate_projection_ids(
        cls, value: tuple[RoutingRecipeProjectionSpec, ...]
    ) -> tuple[RoutingRecipeProjectionSpec, ...]:
        if len({spec.id for spec in value}) != len(value):
            raise ValueError("routing recipe projection specifications must be unique")
        return value

    @model_validator(mode="after")
    def validate_frozen_plan(self) -> Self:
        if (not self.arm_ids and self.top_k) or (self.arm_ids and not self.top_k):
            raise ValueError("routing recipe plan is invalid")
        if (
            self.fallback_arm_id is not None
            and self.fallback_arm_id not in self.arm_ids
        ):
            raise ValueError("routing recipe fallback arm is outside the frozen pool")
        previous = 0
        for value in self.top_k:
            if value <= previous or value > len(self.arm_ids):
                raise ValueError(
                    "routing recipe top-k values must be strictly increasing frozen arm counts"
                )
            previous = value
        if self.plan_digest != routing_recipe_plan_digest(self):
            raise ValueError(
                "routing recipe plan digest does not bind its canonical body"
            )
        return self


def routing_recipe_top_k(arm_count: int) -> tuple[int, ...]:
    if arm_count <= 0:
        return ()
    return tuple(sorted({1, min(3, arm_count), min(5, arm_count)}))
