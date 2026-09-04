"""Shared strict primitives for evaluation manifests and runtime contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_validation import (
    validate_inline_image_url,
    validate_secret_env,
)


class StrictModel(BaseModel):
    """Base contract that rejects silent schema drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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
        return validate_secret_env(value)


class TextPart(StrictModel):
    type: Literal["text"] = "text"
    text: str


class ImageURL(StrictModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None

    @field_validator("url")
    @classmethod
    def validate_inline_image(cls, value: str) -> str:
        return validate_inline_image_url(value)


class ImagePart(StrictModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[ContentPart, ...]
    name: str | None = None
    tool_call_id: str | None = None
