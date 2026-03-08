"""Typed backend model compatibility blocks for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict, RootModel


class ImageGenBackendEntryCompatConfig(BaseModel):
    """Typed schema for image_gen_backends.*."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    model: str | None = None
    default_width: int | None = None
    default_height: int | None = None
    timeout_seconds: int | None = None
    base_url: str | None = None
    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    api_key: str | None = None
    quality: str | None = None
    style: str | None = None


class ImageGenBackendsCompatConfig(
    RootModel[dict[str, ImageGenBackendEntryCompatConfig]]
):
    """Typed schema for image_gen_backends."""


class ProviderProfileCompatConfig(BaseModel):
    """Typed schema for provider_profiles.*."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    base_url: str | None = None
    auth_header: str | None = None
    auth_prefix: str | None = None
    extra_headers: dict[str, str] | None = None
    api_version: str | None = None
    chat_path: str | None = None


class ProviderProfilesCompatConfig(RootModel[dict[str, ProviderProfileCompatConfig]]):
    """Typed schema for provider_profiles."""
