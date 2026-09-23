"""Fail-closed adapter from the canonical catalog to Decision runtime profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from urllib.parse import unquote, urlsplit

from cli.model_catalog import resolve_catalog_provider_model
from cli.model_catalog_types import CatalogProviderModel, ModelCatalogError

from .backend_capabilities import require_runtime_backend
from .runtime_profile import (
    RuntimeProfile,
    RuntimeProfileError,
    load_runtime_profile,
    validate_catalog_revision,
)

DECISION_PROVIDER_ID = "decision-runtime"
SYSTEMONE_PROTOCOL = "typesafe/systemone@1"
_CATALOG_TO_RUNTIME_FAMILY = {
    "decision-encoder": "vela",
    "decision-qwen3.5": "qwen3.5",
}

# These are implementation templates, not model-file allowlists. A newer
# immutable Hub revision keeps the model's rendering and batching policy while
# the artifact resolver obtains that revision's own manifest and file hashes.
# Updating a model card's revision does not require a new runtime build.
_MODEL_PROFILE_TEMPLATES = {
    "Decision-1.0-Kai-0.6B": "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
    "Decision-1.0-Lex-0.6B": "ee8e74d912fca8328a353c11d174b44da3f91781",
    "Decision-1.0-Eos-0.8B": "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
    "Decision-1.0-Sol-2B": "0665a41108e8f0b33a9515c98311c45947b99399",
    "Decision-1.0-Nox-4B": "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4",
    "Decision-1.0-Lux-9B": "bd45a30aee8c84032791c245c70f86dee5389cc8",
}


class RuntimeModelResolutionError(ValueError):
    """The catalog cannot bind a request to one exact supported runtime profile."""


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeModel:
    """Catalog model identity and a compatible runtime-owned family template."""

    catalog: CatalogProviderModel
    repository_id: str
    profile: RuntimeProfile
    template_revision: str


def resolve_decision_runtime_model(
    model_id: str,
    *,
    revision: str | None = None,
    catalog_version: str = "latest",
    backend: str | None = None,
    target: str | None = None,
) -> ResolvedRuntimeModel:
    """Resolve one exact model ID and immutable Hub revision."""

    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise RuntimeModelResolutionError("an exact canonical model ID is required")
    try:
        catalog = resolve_catalog_provider_model(
            model_id,
            provider_id=DECISION_PROVIDER_ID,
            catalog_version=catalog_version,
        )
    except ModelCatalogError as error:
        raise RuntimeModelResolutionError(str(error)) from error

    # The catalog resolver also supports internal, lower-case physical card IDs.
    # The public Decision runtime intentionally does not: its request identity is
    # the provider's exact Hugging Face repository ID.
    if model_id != catalog.model_id:
        raise RuntimeModelResolutionError(
            f"model ID must be the exact canonical identity {catalog.model_id!r}"
        )
    if catalog.provider_id != DECISION_PROVIDER_ID:
        raise RuntimeModelResolutionError("catalog provider identity drifted")
    if catalog.protocols != (SYSTEMONE_PROTOCOL,):
        raise RuntimeModelResolutionError("catalog SystemOne protocol binding drifted")

    repository_id = _hugging_face_repository_id(catalog.distribution_source)
    if repository_id != catalog.model_id:
        raise RuntimeModelResolutionError(
            "catalog distribution source does not match the canonical model ID"
        )

    expected_family = _CATALOG_TO_RUNTIME_FAMILY.get(catalog.family)
    if expected_family is None:
        raise RuntimeModelResolutionError(
            f"catalog family {catalog.family!r} has no Decision runtime implementation"
        )
    template_revision = _MODEL_PROFILE_TEMPLATES.get(
        catalog.model_id.rsplit("/", 1)[-1]
    )
    if template_revision is None:
        raise RuntimeModelResolutionError(
            "this Decision model has no installed family template"
        )
    try:
        selected_revision = validate_catalog_revision(
            catalog.revision if revision is None else revision
        )
        profile = load_runtime_profile(template_revision)
    except RuntimeProfileError as error:
        raise RuntimeModelResolutionError(str(error)) from error
    if profile.family != expected_family:
        raise RuntimeModelResolutionError(
            "runtime profile does not match the catalog model family"
        )
    catalog = replace(catalog, revision=selected_revision)
    profile = replace(profile, revision=selected_revision)
    if backend is not None:
        try:
            require_runtime_backend(catalog, profile, backend, target=target)
        except RuntimeProfileError as error:
            raise RuntimeModelResolutionError(str(error)) from error
    elif target is not None:
        raise RuntimeModelResolutionError("a target requires an explicit backend")

    return ResolvedRuntimeModel(
        catalog=catalog,
        repository_id=repository_id,
        profile=profile,
        template_revision=template_revision,
    )


def _hugging_face_repository_id(source: str) -> str:
    parsed = urlsplit(source)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeModelResolutionError(
            "catalog distribution source must be a canonical Hugging Face URL"
        )
    path = unquote(parsed.path)
    if path != parsed.path or not path.startswith("/") or path.endswith("/"):
        raise RuntimeModelResolutionError(
            "catalog distribution source must use an unescaped repository path"
        )
    repository_id = path[1:]
    parts = repository_id.split("/")
    if (
        len(parts) != 2
        or any(not part or part in {".", ".."} for part in parts)
        or any("\\" in part or ":" in part for part in parts)
    ):
        raise RuntimeModelResolutionError(
            "catalog distribution source does not identify one repository"
        )
    return repository_id
