"""Fail-closed adapter from the canonical catalog to Decision runtime profiles."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from cli.model_catalog import resolve_catalog_provider_model
from cli.model_catalog_types import CatalogProviderModel, ModelCatalogError

from .runtime_profile import RuntimeProfile, RuntimeProfileError, load_runtime_profile

DECISION_PROVIDER_ID = "decision-runtime"
SYSTEMONE_PROTOCOL = "typesafe/systemone@1"
_CATALOG_TO_RUNTIME_FAMILY = {
    "decision-encoder": "vela",
    "decision-qwen3.5": "qwen3.5",
}


class RuntimeModelResolutionError(ValueError):
    """The catalog cannot bind a request to one exact supported runtime profile."""


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeModel:
    """Catalog-owned model identity paired with its revision-owned profile."""

    catalog: CatalogProviderModel
    repository_id: str
    profile: RuntimeProfile


def resolve_decision_runtime_model(
    model_id: str,
    *,
    catalog_version: str = "latest",
    backend: str | None = None,
    target: str | None = None,
) -> ResolvedRuntimeModel:
    """Resolve one exact canonical ID without aliases or mutable revisions."""

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
    try:
        profile = load_runtime_profile(catalog.revision)
    except RuntimeProfileError as error:
        raise RuntimeModelResolutionError(str(error)) from error
    if profile.revision != catalog.revision or profile.family != expected_family:
        raise RuntimeModelResolutionError(
            "runtime profile does not match the catalog revision and family"
        )
    if backend is not None:
        try:
            profile.require_backend(backend, target=target)
        except RuntimeProfileError as error:
            raise RuntimeModelResolutionError(str(error)) from error
    elif target is not None:
        raise RuntimeModelResolutionError("a target requires an explicit backend")

    return ResolvedRuntimeModel(
        catalog=catalog,
        repository_id=repository_id,
        profile=profile,
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
