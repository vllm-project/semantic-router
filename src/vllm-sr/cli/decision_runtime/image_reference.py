"""Conservative validation for immutable Decision runtime OCI references."""

from __future__ import annotations

import re

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_NAME_COMPONENT = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_REGISTRY_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_ASCII_CONTROL_THRESHOLD = 32
_MAX_IMAGE_NAME_LENGTH = 255
_MAX_PORT = 65535
_MAX_DNS_HOST_LENGTH = 253


class ImmutableImageReferenceError(ValueError):
    """An image reference is unsafe, mutable, or not conservatively valid."""


def is_local_docker_image_id(reference: object) -> bool:
    """Recognize one complete, lowercase Docker image ID without abbreviations."""

    return isinstance(reference, str) and _DIGEST.fullmatch(reference) is not None


def validate_decision_image_reference(reference: object) -> str:
    """Accept a published digest reference or an exact local Docker image ID."""

    if isinstance(reference, str) and is_local_docker_image_id(reference):
        return reference
    return validate_immutable_image_reference(reference)


def validate_immutable_image_reference(reference: object) -> str:
    """Return a digest-qualified OCI reference after strict validation.

    The accepted subset intentionally covers the registry/repository forms used
    by Decision runtime images while rejecting shell-option-shaped values,
    schemes, whitespace, ambiguous separators, and mutable tag-only references.
    """

    if (
        not isinstance(reference, str)
        or not reference
        or reference != reference.strip()
        or reference.startswith("-")
        or any(
            character.isspace() or ord(character) < _ASCII_CONTROL_THRESHOLD
            for character in reference
        )
        or "://" in reference
        or reference.count("@") != 1
    ):
        raise ImmutableImageReferenceError(
            "image must be a safe digest-qualified OCI reference"
        )

    name_and_tag, digest = reference.rsplit("@", 1)
    if not _DIGEST.fullmatch(digest):
        raise ImmutableImageReferenceError(
            "image digest must be sha256 followed by 64 lowercase hex characters"
        )
    if (
        len(name_and_tag) > _MAX_IMAGE_NAME_LENGTH
        or name_and_tag.startswith("/")
        or name_and_tag.endswith("/")
    ):
        raise ImmutableImageReferenceError("image repository is invalid")

    parts = name_and_tag.split("/")
    if any(not part for part in parts):
        raise ImmutableImageReferenceError("image repository is invalid")

    first_is_registry = len(parts) > 1 and (
        parts[0] == "localhost" or "." in parts[0] or ":" in parts[0]
    )
    if first_is_registry:
        registry = parts.pop(0)
        _validate_registry(registry)

    last = parts[-1]
    if ":" in last:
        repository, tag = last.rsplit(":", 1)
        if not repository or not _TAG.fullmatch(tag):
            raise ImmutableImageReferenceError("image tag is invalid")
        parts[-1] = repository

    if not parts or any(not _NAME_COMPONENT.fullmatch(part) for part in parts):
        raise ImmutableImageReferenceError("image repository is invalid")
    return reference


def immutable_image_digest(reference: object) -> str:
    """Return the validated immutable digest portion of an OCI reference."""

    return validate_immutable_image_reference(reference).rsplit("@", 1)[1]


def _validate_registry(registry: str) -> None:
    host = registry
    if ":" in registry:
        if registry.count(":") != 1:
            raise ImmutableImageReferenceError("image registry is invalid")
        host, port_text = registry.rsplit(":", 1)
        if not port_text.isascii() or not port_text.isdigit():
            raise ImmutableImageReferenceError("image registry port is invalid")
        port = int(port_text)
        if not 1 <= port <= _MAX_PORT:
            raise ImmutableImageReferenceError("image registry port is invalid")

    if host == "localhost":
        return
    if len(host) > _MAX_DNS_HOST_LENGTH or any(
        not _REGISTRY_LABEL.fullmatch(label) for label in host.split(".")
    ):
        raise ImmutableImageReferenceError("image registry is invalid")
