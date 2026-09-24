"""Load Decision image defaults packaged with the matching CLI version.

Source checkouts have no image defaults. A release can include this resource
only after its backend images have been published and validated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

from cli import __version__

from .image_reference import (
    ImmutableImageReferenceError,
    validate_immutable_image_reference,
)

IMAGE_LOCK_RESOURCE = "decision-images.lock.json"
_BACKENDS = frozenset({"rocm", "cuda", "cpu"})
_SOURCE_SHA = re.compile(r"[0-9a-f]{40}\Z")
_MAX_LOCK_BYTES = 16 * 1024


class DecisionImageLockError(ValueError):
    """The packaged Decision image lock is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DecisionImageLock:
    """One package version, source commit, and backend image inventory."""

    package_version: str
    source_sha: str
    images: Mapping[str, str]


def load_decision_image_lock() -> DecisionImageLock:
    """Read the release-injected package resource, failing closed if absent."""

    try:
        payload = (
            resources.files("cli.decision_runtime")
            .joinpath(IMAGE_LOCK_RESOURCE)
            .read_bytes()
        )
    except FileNotFoundError as error:
        raise DecisionImageLockError(
            "This CLI build has no default Decision image. Install a Decision-enabled "
            "release, or use --image for local development."
        ) from error
    except OSError as error:
        raise DecisionImageLockError("Decision image lock cannot be read") from error
    return parse_decision_image_lock(payload)


def parse_decision_image_lock(payload: bytes) -> DecisionImageLock:
    """Validate the small, exact release contract before any image is used."""

    if not isinstance(payload, bytes) or not 0 < len(payload) <= _MAX_LOCK_BYTES:
        raise DecisionImageLockError("Decision image lock has an invalid size")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except DecisionImageLockError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionImageLockError("Decision image lock is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "package_version",
        "source_sha",
        "images",
    }:
        raise DecisionImageLockError("Decision image lock fields are invalid")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise DecisionImageLockError("Decision image lock schema is unsupported")
    package_version = document["package_version"]
    if not isinstance(package_version, str) or package_version != __version__:
        raise DecisionImageLockError(
            "Decision image lock does not match the installed CLI version"
        )
    source_sha = document["source_sha"]
    if not isinstance(source_sha, str) or _SOURCE_SHA.fullmatch(source_sha) is None:
        raise DecisionImageLockError("Decision image lock source SHA is invalid")
    images = document["images"]
    if not isinstance(images, dict) or not images or not set(images) <= _BACKENDS:
        raise DecisionImageLockError("Decision image lock backends are invalid")

    checked: dict[str, str] = {}
    for backend, reference in images.items():
        try:
            image = validate_immutable_image_reference(reference)
        except ImmutableImageReferenceError as error:
            raise DecisionImageLockError(
                f"Decision image lock has an invalid {backend} image: {error}"
            ) from error
        repository = image.rsplit("@", 1)[0].rsplit("/", 1)[-1].split(":", 1)[0]
        if repository != f"decision-runtime-{backend}":
            raise DecisionImageLockError(
                f"Decision image lock maps {backend} to a different backend image"
            )
        checked[backend] = image
    return DecisionImageLock(package_version, source_sha, MappingProxyType(checked))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionImageLockError("Decision image lock has duplicate fields")
        result[key] = value
    return result
