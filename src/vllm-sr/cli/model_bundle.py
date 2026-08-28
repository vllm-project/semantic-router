"""Shared contracts for packaged built-in model recipe bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol

MODEL_BUNDLE_FILES = (
    "README.md",
    "config.yaml",
    "metadata.yaml",
    "probes.yaml",
    "recipe.dsl",
)

OPTIONAL_MOM_BUNDLE_FILES = (
    "mom-evaluation.yaml",
    "evaluation-scorecard.md",
)


def model_bundle_optional_files(bundle: BundleResource) -> tuple[str, ...]:
    """Return optional MoM evaluation files present in a bundle directory."""

    return tuple(
        name
        for name in OPTIONAL_MOM_BUNDLE_FILES
        if bundle.joinpath(name).is_file()
    )


def _validate_model_bundle_directory(bundle: BundleResource) -> None:
    if not bundle.is_dir():
        raise ValueError("built-in model bundle is not installed")
    actual = sorted(item.name for item in bundle.iterdir() if item.is_file())
    required = sorted(MODEL_BUNDLE_FILES)
    allowed = set(required) | set(OPTIONAL_MOM_BUNDLE_FILES)
    missing = sorted(set(required) - set(actual))
    extra = sorted(set(actual) - allowed)
    if missing or extra:
        raise ValueError(
            "built-in model bundle files differ from the five-file contract; "
            f"missing={missing}, extra={extra}"
        )


class BundleResource(Protocol):
    """Small ``Traversable`` subset used by paths and package resources."""

    name: str

    def is_dir(self) -> bool: ...

    def is_file(self) -> bool: ...

    def iterdir(self): ...

    def joinpath(self, *descendants: str): ...

    def read_bytes(self) -> bytes: ...


def model_bundle_digest(bundle: BundleResource) -> str:
    """Return a path-bound digest for one built-in Recipe bundle."""

    _validate_model_bundle_directory(bundle)

    contents: dict[str, bytes] = {}
    for name in MODEL_BUNDLE_FILES:
        resource = bundle.joinpath(name)
        if not resource.is_file():
            raise ValueError(f"built-in model bundle resource is invalid: {name}")
        contents[name] = resource.read_bytes()
    return model_bundle_digest_from_files(contents)


def model_bundle_digest_from_files(files: Mapping[str, bytes]) -> str:
    """Digest an already-validated exact-five bundle file mapping."""

    actual = sorted(files)
    expected = sorted(MODEL_BUNDLE_FILES)
    if actual != expected:
        raise ValueError(
            "built-in model bundle files differ from the five-file contract; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    digest = hashlib.sha256()
    for name in MODEL_BUNDLE_FILES:
        content = files[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()
