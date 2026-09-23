"""Release image lock validation for default Decision runtime images."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.decision_runtime import image_lock
from cli.decision_runtime.image_lock import (
    DecisionImageLockError,
    load_decision_image_lock,
    parse_decision_image_lock,
)

SOURCE_SHA = "a" * 40
IMAGES = {
    backend: f"example.test/decision-runtime-{backend}@sha256:{letter * 64}"
    for backend, letter in (("cpu", "b"), ("rocm", "c"), ("cuda", "d"))
}


def lock_bytes(**changes: object) -> bytes:
    document: dict[str, object] = {
        "schema_version": 1,
        "source_sha": SOURCE_SHA,
        "images": IMAGES,
    }
    document.update(changes)
    return json.dumps(document).encode()


def test_release_image_lock_loads_all_three_immutable_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / image_lock.IMAGE_LOCK_RESOURCE).write_bytes(lock_bytes())
    monkeypatch.setattr(image_lock.resources, "files", lambda _package: tmp_path)

    lock = load_decision_image_lock()

    assert lock.source_sha == SOURCE_SHA
    assert lock.images == IMAGES
    with pytest.raises(TypeError):
        lock.images["cpu"] = IMAGES["cpu"]  # type: ignore[index]


def test_source_checkout_without_release_lock_fails_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_lock.resources, "files", lambda _package: tmp_path)

    with pytest.raises(DecisionImageLockError, match=r"not installed.*--image"):
        load_decision_image_lock()


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (lock_bytes(schema_version=2), "schema is unsupported"),
        (lock_bytes(source_sha="main"), "source SHA is invalid"),
        (lock_bytes(images={}), "backends are invalid"),
        (lock_bytes(images={"mlx": IMAGES["cpu"]}), "backends are invalid"),
        (
            lock_bytes(images={"cpu": "example.test/decision-runtime-cpu:latest"}),
            "invalid cpu image",
        ),
        (
            lock_bytes(images={"cpu": IMAGES["rocm"]}),
            "different backend image",
        ),
        (lock_bytes(extra="unreviewed"), "fields are invalid"),
        (b'{"schema_version":1,"schema_version":1}', "duplicate fields"),
        (b" " * (16 * 1024 + 1), "invalid size"),
    ),
)
def test_release_lock_rejects_unqualified_or_ambiguous_content(
    payload: bytes, message: str
) -> None:
    with pytest.raises(DecisionImageLockError, match=message):
        parse_decision_image_lock(payload)
