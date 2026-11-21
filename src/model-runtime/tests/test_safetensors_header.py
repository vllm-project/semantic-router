"""Malformed safetensors headers are permanent package-format failures."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.registry.artifacts import (
    HEADER_LENGTH_BYTES,
    MAX_SAFETENSORS_HEADER,
    safetensors_header,
)


def write_header(tmp_path: Path, payload: bytes, *, name: str = "model.safetensors") -> Path:
    """A safetensors file whose 8-byte length prefix matches ``payload``."""
    path = tmp_path / name
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


def test_valid_header_parses(tmp_path: Path) -> None:
    payload = b'{"__metadata__": {"format": "pt"}}'
    header = safetensors_header(write_header(tmp_path, payload))
    assert header == {"__metadata__": {"format": "pt"}}


def test_invalid_json_payload_is_a_package_error(tmp_path: Path) -> None:
    path = write_header(tmp_path, b'{"__metadata__": ')

    with pytest.raises(PackageError) as caught:
        safetensors_header(path)

    message = str(caught.value)
    assert "model.safetensors" in message
    assert "invalid safetensors header" in message
    assert isinstance(caught.value.__cause__, ValueError)


def test_invalid_utf8_payload_is_a_package_error(tmp_path: Path) -> None:
    path = write_header(tmp_path, b"\xff\xfe\x00\x01")

    with pytest.raises(PackageError) as caught:
        safetensors_header(path)

    message = str(caught.value)
    assert "model.safetensors" in message
    assert "invalid safetensors header" in message
    assert isinstance(caught.value.__cause__, ValueError)


def test_truncated_length_prefix_is_still_a_package_error(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"\x10\x00")

    with pytest.raises(PackageError, match="truncated safetensors header: model.safetensors"):
        safetensors_header(path)


def test_truncated_payload_is_still_a_package_error(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 64) + b'{"__metadata__": {}}')

    with pytest.raises(PackageError, match="truncated safetensors header: model.safetensors"):
        safetensors_header(path)


@pytest.mark.parametrize("size", [0, 1, MAX_SAFETENSORS_HEADER + 1])
def test_invalid_header_size_is_still_a_package_error(tmp_path: Path, size: int) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", size) + b"{}")

    with pytest.raises(PackageError, match="invalid safetensors header: model.safetensors"):
        safetensors_header(path)


@pytest.mark.parametrize("payload", [b"[]", b"null", b"12"])
def test_non_object_header_keeps_its_existing_error(tmp_path: Path, payload: bytes) -> None:
    with pytest.raises(PackageError, match="invalid safetensors header: model.safetensors"):
        safetensors_header(write_header(tmp_path, payload))


def test_error_message_names_the_file_and_reason(tmp_path: Path) -> None:
    path = write_header(tmp_path, b"not json at all", name="layers.safetensors")

    with pytest.raises(PackageError) as caught:
        safetensors_header(path)

    message = str(caught.value)
    assert message.startswith("invalid safetensors header: layers.safetensors: ")
    assert message.count(":") >= 2
    assert len(message) > len("invalid safetensors header: layers.safetensors: ")


def test_header_length_constant_matches_the_prefix(tmp_path: Path) -> None:
    assert HEADER_LENGTH_BYTES == 8
    assert write_header(tmp_path, b"{}").stat().st_size == HEADER_LENGTH_BYTES + 2
