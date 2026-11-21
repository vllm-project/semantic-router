"""Verification of ``vela3-pico/1`` packages: the format, the loaded files, their digests and the markers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.vela3_pico.package import (
    FORMAT,
    LOADED_FILES,
    member_of,
    verify,
)
from vllm_srun.registry.artifacts import named_files, sha256_json
from vllm_srun.testing.vela3 import write_encoder_package
from vllm_srun.testing.vela3_pico import write_pico_package


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return write_pico_package(tmp_path_factory.mktemp("pico") / "pico")


def test_only_the_pico_format_is_a_pico(package, tmp_path) -> None:
    assert member_of(package) == FORMAT
    assert member_of(write_encoder_package(tmp_path / "fast")) is None
    other = tmp_path / "d3"
    other.mkdir()
    (other / "vela3_config.json").write_text(json.dumps({"format": "vela3/3"}))
    assert member_of(other) is None
    (other / "vela3_config.json").write_text("[]")
    assert member_of(other) is None


def test_the_identity_is_the_weight_files_digest(package) -> None:
    verified = verify(package, None)
    digests = named_files(package, list(LOADED_FILES))
    assert verified.files == digests
    assert verified.model_sha256 == sha256_json(
        {"model.safetensors": digests["model.safetensors"]}
    )
    assert verified.max_input_tokens == 1024


def test_an_unpinned_package_must_match_its_sha256sums(package, tmp_path) -> None:
    copy = Path(shutil.copytree(package, tmp_path / "tampered"))
    (copy / "router_questions.json").write_text('{"questions": {}}')
    with pytest.raises(PackageError, match="SHA256SUMS"):
        verify(copy, None)
    (copy / "SHA256SUMS").unlink()
    with pytest.raises(PackageError, match="has no SHA256SUMS"):
        verify(copy, None)


def test_a_pinned_package_needs_no_sha256sums_but_its_digests(
    package, tmp_path
) -> None:
    copy = Path(shutil.copytree(package, tmp_path / "pinned"))
    (copy / "SHA256SUMS").unlink()
    digests = named_files(copy, list(LOADED_FILES))
    assert verify(copy, digests).model_sha256
    (copy / "vela3_config.json").write_text(
        (copy / "vela3_config.json").read_text().replace("0.55", "0.56")
    )
    with pytest.raises(PackageError, match="pinned revision"):
        verify(copy, digests)


def test_the_markers_must_index_the_backbone_vocabulary(package, tmp_path) -> None:
    copy = Path(shutil.copytree(package, tmp_path / "markers"))
    config = json.loads((copy / "config.json").read_text())
    config["decision"]["marker_ids"]["[Q]"] = config["vocab_size"]
    (copy / "config.json").write_text(json.dumps(config))
    (copy / "SHA256SUMS").unlink()
    with pytest.raises(PackageError, match="below vocab_size"):
        verify(copy, named_files(copy, list(LOADED_FILES)))
