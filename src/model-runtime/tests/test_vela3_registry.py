"""The Vela 3.0 built-in table: both members' pins, identity formula and the vela3v detection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.vela3.family import Vela3Family
from vllm_srun.families.vela3.package import (
    VELA3V_CONFIG_FILE,
    VELA3V_ENCODER,
    member_of,
    member_of_vision,
)
from vllm_srun.plugins.base import PackageRef
from vllm_srun.registry import builtin
from vllm_srun.registry.artifacts import sha256_json
from vllm_srun.testing.vela3 import write_encoder_package

MODELS = builtin.all_models("vela3")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
VISION = builtin.lookup("vela3-fast-vision")
# The weight files the image package's identity is the digest set of: the raw
# backbone, the two task-named backbones and the four task adapters.
VISION_WEIGHTS = (
    "model.safetensors",
    "backbone_h2.safetensors",
    "backbone_v2.safetensors",
    "span_adapter.safetensors",
    "decision_adapter.safetensors",
    "box_adapter.safetensors",
    "ocr_adapter.safetensors",
)


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return write_encoder_package(tmp_path_factory.mktemp("vela3") / "fast")


def test_the_table_pins_both_members() -> None:
    assert [model.repo_id for model in MODELS] == [
        "vllm-sr/vela3-fast",
        "vllm-sr/vela3-fast-vision",
    ]
    assert [model.backbone for model in MODELS] == ["ultrabert", "ultravbert"]
    for model in MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision), model.repo_id
        assert model.access == "public"
        assert builtin.lookup(model.repo_id.split("/", 1)[1]) is model
        assert builtin.by_identity(model.model_sha256) is model


def test_the_text_entry_keeps_its_golden_answers_and_the_vision_has_none() -> None:
    fast, vision = MODELS
    assert fast.golden_answers, "vela3-fast must keep its recorded references"
    assert vision.golden_answers == {}


def test_the_vision_identity_is_the_weight_digest_set() -> None:
    assert set(VISION.files) == {
        "config.json",
        VELA3V_CONFIG_FILE,
        "router_questions.json",
        "tokenizer.json",
        *VISION_WEIGHTS,
    }
    for name, digest in VISION.files.items():
        assert DIGEST.fullmatch(digest), name
    weights = {name: VISION.files[name] for name in VISION_WEIGHTS}
    assert sha256_json(weights) == VISION.model_sha256
    assert sorted(weights) == sorted(set(VISION_WEIGHTS))


def test_the_vision_pins_match_the_published_sha256sums() -> None:
    """The pins re-derived from the release's own SHA256SUMS, transcribed at review time."""
    sums = """65e0c0284eb8c6e494c058d1988ad1b3d2eab60c368bb3d704e09174d1220bd5  backbone_h2.safetensors
46e65f6e6b906a64e4ad57e2e76957522bd01e2eeb5d8467078b895410d1e0aa  backbone_v2.safetensors
62d05d58ddbb7f9248dff4e84dc4864c09c78a56e19088cae112ccbd9035fb6b  box_adapter.safetensors
96c0a13f682092cb4b7289ec10d54179dd0085bbc988cba269a7b3d1ace1c079  config.json
8a8ddc13ccc09ebe41f4200b8685a72a7c19d4567c7c1a85f600b25dad53ae55  decision_adapter.safetensors
9387cfd5bc53082b6ff634453c6b0cb3123a67e4c197209e308ecfaf7a578900  model.safetensors
882f2d71d6aee9d90f75b68d35c6eb74388fa96c1013d959b0ab18fe8b0f9719  ocr_adapter.safetensors
f7bf64a8c5265a843372147aef70a8babaa584ea9b9e2e7751fa1138d656b10f  router_questions.json
9cbfd04d5960818001c1b6ff7a3f02a5595dd73f3c5d32a256fbe1855f4055bd  span_adapter.safetensors
bda9af747539d9444eca3fc9df922811ae318d93ced0b83ecf2001c77a925e8a  tokenizer.json
f8d159468b49ee268c29d29c99ec488226ef7524fd967caa34d8d71f333f71f2  vela3v_config.json
"""
    published = {}
    for line in sums.splitlines():
        digest, _, name = line.strip().partition(" ")
        published[name.strip()] = digest
    assert published == VISION.files


def test_the_vision_package_is_detected_and_refused_at_serving(tmp_path) -> None:
    root = tmp_path / "vision"
    root.mkdir()
    (root / VELA3V_CONFIG_FILE).write_text(
        json.dumps({"name": "vela3-fast-vision", "format": VELA3V_ENCODER}),
        encoding="utf-8",
    )
    assert member_of_vision(root) == VELA3V_ENCODER
    family = Vela3Family()
    assert family.detect(PackageRef(root))
    with pytest.raises(PackageError, match="image serving path is not implemented"):
        family.verify(PackageRef(root))


def test_a_text_package_is_never_detected_as_vision(package, tmp_path) -> None:
    assert member_of(package) == "vela3-encoder/1"
    assert member_of_vision(package) is None
    family = Vela3Family()
    assert family.detect(PackageRef(package))
    verified = family.verify(PackageRef(package))
    assert verified.family == "vela3"


def test_vision_detection_rejects_garbage(tmp_path) -> None:
    root = tmp_path / "other"
    root.mkdir()
    (root / VELA3V_CONFIG_FILE).write_text(
        json.dumps({"format": "vela3v/9"}), encoding="utf-8"
    )
    assert member_of_vision(root) is None
    assert not Vela3Family().detect(PackageRef(root))
