"""The Vela 2.0 built-in table: every size's pin, loaded-file digests, identity, golden answers and kernel choices."""

from __future__ import annotations

import itertools
import json
import re

import pytest
from vllm_srun.families.vela2.family import GOLDEN_QUESTIONS
from vllm_srun.families.vela2.package import (
    BROAD_HEAD_FILE,
    COMMON_FILES,
    MANIFEST_FILE,
)
from vllm_srun.plugins.decisions import compare_answers, well_formed
from vllm_srun.registry import builtin
from vllm_srun.registry.artifacts import sha256_json
from vllm_srun.registry.tables.common import REGISTRY_DIR
from vllm_srun.supervision.readiness import GPU_TOLERANCE

MODELS = builtin.all_models("vela2")
DECODERS = [model for model in MODELS if model.backbone == "qwen3_5_text"]
SIZES = ("0.3B", "0.8B", "4B", "9B")
# The Decision 2.0 model whose backbone each decoder continues: the same FLA tuning keys.
BASES = {"0.8B": "Eos-0.8B", "4B": "Nox-4B", "9B": "Lux-9B"}
GOLDEN_IDS = {
    "domain",
    "jailbreak",
    "urgency",
    "topics.medication",
    "topics.billing",
    "pii",
    "halu",
}
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
SHARD = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors\Z")


def size(model: builtin.BuiltinModel) -> str:
    return model.repo_id.rsplit("-", 1)[1]


def weights(model: builtin.BuiltinModel) -> list[str]:
    if model.backbone == "modernbert":
        return ["model.safetensors"]
    return sorted(name for name in model.files if SHARD.fullmatch(name))


def test_the_table_pins_every_size() -> None:
    assert [model.repo_id for model in MODELS] == [
        f"vllm-sr/Vela-2.0-{name}" for name in SIZES
    ]
    assert [model.backbone for model in MODELS] == ["modernbert"] + ["qwen3_5_text"] * 3
    for model in MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision), model.repo_id
        assert model.access == "public"
        assert builtin.lookup(model.repo_id.split("/", 1)[1]) is model
        assert builtin.by_identity(model.model_sha256) is model
    for smaller, larger in itertools.pairwise(MODELS):
        assert smaller.loaded_parameters < larger.loaded_parameters
        assert smaller.min_device_memory_gib <= larger.min_device_memory_gib


@pytest.mark.parametrize("model", MODELS, ids=size)
def test_every_loaded_file_is_pinned_and_the_identity_is_the_weights(model) -> None:
    names = set(model.files)
    required = set(COMMON_FILES) | set(weights(model))
    optional: set[str] = set()
    if model.backbone == "qwen3_5_text":
        required.add("model.safetensors.index.json")
        optional = {BROAD_HEAD_FILE, MANIFEST_FILE}
        shards = [SHARD.fullmatch(name).groups() for name in weights(model)]
        count = int(shards[0][1])
        assert sorted(int(index) for index, _ in shards) == list(range(1, count + 1))
        assert {int(total) for _, total in shards} == {count}
    assert required <= names <= required | optional
    assert all(DIGEST.fullmatch(digest) for digest in model.files.values())
    assert model.manifest_sha256 == model.files.get(MANIFEST_FILE, "")
    identity = weights(model) + ([BROAD_HEAD_FILE] if BROAD_HEAD_FILE in names else [])
    assert model.model_sha256 == sha256_json(
        {name: model.files[name] for name in identity}
    )


@pytest.mark.parametrize(
    "file_name", ["golden_answers_vela2.json", "kernel_choices.json"]
)
def test_recorded_references_are_for_the_pinned_revisions(file_name) -> None:
    recorded = json.loads((REGISTRY_DIR / file_name).read_text(encoding="utf-8"))
    for model in MODELS:
        if model.repo_id in recorded:
            assert recorded[model.repo_id]["revision"] == model.revision, model.repo_id


@pytest.mark.parametrize("model", MODELS, ids=size)
def test_every_size_has_well_formed_reference_answers(model) -> None:
    assert set(model.golden_answers) - {"npu"} == {"cpu", "rocm"}
    levels = GOLDEN_QUESTIONS["urgency"]["criteria"]
    for answers in model.golden_answers.values():
        assert set(answers) == GOLDEN_IDS
        assert all(well_formed(answer) for answer in answers.values())
        domain = answers["domain"]
        assert set(domain["probabilities"]) == set(
            GOLDEN_QUESTIONS["domain"]["criteria"]
        )
        assert domain["choice"] == max(
            domain["probabilities"], key=domain["probabilities"].get
        )
        assert answers["urgency"]["legend"] == {
            str(index): level for index, level in enumerate(levels)
        }
    cpu, rocm = model.golden_answers["cpu"], model.golden_answers["rocm"]
    assert compare_answers(cpu, rocm, GPU_TOLERANCE) == (
        len(GOLDEN_IDS),
        len(GOLDEN_IDS),
    )
    assert cpu["domain"]["choice"] == rocm["domain"]["choice"]


@pytest.mark.parametrize("model", DECODERS, ids=size)
def test_decoders_pin_the_kernel_choices_of_the_backbone_they_continue(model) -> None:
    base = builtin.lookup(f"vllm-sr/Decision-2.0-{BASES[size(model)]}")
    assert set(model.kernel_choices) == {"rocm:gfx942"}
    assert model.kernel_choices == base.kernel_choices
    assert model.min_device_memory_gib == base.min_device_memory_gib


def test_the_encoder_has_no_kernel_choices() -> None:
    assert [model.kernel_choices for model in MODELS if model not in DECODERS] == [{}]
