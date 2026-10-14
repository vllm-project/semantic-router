"""Decision 1.0 packages: the pointer's file map, pinned digests, identity, manifests, presets and the table."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.decision1 import package as pkg
from vllm_srun.families.decision1.family import Decision1Family
from vllm_srun.plugins.base import PackageRef
from vllm_srun.registry import builtin
from vllm_srun.registry.artifacts import named_files, sha256_file
from vllm_srun.testing.decision1 import write_package


@pytest.fixture(scope="module")
def qwen_package(tmp_path_factory):
    return write_package(tmp_path_factory.mktemp("d1") / "qwen", seed=1)


@pytest.fixture(scope="module")
def vela_package(tmp_path_factory):
    return write_package(
        tmp_path_factory.mktemp("d1") / "vela", runtime=pkg.VELA, seed=2, presets=True
    )


def pointer(root):
    return json.loads((root / pkg.POINTER_FILE).read_text())


def test_detect_reads_only_the_pointer(qwen_package, vela_package, tmp_path):
    family = Decision1Family()
    assert family.detect(PackageRef(qwen_package))
    assert family.detect(PackageRef(vela_package))
    (tmp_path / "config.json").write_text(
        json.dumps({**pkg.POINTER, "runtime_family": "x"})
    )
    assert not family.detect(PackageRef(tmp_path))
    (tmp_path / "config.json").write_text("not json")
    assert not family.detect(PackageRef(tmp_path))


def test_file_map_lists_what_inference_reads(qwen_package, vela_package):
    files = pkg.file_map(pointer(qwen_package))
    assert files.runtime == pkg.QWEN and files.temperature == 1.25
    assert files.model_files() == sorted(
        [
            "backbone/config.json",
            "backbone/model.safetensors",
            "decision_config.json",
            "decision_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
    )
    encoder = pkg.file_map(pointer(vela_package))
    assert set(encoder.decision_weights) == set(pkg.VELA_WEIGHTS)
    assert "native/tokenizer/special_tokens_map.json" in encoder.model_files()
    assert pkg.PRESETS_FILE in pkg.inventory(encoder, vela_package)


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": " "},
        {"backbone": {"config": "../escape.json", "weights": ["w.safetensors"]}},
        {"backbone": {"config": "c.json", "weights": []}},
        {"decision_weights": {"decision_head": "h.safetensors", "extra": "x"}},
        {"calibration": {}},
        {"calibration": {"temperature": 0}},
        {"calibration": {"temperature": 1.0, "temperature_file": "t.json"}},
    ],
)
def test_file_map_rejects_malformed_pointers(qwen_package, change):
    with pytest.raises(PackageError):
        pkg.file_map({**pointer(qwen_package), **change})


def test_local_verification_computes_the_identity(qwen_package):
    verified = pkg.verify(qwen_package, None)
    assert verified.verification == "local" and verified.manifest_sha256 == ""
    digests = named_files(qwen_package, verified.files.model_files())
    assert verified.model_sha256 == pkg.sha256_json(digests)
    assert verified.temperatures == {"choice": 1.25, "noul": 1.25, "score": 1.25}


def test_pinned_verification_checks_every_loaded_file(qwen_package, tmp_path):
    verified = pkg.verify(qwen_package, None)
    assert pkg.verify(qwen_package, verified.digests).verification == "pinned"
    partial = dict(verified.digests)
    partial.pop("tokenizer.json")
    with pytest.raises(PackageError, match="not pinned"):
        pkg.verify(qwen_package, partial)
    changed = {**verified.digests, "decision_head.safetensors": "0" * 64}
    with pytest.raises(PackageError, match="differ"):
        pkg.verify(qwen_package, changed)


def test_tampered_and_linked_files_are_refused(qwen_package, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(qwen_package, copy)
    digests = pkg.verify(copy, None).digests
    with (copy / "decision_config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(PackageError):
        pkg.verify(copy, digests)
    os.remove(copy / "tokenizer.json")
    os.symlink(qwen_package / "tokenizer.json", copy / "tokenizer.json")
    with pytest.raises(PackageError, match="link"):
        pkg.verify(copy, None)
    os.remove(copy / "tokenizer.json")
    with pytest.raises(PackageError, match="miss"):
        pkg.verify(copy, None)


def test_a_package_manifest_must_agree(vela_package, tmp_path):
    import shutil

    copy = tmp_path / "route"
    shutil.copytree(vela_package, copy)
    listed = {"files": {"config.json": {"sha256": sha256_file(copy / "config.json")}}}
    (copy / pkg.MANIFEST_FILE).write_text(json.dumps(listed))
    verified = pkg.verify(copy, None)
    assert verified.manifest_sha256 == sha256_file(copy / pkg.MANIFEST_FILE)
    listed["files"]["config.json"]["sha256"] = "0" * 64
    (copy / pkg.MANIFEST_FILE).write_text(json.dumps(listed))
    with pytest.raises(PackageError, match="MANIFEST"):
        pkg.verify(copy, None)


def test_presets_expand_categories(vela_package):
    presets = pkg.presets(vela_package)
    assert sorted(presets) == [
        "domain",
        "hazard.violence",
        "hazard.weapons",
        "jailbreak",
    ]
    assert presets["domain"]["type"] == "choice"
    assert presets["hazard.weapons"] == {
        "type": "noul",
        "instructions": "Is this a harmful request, and does its harm involve weapons or explosives?",
        "criteria": {
            "false": "No. It is not harmful, or its harm is of another kind.",
            "true": "Yes. Fulfilling it would cause harm of this kind.",
        },
    }


def test_presets_reject_unknown_types(tmp_path):
    (tmp_path / pkg.PRESETS_FILE).write_text(json.dumps({"x": {"type": "Score"}}))
    with pytest.raises(PackageError):
        pkg.presets(tmp_path)


def test_per_type_temperatures_from_a_file(tmp_path):
    root = write_package(
        tmp_path / "q", temperatures={"choice": 1.5, "noul": 0.75, "score": 2.0}
    )
    assert pkg.verify(root, None).temperatures == {
        "choice": 1.5,
        "noul": 0.75,
        "score": 2.0,
    }


def test_family_verifies_against_the_built_in_pin(qwen_package, monkeypatch):
    family = Decision1Family()
    local = family.verify(PackageRef(qwen_package))
    assert local.licence is None and local.loaded_parameters > 0
    details = local.details["package"]
    pinned = replace(
        builtin.lookup("Decision-1.0-Eos-0.8B"),
        files=details.digests,
        model_sha256=details.model_sha256,
    )
    monkeypatch.setattr(builtin, "lookup", lambda name: pinned)
    ref = PackageRef(qwen_package, repo_id=pinned.repo_id, revision=pinned.revision)
    verified = family.verify(ref)
    assert verified.details["package"].verification == "pinned"
    assert verified.licence == "apache-2.0"
    monkeypatch.setattr(
        builtin, "lookup", lambda name: replace(pinned, model_sha256="0" * 64)
    )
    with pytest.raises(PackageError, match="identity"):
        family.verify(ref)


def test_only_measured_packages_consent_to_a_reduced_copy():
    consent = {
        model.repo_id.rsplit("/", 1)[1]: dict(model.reduced)
        for model in builtin.all_models("decision1")
        if model.reduced
    }
    assert consent == {
        f"Decision-1.0-{name}": {"cpu": "float32-packed"}
        for name in ("Kai-0.6B", "Lex-0.6B", "Route-0.6B")
    }


def test_built_in_table_pins_all_seven_packages():
    models = builtin.all_models("decision1")
    names = [model.repo_id.rsplit("/", 1)[1] for model in models]
    assert names == [
        f"Decision-1.0-{size}"
        for size in (
            "Kai-0.6B",
            "Lex-0.6B",
            "Route-0.6B",
            "Eos-0.8B",
            "Sol-2B",
            "Nox-4B",
            "Lux-9B",
        )
    ]
    for model in models:
        assert len(model.revision) == 40 and len(model.model_sha256) == 64
        assert pkg.POINTER_FILE in model.files
        assert all(len(digest) == 64 for digest in model.files.values())
        decoder = model.backbone == "qwen3_5_text"
        assert bool(model.kernel_choices) == decoder
    route = builtin.lookup("Decision-1.0-Route-0.6B")
    assert route.manifest_sha256 and pkg.PRESETS_FILE in route.files


def test_golden_references_cover_every_pin_and_device_class():
    from vllm_srun.families.decision1.family import GOLDEN_QUESTIONS

    for model in builtin.all_models("decision1"):
        assert set(model.golden_answers) == {"cpu", "rocm"}, model.repo_id
        for answers in model.golden_answers.values():
            presets = {name for name in answers if name.startswith("preset:")}
            assert set(answers) - presets == set(GOLDEN_QUESTIONS)
            assert bool(presets) == model.repo_id.endswith("Route-0.6B")
