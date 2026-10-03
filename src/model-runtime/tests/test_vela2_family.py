"""The vela2 family end to end on tiny packages: verification, both members, answers and the tree forward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.engines.native.engine import NativeEngine
from vllm_sr_runtime.errors import PackageError
from vllm_sr_runtime.families.vela2.family import (
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
    Vela2Family,
)
from vllm_sr_runtime.plugins.base import (
    DEADLINE,
    EngineOptions,
    PackageRef,
    SurfacePlan,
    TreeBatch,
)
from vllm_sr_runtime.testing.vela2 import write_decoder_package, write_encoder_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def packages(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("vela2")
    return {
        "encoder": write_encoder_package(root / "encoder"),
        "decoder": write_decoder_package(root / "decoder"),
        "short": write_decoder_package(root / "short", max_length=200, broad=False),
    }


def load(root: Path):
    family = Vela2Family()
    verified = family.verify(PackageRef(root))
    spec = family.describe(verified)
    accelerator = CPUAccelerator()
    engine_model = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    return family.load(verified, spec, engine_model)


@pytest.fixture(scope="module")
def models(packages):
    return {name: load(path) for name, path in packages.items()}


def ask(model, state, questions):
    plan = model.plan(state, questions)
    results = model.run(plan.items) if plan.items else []
    return model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), results
    )


def test_detects_only_vela2_packages(packages, tmp_path) -> None:
    family = Vela2Family()
    assert family.detect(PackageRef(packages["encoder"]))
    assert family.detect(PackageRef(packages["decoder"]))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "modernbert"}), encoding="utf-8"
    )
    assert not family.detect(PackageRef(tmp_path))


def test_verification_checks_sums_and_manifest_weights(packages, tmp_path) -> None:
    import shutil

    copy = Path(shutil.copytree(packages["decoder"], tmp_path / "copy"))
    verified = Vela2Family().verify(PackageRef(copy))
    assert verified.manifest_sha256 and verified.model_sha256
    sums = (copy / "SHA256SUMS").read_text(encoding="utf-8")
    calibration = json.loads((copy / "calibration.json").read_text(encoding="utf-8"))
    calibration["thresholds"]["set:*"] = 0.9
    (copy / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(PackageError, match="SHA256SUMS"):
        Vela2Family().verify(PackageRef(copy))
    (copy / "SHA256SUMS").unlink()
    Vela2Family().verify(PackageRef(copy))
    (copy / "SHA256SUMS").write_text(sums, encoding="utf-8")
    manifest = json.loads((copy / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files_sha256"]["model-00001-of-00002.safetensors"] = "0" * 64
    (copy / "MODEL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (copy / "SHA256SUMS").unlink()
    with pytest.raises(PackageError, match="MODEL_MANIFEST"):
        Vela2Family().verify(PackageRef(copy))


def test_bundled_engine_is_never_imported(models) -> None:
    import sys

    assert "vela2_inference" not in sys.modules


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_golden_request_answers_every_question_type(models, name) -> None:
    model = models[name]
    response = ask(model, GOLDEN_STATE, GOLDEN_QUESTIONS)
    answers = response["answers"]
    assert set(answers) == {
        "domain",
        "jailbreak",
        "urgency",
        "topics.medication",
        "topics.billing",
        "pii",
        "halu",
    }
    assert answers["domain"]["choice"] in GOLDEN_QUESTIONS["domain"]["criteria"]
    assert abs(sum(answers["domain"]["probabilities"].values()) - 1) < 1e-9
    assert 0 <= answers["domain"]["abstain_probability"] <= 1
    assert 0 <= answers["jailbreak"]["noul"] <= 1
    assert set(answers["urgency"]["legend"]) == {"0", "1", "2"}
    assert set(response["sets"]) == {"topics"} and set(response["spans"]) == {
        "pii",
        "halu",
    }
    assert set(response["thresholds"]) == {"topics", "pii", "halu"}
    assert ("span_heads" in response) == (name == "decoder")
    for span in response["spans"]["pii"]:
        assert GOLDEN_STATE["request"][span["start"] : span["end"]] == span["text"]
    assert response["usage"]["input_tokens"] > 0
    assert ask(model, GOLDEN_STATE, GOLDEN_QUESTIONS) == response
    assert model.info.question_types == ("choice", "noul", "score", "set", "span")
    assert model.info.presets == ("pii", "halu", "relevance")


def test_encoder_runs_the_readout_in_the_family(models) -> None:
    model = models["encoder"]
    assert model.member.graph is False
    assert (
        model.info.parameters
        == model.engine_model.parameter_count() + model.member.parameters()
    )


def test_decoder_questions_do_not_depend_on_each_other(models) -> None:
    model = models["decoder"]
    together = ask(model, GOLDEN_STATE, GOLDEN_QUESTIONS)
    for name in ("domain", "urgency"):
        alone = ask(model, GOLDEN_STATE, {name: GOLDEN_QUESTIONS[name]})
        for key, value in alone["answers"][name]["probabilities"].items():
            assert value == pytest.approx(
                together["answers"][name]["probabilities"][key], abs=1e-5
            )


def test_decoder_answers_a_halu_alias_with_the_callers_label(models) -> None:
    question = {
        "type": "span",
        "instructions": "Which spans are unsupported?",
        "criteria": {"Hallucinated": "a claim the context does not support"},
        "threshold": 0.0,
    }
    response = ask(models["decoder"], GOLDEN_STATE, {"h": question})
    assert response["span_heads"] == {"h": "router"}
    assert {span["label"] for span in response["spans"]["h"]} == {"Hallucinated"}


def test_decoder_shared_context_path_runs_one_parts_pass(models, monkeypatch) -> None:
    model = models["decoder"]
    state = {"request": "Tom Baker wrote", "answer": "Adults may take 6 grams."}
    questions = {
        "pii": {"preset": "pii", "over": "request"},
        "halu": {"preset": "halu"},
        "q": GOLDEN_QUESTIONS["jailbreak"],
    }
    plan = model.plan(state, questions)
    assert len(plan.rows) == 2 and len(plan.items) == 2
    calls = []
    tree = model.engine_model.tree
    monkeypatch.setattr(
        model.engine_model, "tree", lambda batch: calls.append(batch) or tree(batch)
    )
    exact = model.run(plan.items)
    packed = model.run(plan.items, shared_prefix=1)
    assert [c.layout for c in calls] == ["rows", "packed"]
    assert len(calls[0].prefixes) == 2 and len(calls[1].prefixes) == 1
    for exact_tree, packed_tree in zip(exact, packed, strict=True):
        for a, b in zip(exact_tree, packed_tree, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_shared_context_profile_packs_only_decoder_trees(models, name) -> None:
    from vllm_sr_runtime.plugins.base import Job
    from vllm_sr_runtime.profiles.shared_context import SharedContextProfile

    model = models[name]
    profile = SharedContextProfile()
    assert profile.available(model) is None
    plan = model.plan(GOLDEN_STATE, GOLDEN_QUESTIONS)
    job = Job(items=plan.items, deadline=None, enqueued=0.0, profile=profile.name)
    batches = profile.plan([job], model.forward_token_budget())
    assert [b.shared_prefix for b in batches] == ([1] if name == "decoder" else [0])


@pytest.mark.parametrize("layout", ["rows", "packed"])
def test_tree_blocks_equal_their_full_sequences(models, layout) -> None:
    engine_model = models["decoder"].engine_model
    prefixes = [list(range(5, 25)), list(range(60, 73))]
    blocks = [list(range(30, 37)), list(range(40, 52)), list(range(80, 85))]
    owners = [0, 0, 1]
    hidden = engine_model.tree(TreeBatch(prefixes, blocks, owners, layout)).hidden
    for index, (block, owner) in enumerate(zip(blocks, owners, strict=True)):
        ids = torch.tensor([prefixes[owner] + block])
        with torch.inference_mode():
            full = engine_model.backbone(ids, torch.ones_like(ids))[
                0, len(prefixes[owner]) :
            ]
        torch.testing.assert_close(
            hidden[index, : len(block)], full, atol=1e-4, rtol=1e-4
        )


def test_failed_rows_deadlines_and_invalid_questions(models) -> None:
    model = models["short"]
    response = ask(
        model,
        "text",
        {
            "big": GOLDEN_QUESTIONS["domain"] | {"instructions": "x " * 400},
            "bad": {"type": "rank"},
        },
    )
    assert response["answers"] == {
        "big": {"type": "choice", "error": "max_length_exceeded"},
        "bad": {"type": "rank", "error": "invalid_question"},
    }
    plan = model.plan("text", {"q": GOLDEN_QUESTIONS["jailbreak"]})
    expired = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), DEADLINE
    )
    assert expired["answers"]["q"] == {"type": "noul", "error": "deadline_exceeded"}
    with pytest.raises(ValueError):
        model.plan("   ", {"q": GOLDEN_QUESTIONS["jailbreak"]})


def test_noul_calibration_is_a_model_option(packages) -> None:
    from vllm_sr_runtime.plugins.base import RegistryOptions

    family = Vela2Family(RegistryOptions(model_options={"noul_calibration": True}))
    verified = family.verify(PackageRef(packages["decoder"]))
    spec = family.describe(verified)
    accelerator = CPUAccelerator()
    engine_model = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    calibrated = family.load(verified, spec, engine_model)
    assert calibrated.answerer.noul_calibration is True
