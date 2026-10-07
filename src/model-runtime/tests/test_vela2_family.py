"""The vela2 family end to end on tiny packages: verification, both members, answers and the tree forward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import PackageError
from vllm_srun.families.vela2.family import (
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
    Vela2Family,
)
from vllm_srun.plugins.base import (
    DEADLINE,
    EngineOptions,
    PackageRef,
    SurfacePlan,
    TreeBatch,
)
from vllm_srun.testing.vela2 import write_decoder_package, write_encoder_package

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
# The 0.8B's backbone has as many gated-delta value heads as key heads; the 4B and 9B have twice as many.
DECODERS = ("decoder", "decoder-08b")


@pytest.fixture(scope="module")
def packages(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("vela2")
    return {
        "encoder": write_encoder_package(root / "encoder"),
        "decoder": write_decoder_package(root / "decoder"),
        "decoder-08b": write_decoder_package(
            root / "decoder-08b", seed=1, backbone={"linear_num_value_heads": 2}
        ),
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


@pytest.mark.parametrize("name", ["encoder", *DECODERS])
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
    assert ("span_heads" in response) == (name != "encoder")
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


@pytest.mark.parametrize("decoder", DECODERS)
def test_decoder_questions_do_not_depend_on_each_other(models, decoder) -> None:
    model = models[decoder]
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


@pytest.mark.parametrize("name", DECODERS)
def test_decoder_shared_context_path_runs_one_parts_pass(
    models, monkeypatch, name
) -> None:
    model = models[name]
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
    packed = model.run_shared(plan.items, 1)
    assert [c.layout for c in calls] == ["rows", "packed"]
    assert len(calls[0].prefixes) == 2 and len(calls[1].prefixes) == 1
    for exact_tree, packed_tree in zip(exact, packed, strict=True):
        for a, b in zip(exact_tree, packed_tree, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_shared_context_profile_packs_only_decoder_trees(models, name) -> None:
    from vllm_srun.plugins.base import Job
    from vllm_srun.profiles.shared_context import SharedContextProfile

    model = models[name]
    profile = SharedContextProfile()
    assert profile.available(model) is None
    profile.bind(model)
    plan = model.plan(GOLDEN_STATE, GOLDEN_QUESTIONS)
    job = Job(items=plan.items, deadline=None, enqueued=0.0, profile=profile.name)
    batches = profile.plan([job], model.forward_token_budget())
    assert [b.shared_prefix for b in batches] == ([1] if name == "decoder" else [0])
    surface = SurfacePlan("decisions", plan.items, plan.input_tokens, plan)
    assert model.finish_surface(
        surface, model.run_approximate(plan.items)
    ) == model.finish_surface(surface, model.run_shared(plan.items, 1))


def test_coalescing_profiles_fill_a_cpu_forward_only_up_to_the_budget(models) -> None:
    from vllm_srun.families.vela2.family import CPU_PACKED_TOKENS
    from vllm_srun.plugins.base import Job
    from vllm_srun.profiles.batching import BatchingProfile
    from vllm_srun.profiles.exact import ExactProfile
    from vllm_srun.scheduler.planner import padded

    model = models["encoder"]
    assert model.forward_token_budget() == CPU_PACKED_TOKENS
    assert models["decoder"].forward_token_budget() is None
    short = model.plan(GOLDEN_STATE, {"domain": GOLDEN_QUESTIONS["domain"]}).items
    long = model.plan("A much longer request about billing. " * 60, GOLDEN_QUESTIONS)
    jobs = [
        Job(items=items, deadline=None, enqueued=0.0, profile="batching")
        for items in [short] * 4 + [long.items] * 2
    ]
    exact = ExactProfile()
    exact.bind(model)
    planned = exact.plan(jobs, model.forward_token_budget())
    assert [[len(i) for _, i in b.parts] for b in planned] == [
        [len(job.items)] for job in jobs
    ]
    assert model.exact_batches([]) == []
    batches = BatchingProfile().plan(jobs, model.forward_token_budget())
    for batch in batches:
        rows = [job.items[i] for job, indices in batch.parts for i in indices]
        width = max(padded(len(row.ids)) for row in rows)
        assert len(rows) == 1 or width * len(rows) <= CPU_PACKED_TOKENS
    assert any(len(batch.parts) > 1 for batch in batches)


def test_packed_encoder_batches_match_padded_ones(models) -> None:
    model = models["encoder"]
    plans = [
        model.plan(GOLDEN_STATE | {"request": text}, GOLDEN_QUESTIONS)
        for text in ("Short note.", "A much longer request about billing. " * 12)
    ]
    items = [item for plan in plans for item in plan.items]

    def arrays(results):
        return [
            array
            for logits, block in results
            for array in [*logits, *([] if block is None else [block])]
        ]

    padded, packed = arrays(model.run(items)), arrays(model.run_approximate(items))
    assert len(padded) == len(packed)
    for a, b in zip(padded, packed, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-5, rtol=1e-5)
    single = model.plan(GOLDEN_STATE, {"domain": GOLDEN_QUESTIONS["domain"]}).items
    assert len(single) == 1
    for a, b in zip(
        arrays(model.run(single)), arrays(model.run_approximate(single)), strict=True
    ):
        assert np.array_equal(a, b)


def consent(monkeypatch, verified, reduced: dict[str, str]) -> None:
    """Make ``verified`` a built-in whose entry consents to ``reduced`` copies."""
    from vllm_srun.families.vela2 import family as module
    from vllm_srun.registry.tables.common import BuiltinModel

    entry = BuiltinModel(
        repo_id="vllm-sr/fixture",
        revision="0" * 40,
        family="vela2",
        model_sha256=verified.model_sha256,
        manifest_sha256="",
        loaded_parameters=0,
        backbone="modernbert",
        min_device_memory_gib=1,
        reduced=reduced,
    )
    monkeypatch.setattr(
        module.builtin,
        "by_identity",
        lambda identity: entry if identity == verified.model_sha256 else None,
    )


def test_encoder_consent_to_a_reduced_copy_comes_from_its_builtin_entry(
    packages, monkeypatch
) -> None:
    family = Vela2Family()
    encoder = family.verify(PackageRef(packages["encoder"]))
    unknown = family.describe(encoder).dtype
    assert (unknown.reduced_cpu, unknown.reduced_gpu) == (None, None)
    consent(monkeypatch, encoder, {"cpu": "float32-packed"})
    dtype = family.describe(encoder).dtype
    assert (dtype.reduced_cpu, dtype.reduced_gpu) == ("float32-packed", None)
    assert (dtype.autocast, dtype.bf16_resident) == (None, False)


def test_only_the_measured_copy_is_consented() -> None:
    from vllm_srun.registry import builtin

    consent = {
        model.repo_id.rsplit("/", 1)[1]: dict(model.reduced)
        for model in builtin.all_models("vela2")
        if model.reduced
    }
    assert consent == {"Vela-2.0-0.3B": {"cpu": "float32-packed"}}


def test_max_speed_runs_encoder_approximate_batches_on_the_copy(
    packages, monkeypatch
) -> None:
    from vllm_srun.engines.native.reduced import unavailable
    from vllm_srun.profiles.max_speed import MaxSpeedProfile

    reason = unavailable("float32-packed", torch.device("cpu"))
    if reason:
        pytest.skip(reason)
    family = Vela2Family()
    verified = family.verify(PackageRef(packages["encoder"]))
    consent(monkeypatch, verified, {"cpu": "float32-packed"})
    spec = family.describe(verified)
    accelerator = CPUAccelerator()
    device = accelerator.devices()[0]
    exact_model = NativeEngine().load(spec, accelerator, device, EngineOptions())
    assert "reduced" not in exact_model.receipt()
    engine_model = NativeEngine().load(
        spec, accelerator, device, MaxSpeedProfile().engine_options(EngineOptions())
    )
    assert engine_model.receipt()["reduced"]["kind"] == "float32-packed"
    model = family.load(verified, spec, engine_model)
    batches = []
    encode = engine_model.encode
    monkeypatch.setattr(
        engine_model, "encode", lambda batch: batches.append(batch) or encode(batch)
    )
    items = model.plan(GOLDEN_STATE, GOLDEN_QUESTIONS).items
    exact = model.run(items)
    assert batches and not any(batch.reduced for batch in batches)
    batches.clear()
    approximate = model.run_approximate(items)
    assert batches and all(batch.reduced and batch.lengths for batch in batches)
    for (exact_logits, exact_block), (logits, block) in zip(
        exact, approximate, strict=True
    ):
        for a, b in zip(exact_logits, logits, strict=True):
            np.testing.assert_allclose(b, a, atol=1e-4, rtol=1e-4)
        assert (exact_block is None) == (block is None)
        if block is not None:
            np.testing.assert_allclose(block, exact_block, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", DECODERS)
@pytest.mark.parametrize("layout", ["rows", "packed"])
def test_tree_blocks_equal_their_full_sequences(models, layout, name) -> None:
    engine_model = models[name].engine_model
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
    assert {key: answer["error"] for key, answer in response["answers"].items()} == {
        "big": "max_length_exceeded",
        "bad": "invalid_question",
    }
    assert response["answers"]["big"]["type"] == "choice"
    assert response["answers"]["bad"]["type"] == "rank"
    assert "type must be one of" in response["answers"]["bad"]["message"]
    plan = model.plan("text", {"q": GOLDEN_QUESTIONS["jailbreak"]})
    expired = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan), DEADLINE
    )
    assert expired["answers"]["q"] == {"type": "noul", "error": "deadline_exceeded"}
    with pytest.raises(ValueError):
        model.plan("   ", {"q": GOLDEN_QUESTIONS["jailbreak"]})


def test_noul_calibration_is_a_model_option(packages) -> None:
    from vllm_srun.plugins.base import RegistryOptions

    family = Vela2Family(RegistryOptions(model_options={"noul_calibration": True}))
    verified = family.verify(PackageRef(packages["decoder"]))
    spec = family.describe(verified)
    accelerator = CPUAccelerator()
    engine_model = NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )
    calibrated = family.load(verified, spec, engine_model)
    assert calibrated.answerer.noul_calibration is True
