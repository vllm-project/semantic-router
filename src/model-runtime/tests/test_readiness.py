import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from vllm_sr_runtime.accel import autotune
from vllm_sr_runtime.accel.autotune import freeze_autotune
from vllm_sr_runtime.cli import build_parser, config_from_args
from vllm_sr_runtime.families.decision2.family import Decision2Family
from vllm_sr_runtime.plugins.base import (
    DeviceInfo,
    PackageRef,
    RegistryOptions,
    VerifiedPackage,
)
from vllm_sr_runtime.registry import builtin
from vllm_sr_runtime.supervision.readiness import (
    CPU_TOLERANCE,
    GPU_TOLERANCE,
    golden_check,
)

FLA_FORWARD_KERNELS = {
    "chunk_fwd_kernel_o",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "l2norm_fwd_kernel",
    "recompute_w_u_fwd_kernel",
}

QUESTIONS = {"q": {"type": "noul", "instructions": "x"}}


def answers(noul):
    return {"q": {"type": "noul", "noul": noul}}


def check(produced, reference, device_class):
    golden = {
        "state": "s",
        "questions": QUESTIONS,
        "expected": {device_class: reference},
    }
    return golden_check(lambda state, questions: produced, [golden], device_class)


def test_cpu_references_allow_instruction_set_rounding():
    assert (
        check(answers(0.5), answers(0.5 + CPU_TOLERANCE / 2), "cpu").status == "matched"
    )
    failed = check(answers(0.5), answers(0.5 + 2 * CPU_TOLERANCE), "cpu")
    assert failed.status == "failed"
    assert failed.matched == 0


def test_gpu_references_use_the_gpu_tolerance():
    assert (
        check(answers(0.5), answers(0.5 + GPU_TOLERANCE / 2), "rocm").status
        == "matched"
    )
    assert (
        check(answers(0.5), answers(0.5 + 2 * GPU_TOLERANCE), "rocm").status == "failed"
    )


def test_a_device_class_without_a_reference_is_unverified():
    other_class_only = {
        "state": "s",
        "questions": QUESTIONS,
        "expected": {"rocm": answers(0.9)},
    }
    result = golden_check(
        lambda state, questions: answers(0.5), [other_class_only], "cuda"
    )
    assert result.status == "unverified"
    assert result.checked == 0


def test_freeze_autotune_points_triton_at_one_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_AUTOTUNING", raising=False)
    path = freeze_autotune(str(tmp_path / "autotune"))
    assert path.is_dir()
    assert os.environ["TRITON_CACHE_DIR"] == str(path)
    assert os.environ["TRITON_CACHE_AUTOTUNING"] == "1"


CHOICES = {
    "fla": "0.5.2",
    "triton": "3.7.1",
    "kernels": {
        "l2norm_fwd_kernel": [
            {"key": [128, 1, "torch.bfloat16"], "config": {"kwargs": {"BT": 8}}},
            {"key": [128, 2, "torch.bfloat16"], "config": {"kwargs": {"BT": 32}}},
        ]
    },
}


def test_pinned_kernel_choices_become_fla_config_files(monkeypatch):
    monkeypatch.delitem(sys.modules, "fla", raising=False)
    monkeypatch.setattr(autotune, "fla_version", lambda: "0.5.2")
    monkeypatch.setenv("FLA_CONFIG_DIR", "")
    monkeypatch.setenv("FLA_CACHE_MODE", "")
    directory = autotune.pin_kernel_choices(CHOICES)
    assert os.environ["FLA_CONFIG_DIR"] == str(directory)
    assert os.environ["FLA_CACHE_MODE"] == "full"
    document = json.loads((directory / "l2norm_fwd_kernel.json").read_text())
    # FLA's AutotuneKey.key_hash: MD5 of the compact, key-sorted JSON of the tuning key.
    digest = hashlib.md5(b'[128,2,"torch.bfloat16"]').hexdigest()
    assert document["autotune_entries"][digest]["config"] == {"kwargs": {"BT": 32}}
    assert document["default_config"] == {"kwargs": {"BT": 8}}


def test_kernel_choices_need_the_recorded_fla_before_it_is_imported(monkeypatch):
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)
    monkeypatch.setattr(autotune, "fla_version", lambda: "0.6.0")
    monkeypatch.delitem(sys.modules, "fla", raising=False)
    assert autotune.pin_kernel_choices(CHOICES) is None
    monkeypatch.setattr(autotune, "fla_version", lambda: "0.5.2")
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    assert autotune.pin_kernel_choices(CHOICES) is None
    assert "FLA_CONFIG_DIR" not in os.environ


def choices_of(*entries):
    return {
        "fla": "0.5.2",
        "kernels": {
            name: [{"key": key, "config": {"kwargs": {"BT": bt}}}]
            for name, key, bt in entries
        },
    }


def test_choices_merge_into_one_set_and_refuse_a_conflict():
    first = choices_of(("l2norm_fwd_kernel", [128, 1], 8))
    second = choices_of(
        ("l2norm_fwd_kernel", [256, 1], 16), ("chunk_fwd_kernel_o", [64], 4)
    )
    merged = autotune.merge_kernel_choices(
        autotune.merge_kernel_choices({}, first), second
    )
    assert [e["key"] for e in merged["kernels"]["l2norm_fwd_kernel"]] == [
        [128, 1],
        [256, 1],
    ]
    assert set(merged["kernels"]) == {"l2norm_fwd_kernel", "chunk_fwd_kernel_o"}
    assert autotune.merge_kernel_choices(merged, first) == merged
    with pytest.raises(autotune.KernelChoiceConflict, match="two configurations"):
        autotune.merge_kernel_choices(
            merged, choices_of(("l2norm_fwd_kernel", [128, 1], 32))
        )
    with pytest.raises(autotune.KernelChoiceConflict, match="FLA 0.6.0"):
        autotune.merge_kernel_choices(merged, {**first, "fla": "0.6.0"})


@pytest.fixture()
def two_fla_models(tmp_path, monkeypatch):
    """Two Decision 1.0 decoders whose built-in entries would pin different FLA choices."""
    from vllm_sr_runtime import runtime as runtime_module
    from vllm_sr_runtime.families.decision1.family import Decision1Family
    from vllm_sr_runtime.testing.decision1 import write_package

    roots = [
        write_package(tmp_path / name, seed=index, model_name=name)
        for index, name in enumerate(("first", "second"))
    ]
    recorded = {
        "first": choices_of(("l2norm_fwd_kernel", [128, 1], 8)),
        "second": choices_of(("l2norm_fwd_kernel", [256, 1], 16)),
    }
    events = []
    load = Decision1Family.load

    def loading(self, package, spec, engine_model):
        events.append(("load", package.model_name))
        return load(self, package, spec, engine_model)

    def pin(choices):
        events.append(("pin", choices))
        return autotune.pin_kernel_choices(choices)

    monkeypatch.setattr(
        Decision1Family,
        "kernel_choices",
        lambda self, package, device: recorded[package.model_name],
    )
    monkeypatch.setattr(Decision1Family, "load", loading)
    monkeypatch.setattr(runtime_module, "pin_kernel_choices", pin)
    monkeypatch.delitem(sys.modules, "fla", raising=False)
    monkeypatch.setattr(autotune, "fla_version", lambda: "0.5.2")
    monkeypatch.setenv("FLA_CONFIG_DIR", "")
    monkeypatch.setenv("FLA_CACHE_MODE", "")
    return roots, recorded, events


def serve(roots):
    from vllm_sr_runtime.config import ModelConfig, ServeConfig
    from vllm_sr_runtime.runtime import Runtime

    return Runtime(
        ServeConfig(
            models=tuple(
                ModelConfig(model=str(root), device="cpu", name=root.name)
                for root in roots
            )
        )
    )


def test_two_fla_models_in_one_process_pin_one_set_before_either_loads(
    two_fla_models,
):
    roots, recorded, events = two_fla_models
    runtime = serve(roots)
    runtime.start(background=False)
    try:
        assert [event[0] for event in events] == ["pin", "load", "load"]
        merged = events[0][1]
        assert merged == autotune.merge_kernel_choices(
            recorded["first"], recorded["second"]
        )
        document = json.loads(
            (Path(os.environ["FLA_CONFIG_DIR"]) / "l2norm_fwd_kernel.json").read_text()
        )
        assert {
            autotune.fla_key_hash([128, 1]),
            autotune.fla_key_hash([256, 1]),
        } == set(document["autotune_entries"])
        assert all(served.health.state == "ready" for served in runtime.served)
    finally:
        runtime.stop()


def test_a_model_whose_choices_conflict_is_refused_not_silently_unpinned(
    two_fla_models,
):
    roots, recorded, events = two_fla_models
    recorded["second"] = choices_of(("l2norm_fwd_kernel", [128, 1], 32))
    runtime = serve(roots)
    runtime.start()
    try:
        runtime.wait(timeout=120)
        first, second = runtime.served
        assert first.health.state == "ready"
        assert second.health.state == "failed"
        assert "conflict" in second.health.reason
        assert events[0] == ("pin", recorded["first"])
        assert ("load", "second") not in events
    finally:
        runtime.stop()
    with pytest.raises(RuntimeError, match="conflict"):
        serve(roots).start(background=False)


def test_built_in_models_pin_their_released_kernel_choices_on_gfx942(tmp_path):
    known = builtin.lookup("vllm-sr/Decision-2.0-Eos-0.8B")
    package = VerifiedPackage(
        ref=PackageRef(root=tmp_path),
        family="decision2",
        model_name="Decision-2.0-Eos-0.8B",
        manifest={},
        manifest_sha256="c" * 64,
        model_sha256=known.model_sha256,
        max_input_tokens=4096,
        licence="apache-2.0",
    )
    family = Decision2Family(RegistryOptions())
    gfx942 = DeviceInfo(accelerator="rocm", index=0, name="MI325X", arch="gfx942")
    assert family.kernel_choices(package, gfx942) == known.kernel_choices["rocm:gfx942"]
    other = DeviceInfo(accelerator="rocm", index=0, name="MI210", arch="gfx90a")
    assert family.kernel_choices(package, other) == {}
    for model in builtin.all_models("decision2"):
        choices = model.kernel_choices.get("rocm:gfx942")
        if model.backbone == "qwen3":
            assert choices is None, model.repo_id
            continue
        assert choices["fla"] == "0.5.2", model.repo_id
        assert set(choices["kernels"]) == FLA_FORWARD_KERNELS, model.repo_id


def test_autotune_cache_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SR_RUNTIME_AUTOTUNE_CACHE", "/cache/from-env")
    parsed = build_parser().parse_args(["serve", "/models/kai"])
    assert config_from_args(parsed).autotune_cache == "/cache/from-env"
    parsed = build_parser().parse_args(
        ["serve", "/models/kai", "--autotune-cache", "/cache/flag"]
    )
    assert config_from_args(parsed).autotune_cache == "/cache/flag"


def test_surface_goldens_compare_flattened_values():
    from vllm_sr_runtime.supervision.readiness import flatten

    response = {
        "results": [
            {"index": 0, "probabilities": [0.25, 0.75]},
            {
                "index": 1,
                "spans": [
                    {"label": "PERSON", "start": 0, "end": 3, "probability": 0.9}
                ],
            },
        ]
    }
    values = flatten("classify", response)
    assert values == {
        "0.probabilities.0": 0.25,
        "0.probabilities.1": 0.75,
        "1.span.0.PERSON.0.3": 0.9,
    }
    golden = {"surface": "classify", "body": {}, "expected": {"cpu": values}}
    assert (
        golden_check(None, [golden], "cpu", run_surface=lambda s, b: values).status
        == "matched"
    )
    shifted = {key: value + 2 * CPU_TOLERANCE for key, value in values.items()}
    assert (
        golden_check(None, [golden], "cpu", run_surface=lambda s, b: shifted).status
        == "failed"
    )
    assert (
        golden_check(
            None, [{**golden, "expected": {}}], "cpu", run_surface=lambda s, b: values
        ).status
        == "unverified"
    )
    assert flatten("embeddings", {"data": [{"index": 0, "embedding": [0.6, 0.8]}]}) == {
        "0.0": 0.6,
        "0.1": 0.8,
    }
    assert flatten("rerank", {"results": [{"index": 2, "logit": 1.5}]}) == {
        "2.logit": 1.5
    }
