import os

import pytest
import torch
from vllm_srun.accel.autotune import freeze_autotune
from vllm_srun.cli import build_parser, config_from_args
from vllm_srun.families.decision2.family import Decision2Family
from vllm_srun.plugins.base import (
    DeviceInfo,
    PackageRef,
    RegistryOptions,
    VerifiedPackage,
)
from vllm_srun.plugins.decisions import compare_answers
from vllm_srun.registry import builtin
from vllm_srun.supervision.readiness import (
    CPU_TOLERANCE,
    GPU_TOLERANCE,
    compare_numbers,
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


def answers_of(produced):
    return lambda surface, body: produced


def decisions(surface, values, reference, tolerance):
    return compare_answers(values, reference, tolerance)


def numbers(surface, values, reference, tolerance):
    return compare_numbers(values, reference, tolerance)


def decisions_golden(expected):
    return {
        "surface": "decisions",
        "body": {"state": "s", "questions": QUESTIONS},
        "expected": expected,
    }


def check(produced, reference, device_class):
    golden = decisions_golden({device_class: reference})
    return golden_check(answers_of(produced), decisions, [golden], device_class)


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


def test_npu_run_to_run_drift_within_tolerance_passes():
    """V2.0 GDN decoders drift bitwise across runs on shared hosts (record:
    npu-parity-ascend910b1); the npu gate compares within the tolerance."""
    runs = iter(
        [
            {"q": {"type": "noul", "noul": 0.3008507118}},
            {"q": {"type": "noul", "noul": 0.3008507119}},
        ]
    )
    golden = decisions_golden({"npu": answers(0.3008507)})
    result = golden_check(lambda surface, body: next(runs), decisions, [golden], "npu")
    assert result.status == "matched"


def test_npu_run_to_run_drift_beyond_tolerance_fails():
    runs = iter(
        [
            {"q": {"type": "noul", "noul": 0.30}},
            {"q": {"type": "noul", "noul": 0.30 + 2 * GPU_TOLERANCE}},
        ]
    )
    golden = decisions_golden({"npu": answers(0.30)})
    result = golden_check(lambda surface, body: next(runs), decisions, [golden], "npu")
    assert result.status == "failed"
    assert result.detail is not None and "not deterministic" in result.detail


def test_npu_run_to_run_shape_change_fails():
    runs = iter(
        [
            {"q": {"type": "noul", "noul": 0.3}},
            {"q": {"other": {"type": "noul", "noul": 0.3}}},
        ]
    )
    golden = decisions_golden({"npu": answers(0.3)})
    result = golden_check(lambda surface, body: next(runs), decisions, [golden], "npu")
    assert result.status == "failed"


def test_npu_second_run_is_validated_against_the_reference():
    """Each run is checked against the reference, not just the first one.

    A second response can stay within the run-to-run tolerance of the first
    and still drift past the recorded reference (0.30 -> 0.319 -> 0.338
    exceeds the 0.02 tolerance); readiness fails rather than letting the
    second run lean on the first.
    """
    runs = iter([answers(0.319), answers(0.338)])
    golden = decisions_golden({"npu": answers(0.30)})
    result = golden_check(lambda surface, body: next(runs), decisions, [golden], "npu")
    assert result.status == "failed"


def test_npu_second_run_non_finite_values_fail():
    """A NaN leaf in the second response fails instead of slipping through the comparison.

    NaN compares False against every bound, so the drift check has to reject
    non-finite leaves explicitly.
    """
    runs = iter([answers(0.3), {"q": {"type": "noul", "noul": float("nan")}}])
    golden = decisions_golden({"npu": answers(0.30)})
    result = golden_check(lambda surface, body: next(runs), decisions, [golden], "npu")
    assert result.status == "failed"


@pytest.mark.parametrize("second,valid", [(0.3190000001, True), (0.338, False)])
def test_npu_checks_both_runs_against_the_selected_hardware_reference(second, valid):
    key = "npu:test-device"
    golden = decisions_golden({"npu": answers(0.32), key: answers(0.30)})
    # Both responses would pass the generic reference and run-to-run tolerance.
    responses = [answers(0.319), answers(second)]
    generic_runs = iter(responses)
    generic = golden_check(
        lambda surface, body: next(generic_runs), decisions, [golden], "npu"
    )
    assert generic.status == "matched"
    runs = iter(responses)
    qualified = golden_check(
        lambda surface, body: next(runs), decisions, [golden], "npu", key
    )
    assert qualified.status == ("matched" if valid else "failed")
    assert qualified.reference == key
    assert qualified.checked == 2
    assert qualified.matched == (2 if valid else 1)


def test_a_device_class_without_a_reference_is_unverified():
    other_class_only = decisions_golden({"rocm": answers(0.9)})
    result = golden_check(
        answers_of(answers(0.5)), decisions, [other_class_only], "cuda"
    )
    assert result.status == "unverified"
    assert result.checked == 0


def test_qualified_reference_precedes_the_legacy_reference():
    golden = decisions_golden(
        {"cpu": answers(0.6), "cpu:x86_64:avx2:bf16_emulated": answers(0.5)}
    )
    result = golden_check(
        answers_of(answers(0.5)),
        decisions,
        [golden],
        "cpu",
        "cpu:x86_64:avx2:bf16_emulated",
    )
    assert result.status == "matched"
    assert result.reference == "cpu:x86_64:avx2:bf16_emulated"
    assert result.checked == result.matched == 1
    # The unqualified check still rejects the original cross-host drift.
    assert (
        golden_check(answers_of(answers(0.5)), decisions, [golden], "cpu").status
        == "failed"
    )


def test_qualified_reference_does_not_widen_tolerance_or_try_another_reference():
    golden = decisions_golden(
        {
            "cpu": answers(0.5),
            "cpu:x86_64:avx2:bf16_emulated": answers(0.5 + 2 * CPU_TOLERANCE),
        }
    )
    result = golden_check(
        answers_of(answers(0.5)),
        decisions,
        [golden],
        "cpu",
        "cpu:x86_64:avx2:bf16_emulated",
    )
    assert result.status == "failed" and result.matched == 0
    assert result.reference == "cpu:x86_64:avx2:bf16_emulated"


def test_unrecorded_hardware_uses_only_its_legacy_device_class():
    golden = decisions_golden({"cpu": answers(0.5), "cuda:sm_89": answers(0.9)})
    result = golden_check(
        answers_of(answers(0.5)),
        decisions,
        [golden],
        "cpu",
        "cpu:x86_64:avx512:bf16_emulated",
    )
    assert result.status == "matched" and result.reference == "cpu"
    result = golden_check(
        answers_of(answers(0.5)), decisions, [golden], "cuda", "cuda:sm_80"
    )
    assert result.status == "unverified" and result.reference is None
    result = golden_check(
        answers_of(answers(0.9)), decisions, [golden], "cuda", "cuda:sm_89"
    )
    assert result.status == "matched" and result.reference == "cuda:sm_89"


@pytest.mark.parametrize("capability", ["AVX2", "AVX512", "NO AVX"])
@pytest.mark.parametrize("bf16", [False, True])
def test_accelerator_selects_the_cpu_instruction_set_reference(
    monkeypatch, capability, bf16
):
    from vllm_srun.accel.cpu import CPUAccelerator
    from vllm_srun.accel.cuda import CUDAAccelerator

    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: capability)
    device = DeviceInfo("cpu", None, "host", arch="x86_64", bf16=bf16)
    assert (
        CPUAccelerator().golden_reference(device)
        == f"cpu:x86_64:{capability.lower().replace(' ', '_')}:{'bf16_native' if bf16 else 'bf16_emulated'}"
    )
    assert (
        CUDAAccelerator().golden_reference(DeviceInfo("cuda", 0, "L4", arch="sm_89"))
        == "cuda:sm_89"
    )
    assert CPUAccelerator().golden_reference(DeviceInfo("cpu", None, "unknown")) is None


def test_freeze_autotune_points_triton_at_one_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_AUTOTUNING", raising=False)
    path = freeze_autotune(str(tmp_path / "autotune"))
    assert path.is_dir()
    assert os.environ["TRITON_CACHE_DIR"] == str(path)
    assert os.environ["TRITON_CACHE_AUTOTUNING"] == "1"


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
    recorded = builtin.kernel_choices(known.model_sha256, "rocm", "gfx942")
    assert recorded == known.kernel_choices["rocm:gfx942"]
    assert builtin.kernel_choices(known.model_sha256, "rocm", "gfx90a") == {}
    assert builtin.kernel_choices(known.model_sha256, "rocm", None) == {}
    gfx942 = DeviceInfo(accelerator="rocm", index=0, name="MI325X", arch="gfx942")
    assert Decision2Family(RegistryOptions()).kernel_choices(package, gfx942) == {}
    for model in builtin.all_models("decision2"):
        choices = model.kernel_choices.get("rocm:gfx942")
        if model.backbone == "qwen3":
            assert choices is None, model.repo_id
            continue
        assert choices["fla"] == "0.5.2", model.repo_id
        assert set(choices["kernels"]) == FLA_FORWARD_KERNELS, model.repo_id


def test_qwen3_5_decoders_pin_released_kernel_choices_on_cuda_sm_90() -> None:
    known = builtin.lookup("vllm-sr/Decision-2.0-Eos-0.8B")
    recorded = builtin.kernel_choices(known.model_sha256, "cuda", "sm_90")
    assert recorded == known.kernel_choices["cuda:sm_90"]
    assert builtin.kernel_choices(known.model_sha256, "cuda", "sm_89") == {}
    for family in ("decision1", "decision2", "vela2"):
        for model in builtin.all_models(family):
            choices = model.kernel_choices.get("cuda:sm_90")
            if model.backbone != "qwen3_5_text":
                assert choices is None, model.repo_id
                continue
            assert choices["fla"] == "0.5.2", model.repo_id
            assert choices["triton"] == "3.6.0", model.repo_id
            assert set(choices["kernels"]) == FLA_FORWARD_KERNELS, model.repo_id


def test_autotune_cache_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SRUN_AUTOTUNE_CACHE", "/cache/from-env")
    parsed = build_parser().parse_args(["serve", "/models/kai"])
    assert config_from_args(parsed).autotune_cache == "/cache/from-env"
    parsed = build_parser().parse_args(
        ["serve", "/models/kai", "--autotune-cache", "/cache/flag"]
    )
    assert config_from_args(parsed).autotune_cache == "/cache/flag"


def test_surface_goldens_compare_flattened_values():
    from vllm_srun.supervision.readiness import flatten

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
        golden_check(answers_of(values), numbers, [golden], "cpu").status == "matched"
    )
    shifted = {key: value + 2 * CPU_TOLERANCE for key, value in values.items()}
    assert (
        golden_check(answers_of(shifted), numbers, [golden], "cpu").status == "failed"
    )
    assert (
        golden_check(
            answers_of(values), numbers, [{**golden, "expected": {}}], "cpu"
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
