import os

from vllm_sr_runtime.accel.autotune import freeze_autotune
from vllm_sr_runtime.cli import build_parser, config_from_args
from vllm_sr_runtime.supervision.readiness import (
    CPU_TOLERANCE,
    GPU_TOLERANCE,
    golden_check,
)

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


def test_autotune_cache_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SR_RUNTIME_AUTOTUNE_CACHE", "/cache/from-env")
    parsed = build_parser().parse_args(["serve", "/models/kai"])
    assert config_from_args(parsed).autotune_cache == "/cache/from-env"
    parsed = build_parser().parse_args(
        ["serve", "/models/kai", "--autotune-cache", "/cache/flag"]
    )
    assert config_from_args(parsed).autotune_cache == "/cache/flag"
