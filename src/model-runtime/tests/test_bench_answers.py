from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_answers import validate_answers
from decision1_parity import _device_class, require_fla_pin

QUESTIONS = {"domain": {"type": "choice", "criteria": ["a", "b"]}}


def test_validate_answers_accepts_a_choice():
    validate_answers(
        {"domain": {"type": "choice", "choice": "a"}},
        QUESTIONS,
        "native",
        "p0",
    )


@pytest.mark.parametrize(
    "answers",
    [
        {"domain": {"error": {"code": "new_error"}}},
        {"domain": {"choice": "a"}, "extra": {"choice": "a"}},
        {},
        {"domain": None},
        {"domain": {}},
        {"other": {"type": "choice", "choice": "a"}},
        {"domain": {"type": "choice", "error": "invalid_question"}},
        {"domain": {"type": "choice", "error": "max_length_exceeded"}},
        {"domain": {"type": "choice", "error": "invalid_model_output"}},
    ],
)
def test_validate_answers_rejects_empty_missing_and_errors(answers):
    with pytest.raises(SystemExit):
        validate_answers(answers, QUESTIONS, "native", "p1")


def test_require_fla_pin_allows_models_without_recorded_choices():
    require_fla_pin(False, "no recorded choices for cuda:sm_90", "cuda:0")


def test_require_fla_pin_stops_when_recorded_choices_fail_to_install():
    with pytest.raises(SystemExit, match="FLA pin required"):
        require_fla_pin(False, "FLA is not installed", "cuda:0")


@pytest.mark.parametrize("device", ["cuda:0", "cuda:-1", "cuda:wrong"])
def test_cuda_pin_selection_never_invents_a_device(monkeypatch, device):
    from vllm_srun.accel.cuda import CUDAAccelerator

    monkeypatch.setattr(CUDAAccelerator, "devices", lambda self: [])
    with pytest.raises(SystemExit, match=r"device|architecture"):
        _device_class(device)


def test_cuda_pin_selection_uses_the_requested_device_architecture(monkeypatch):
    from vllm_srun.accel.cuda import CUDAAccelerator
    from vllm_srun.plugins.base import DeviceInfo

    gpu = DeviceInfo(accelerator="cuda", index=1, name="fixture", arch="sm_80")
    monkeypatch.setattr(CUDAAccelerator, "devices", lambda self: [gpu])
    assert _device_class("cuda:1") == "cuda:sm_80"
