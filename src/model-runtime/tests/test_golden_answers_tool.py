"""Recording hardware goldens must retain a same-revision generic drift gate."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.errors import VerificationError
from vllm_srun.families.decision3.family import Decision3Family
from vllm_srun.plugins.decisions import compare_answers
from vllm_srun.runtime import Runtime
from vllm_srun.supervision.readiness import CPU_TOLERANCE, golden_check, qualified

spec = importlib.util.spec_from_file_location(
    "golden_answers_tool", Path(__file__).parents[1] / "tools/golden_answers.py"
)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def values(noul):
    return {"q": {"type": "noul", "noul": noul}}


def entry(
    device,
    key,
    noul,
    revision="revision-1",
    repo="fixture/model",
    torch=None,
    generic=None,
    dtype=None,
):
    """A recording on ``device``: ``noul`` for the hardware ``key``, ``generic`` (default ``noul``) for its class."""
    produced = {
        "repo": repo,
        "revision": revision,
        "device": device,
        "golden_answers": {
            qualified(device.split(":")[0], dtype or "auto"): values(
                noul if generic is None else generic
            ),
            key: values(noul),
        },
    }
    if torch is not None:
        produced["torch"] = torch
    if dtype is not None:
        produced["dtype"] = dtype
    return produced


def recording_args(dtype="auto"):
    return argparse.Namespace(
        device="cpu",
        dtype=dtype,
        cache_dir=None,
        base_path=None,
        offline=True,
        autotune_cache=None,
    )


def shifted(answers):
    """``answers`` with every compared value moved past the CPU tolerance."""
    moved = copy.deepcopy(answers)
    for answer in moved.values():
        if answer["type"] == "noul":
            answer["noul"] += 2 * CPU_TOLERANCE
        else:
            answer["probabilities"] = {
                option: value + 2 * CPU_TOLERANCE
                for option, value in answer["probabilities"].items()
            }
    return moved


@pytest.mark.parametrize("key", [None, "cpu:fixture"])
def test_real_recording_seeds_the_generic_baseline(
    qwen3_package, tmp_path, monkeypatch, key
):
    """The recording host's hardware entry answers as the baseline it seeds, so only the baseline is kept."""
    monkeypatch.setattr(CPUAccelerator, "golden_reference", lambda self, device: key)
    produced = recorder.answers(recording_args(), str(qwen3_package), None)
    assert set(produced["golden_answers"]) == ({"cpu", key} if key else {"cpu"})
    path = tmp_path / "goldens.json"
    recorder.record(path, [produced])
    recorded = next(iter(json.loads(path.read_text()).values()))["answers"]
    assert recorded == {"cpu": produced["golden_answers"]["cpu"]}


@pytest.mark.parametrize(
    "device,key",
    [("cpu", "cpu:avx512"), ("cuda:0", "cuda:sm_89"), ("rocm:0", "rocm:gfx942")],
)
def test_later_hardware_recording_preserves_generic_and_updates_qualified(
    tmp_path, device, key
):
    path = tmp_path / "goldens.json"
    generic = device.split(":")[0]
    recorder.record(path, [entry(device, key, 0.5)])
    recorder.record(
        path, [entry(device, key, 0.6), entry(device, "other-hardware", 0.7)]
    )
    expected = json.loads(path.read_text())["fixture/model"]["answers"]
    assert expected[generic] == values(0.5)
    assert expected[key] == values(0.6)
    assert expected["other-hardware"] == values(0.7)
    # An unrecorded host must still compare with the generic baseline and detect drift.
    result = golden_check(
        lambda surface, body: values(0.6),
        lambda surface, actual, reference, tolerance: compare_answers(
            actual, reference, tolerance
        ),
        [{"surface": "decisions", "body": {}, "expected": expected}],
        generic,
        "unrecorded-hardware",
    )
    assert result.status == "failed"
    assert result.reference == generic


def test_new_revision_replaces_old_references_and_keeps_other_models(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(
        path,
        [
            entry("cpu", "old-hardware", 0.5, generic=0.4),
            entry("cpu", "other-model", 0.4, repo="fixture/other"),
        ],
    )
    other = json.loads(path.read_text())["fixture/other"]
    recorder.record(
        path, [entry("cpu", "new-hardware", 0.7, revision="revision-2", generic=0.6)]
    )
    table = json.loads(path.read_text())
    assert table["fixture/model"] == {
        "revision": "revision-2",
        "answers": {"cpu": values(0.6), "new-hardware": values(0.7)},
    }
    assert table["fixture/other"] == other


def test_generic_only_record_keeps_existing_baseline(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu", 0.5)])
    recorder.record(path, [entry("cpu", "cpu", 0.6)])
    assert json.loads(path.read_text())["fixture/model"]["answers"] == {
        "cpu": values(0.5)
    }


def test_a_new_pytorch_starts_a_new_generic_baseline(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu:avx512", 0.5, torch="2.10.0+cpu")])
    recorder.record(path, [entry("cpu", "cpu:avx512", 0.6, torch="2.14.1+cpu")])
    recorder.record(path, [entry("cpu", "cpu:avx2", 0.7, torch="2.14.1+cpu")])
    recorded = json.loads(path.read_text())["fixture/model"]
    assert recorded["answers"] == {"cpu": values(0.6), "cpu:avx2": values(0.7)}
    assert recorded["torch"] == dict.fromkeys(recorded["answers"], "2.14.1+cpu")


def test_a_hardware_entry_is_kept_only_while_it_differs_from_its_baseline(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu:avx2", 0.5, torch="2.14.1+cpu")])
    assert json.loads(path.read_text())["fixture/model"]["answers"] == {
        "cpu": values(0.5)
    }
    recorder.record(path, [entry("cpu", "cpu:avx2", 0.6, torch="2.14.1+cpu")])
    recorded = json.loads(path.read_text())["fixture/model"]
    assert recorded["answers"] == {"cpu": values(0.5), "cpu:avx2": values(0.6)}
    # The class answers as its baseline again: its old entry would fail it.
    recorder.record(path, [entry("cpu", "cpu:avx2", 0.5, torch="2.14.1+cpu")])
    recorded = json.loads(path.read_text())["fixture/model"]
    assert recorded == {
        "revision": "revision-1",
        "answers": {"cpu": values(0.5)},
        "torch": {"cpu": "2.14.1+cpu"},
    }


def test_a_recording_at_another_dtype_keys_its_own_baseline(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu:avx512", 0.5, torch="2.14.1+cpu")])
    bf16 = "cpu:avx512@bfloat16"
    recorder.record(
        path,
        [entry("cpu", bf16, 0.7, torch="2.14.1+cpu", generic=0.6, dtype="bfloat16")],
    )
    recorder.record(
        path,
        [entry("cpu", bf16, 0.8, torch="2.14.1+cpu", generic=0.9, dtype="bfloat16")],
    )
    recorded = json.loads(path.read_text())["fixture/model"]
    assert recorded["answers"] == {
        "cpu": values(0.5),
        "cpu@bfloat16": values(0.6),
        bf16: values(0.8),
    }
    assert recorded["torch"] == dict.fromkeys(recorded["answers"], "2.14.1+cpu")


def test_a_baseline_without_its_pytorch_is_replaced(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu", 0.5)])
    recorder.record(path, [entry("cpu", "cpu", 0.6, torch="2.14.1+cpu")])
    assert json.loads(path.read_text())["fixture/model"] == {
        "revision": "revision-1",
        "answers": {"cpu": values(0.6)},
        "torch": {"cpu": "2.14.1+cpu"},
    }


def test_recording_replaces_references_the_model_no_longer_matches(
    d3_package, monkeypatch
):
    """A new PyTorch moved D3's BF16 CPU answers off their references: serving fails, recording records the new ones."""
    current = recorder.answers(recording_args(), str(d3_package), None)
    answered = current["golden_answers"]["cpu"]
    stale = shifted(answered)
    golden = Decision3Family.golden
    monkeypatch.setattr(
        Decision3Family,
        "golden",
        lambda self, package: [
            {**request, "expected": {"cpu": stale}} for request in golden(self, package)
        ],
    )
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(d3_package), device="cpu"),))
    )
    try:
        with pytest.raises(VerificationError, match="differ from the reference"):
            runtime.load()
    finally:
        runtime.stop()
    produced = recorder.answers(recording_args(), str(d3_package), None)
    assert produced["golden_answers"]["cpu"] == answered
    assert produced["readiness"]["status"] == "unverified"
    assert produced["previous"]["cpu"]["checked"] > 0
    assert produced["previous"]["cpu"]["matched"] == 0


@pytest.mark.parametrize("dtype", ["auto", "bfloat16"])
def test_d3_is_checked_against_the_references_recorded_at_its_dtype(
    d3_package, monkeypatch, dtype
):
    """At ``auto`` a CPU checks D3's one FP32 reference; ``--dtype bfloat16`` its CPU class's BF16 one."""
    key = "cpu:fixture"
    monkeypatch.setattr(CPUAccelerator, "golden_reference", lambda self, device: key)
    fp32 = recorder.answers(recording_args(), str(d3_package), None)
    bf16 = recorder.answers(recording_args("bfloat16"), str(d3_package), None)
    assert set(fp32["golden_answers"]) == {"cpu", key}
    assert set(bf16["golden_answers"]) == {"cpu@bfloat16", f"{key}@bfloat16"}
    assert fp32["golden_answers"]["cpu"] != bf16["golden_answers"]["cpu@bfloat16"]
    expected = {
        "cpu": fp32["golden_answers"]["cpu"],
        "cpu@bfloat16": shifted(bf16["golden_answers"]["cpu@bfloat16"]),
        f"{key}@bfloat16": bf16["golden_answers"][f"{key}@bfloat16"],
    }
    golden = Decision3Family.golden
    monkeypatch.setattr(
        Decision3Family,
        "golden",
        lambda self, package: [
            {**request, "expected": expected} for request in golden(self, package)
        ],
    )
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(d3_package), device="cpu", dtype=dtype),)
        )
    )
    try:
        runtime.load()
        result = runtime.lookup(None).health.golden
        assert result.status == "matched" and result.checked == result.matched > 0
        assert result.reference == ("cpu" if dtype == "auto" else f"{key}@bfloat16")
    finally:
        runtime.stop()


def test_family_sweep_skips_the_unserved_image_package(monkeypatch):
    served = argparse.Namespace(repo_id="vllm-sr/vela3-fast", revision="r1")
    unserved = argparse.Namespace(repo_id="vllm-sr/vela3-fast-vision", revision="r2")
    monkeypatch.setattr(
        recorder.builtin, "all_models", lambda family: (served, unserved)
    )
    seen: list[tuple[str, str | None]] = []

    def fake_answers(args, model, revision):
        seen.append((model, revision))
        return {"model": model}

    monkeypatch.setattr(recorder, "answers", fake_answers)
    monkeypatch.setattr(sys, "argv", ["golden_answers.py", "--family", "vela3"])
    recorder.main()
    assert seen == [("vllm-sr/vela3-fast", "r1")]


def test_explicit_model_still_attempts_the_unserved_package(monkeypatch):
    seen: list[tuple[str, str | None]] = []

    def fake_answers(args, model, revision):
        seen.append((model, revision))
        return {"model": model}

    monkeypatch.setattr(recorder, "answers", fake_answers)
    monkeypatch.setattr(sys, "argv", ["golden_answers.py", "vllm-sr/vela3-fast-vision"])
    recorder.main()
    assert seen == [("vllm-sr/vela3-fast-vision", None)]
