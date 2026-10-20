"""Recording hardware goldens must retain a same-revision generic drift gate."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.plugins.decisions import compare_answers
from vllm_srun.supervision.readiness import golden_check

spec = importlib.util.spec_from_file_location(
    "golden_answers_tool", Path(__file__).parents[1] / "tools/golden_answers.py"
)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def values(noul):
    return {"q": {"type": "noul", "noul": noul}}


def entry(device, key, noul, revision="revision-1", repo="fixture/model"):
    return {
        "repo": repo,
        "revision": revision,
        "device": device,
        "golden_answers": {device.split(":")[0]: values(noul), key: values(noul)},
    }


@pytest.mark.parametrize("key", [None, "cpu:fixture"])
def test_real_recording_seeds_generic_and_optional_hardware_key(
    qwen3_package, tmp_path, monkeypatch, key
):
    monkeypatch.setattr(CPUAccelerator, "golden_reference", lambda self, device: key)
    args = argparse.Namespace(
        device="cpu",
        cache_dir=None,
        base_path=None,
        offline=True,
        autotune_cache=None,
    )
    produced = recorder.answers(args, str(qwen3_package), None)
    path = tmp_path / "goldens.json"
    recorder.record(path, [produced])
    recorded = next(iter(json.loads(path.read_text()).values()))["answers"]
    assert set(recorded) == ({"cpu", key} if key else {"cpu"})
    assert recorded["cpu"] == produced["golden_answers"]["cpu"]
    if key:
        assert recorded[key] == recorded["cpu"]


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
            entry("cpu", "old-hardware", 0.5),
            entry("cpu", "other-model", 0.4, repo="fixture/other"),
        ],
    )
    other = json.loads(path.read_text())["fixture/other"]
    recorder.record(path, [entry("cpu", "new-hardware", 0.7, revision="revision-2")])
    table = json.loads(path.read_text())
    assert table["fixture/model"] == {
        "revision": "revision-2",
        "answers": {"cpu": values(0.7), "new-hardware": values(0.7)},
    }
    assert table["fixture/other"] == other


def test_generic_only_record_keeps_existing_baseline(tmp_path):
    path = tmp_path / "goldens.json"
    recorder.record(path, [entry("cpu", "cpu", 0.5)])
    recorder.record(path, [entry("cpu", "cpu", 0.6)])
    assert json.loads(path.read_text())["fixture/model"]["answers"] == {
        "cpu": values(0.5)
    }
