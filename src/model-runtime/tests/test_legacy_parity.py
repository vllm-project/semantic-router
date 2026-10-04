"""The legacy parity driver's job selection and comparison rules (``tools/legacy_parity.py``)."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "legacy_parity.py"


@pytest.fixture(scope="module")
def lp():
    spec = importlib.util.spec_from_file_location("legacy_parity", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache(lp, tmp_path):
    """An HF cache layout holding every pinned snapshot's config."""
    for name, revision in lp.REVISIONS.items():
        snapshot = (
            tmp_path
            / f"models--vllm-sr--Vela-1.0-Encoder-307M-{name}"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"id2label": {"0": "safe", "1": "unsafe"}})
        )
    return tmp_path


def arguments(cache: Path, **values) -> argparse.Namespace:
    defaults = {
        "seed": 0,
        "recipe": "amd",
        "jobs": "",
        "inputs": None,
        "cache": str(cache),
        "limit": 0,
        "repeats": 1,
        "concurrency": 0,
        "seconds": 0.0,
    }
    return argparse.Namespace(**(defaults | values))


def sequence_spec(ids: list[str]) -> dict:
    return {
        "Job": "domain",
        "Repo": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
        "Revision": "r",
        "Mode": "sequence",
        "MaxTokens": 512,
        "Overflow": "truncate",
        "WindowSize": 0,
        "WindowOverlap": 0,
        "Inputs": [{"id": item, "text": item} for item in ids],
    }


def test_recorded_inputs_select_the_jobs_they_hold(lp, cache, tmp_path):
    recorded = [
        {"Job": "pii", "Inputs": [{"id": "p0", "text": "Call 555-0100"}]},
        {"Job": "domain", "Inputs": [{"id": "p1", "text": "Integrate x"}]},
    ]
    inputs = tmp_path / "legacy.jobs.json"
    inputs.write_text(json.dumps(recorded))
    specs = lp.job_specs(arguments(cache, inputs=str(inputs)))
    assert [spec["Job"] for spec in specs] == ["domain", "pii"]
    assert specs[0]["Inputs"] == recorded[1]["Inputs"]
    assert specs[1]["Device"] == "migraphx:0"
    with pytest.raises(SystemExit, match="no job 'shield'"):
        lp.job_specs(arguments(cache, inputs=str(inputs), jobs="shield"))


def test_without_recorded_inputs_every_recipe_job_runs(lp, cache):
    specs = lp.job_specs(arguments(cache, limit=1))
    assert [spec["Job"] for spec in specs] == list(lp.AMD_JOBS)
    assert all(len(spec["Inputs"]) == 1 for spec in specs)


def test_a_label_flip_fails_the_cpu_bar_and_a_near_tie_does_not(lp):
    spec = sequence_spec(["same", "tie", "flip"])
    legacy = {
        ("domain", "same"): {"result": {"Probabilities": [0.9, 0.1]}},
        ("domain", "tie"): {"result": {"Probabilities": [0.5003, 0.4997]}},
        ("domain", "flip"): {"result": {"Probabilities": [0.6, 0.4]}},
    }
    runtime = {
        ("domain", "same"): {"result": {"probabilities": [0.9, 0.1]}},
        ("domain", "tie"): {"result": {"probabilities": [0.4997, 0.5003]}},
        ("domain", "flip"): {"result": {"probabilities": [0.4, 0.6]}},
    }
    report = lp.compare_job(spec, legacy, runtime, lp.CPU_THRESHOLDS)
    assert report["near_ties"] == 1
    assert report["agreement"] == pytest.approx(2 / 3)
    assert {(d["id"], d["reason"]) for d in report["disagreements"]} == {
        ("tie", "label"),
        ("flip", "label"),
        ("flip", "probability"),
    }
    assert not report["passed"]
    without_flip = {key: value for key, value in legacy.items() if key[1] != "flip"}
    spec["Inputs"] = spec["Inputs"][:2]
    assert lp.compare_job(spec, without_flip, runtime, lp.CPU_THRESHOLDS)["passed"]


def test_token_spans_compare_code_points_and_list_every_difference(lp):
    text = "Grüße an Bob"
    spec = {
        **sequence_spec([]),
        "Job": "pii",
        "Mode": "tokens",
        "Inputs": [{"id": "t", "text": text}],
    }
    entity = {"EntityType": "person", "Start": 11, "End": 14, "Confidence": 0.91}
    legacy = {("pii", "t"): {"result": {"Entities": [entity]}}}
    span = {"label": "person", "start": 9, "end": 12, "probability": 0.9101}
    runtime = {("pii", "t"): {"result": {"spans": [span]}}}
    report = lp.compare_job(spec, legacy, runtime, lp.CPU_THRESHOLDS)
    assert report["passed"] and report["max_abs_delta"] == pytest.approx(1e-4)
    extra = {"label": "person", "start": 0, "end": 5, "probability": 0.6}
    runtime[("pii", "t")]["result"]["spans"].append(extra)
    report = lp.compare_job(spec, legacy, runtime, lp.CPU_THRESHOLDS)
    assert not report["passed"]
    assert report["disagreements"] == [
        {
            "id": "t",
            "reason": "spans",
            "legacy_only": [],
            "runtime_only": [("PERSON", 0, 5)],
        }
    ]


def test_a_partial_scan_compares_with_a_truncated_answer(lp):
    spec = {
        **sequence_spec([]),
        "Job": "pii_truncate",
        "Mode": "tokens",
        "Inputs": [{"id": "long", "text": "Bob"}],
    }
    entity = {"EntityType": "person", "Start": 0, "End": 3, "Confidence": 0.8}
    legacy = {
        ("pii_truncate", "long"): {
            "error": f"scan: {lp.PARTIAL} at 512 tokens",
            "result": {"Entities": [entity]},
        }
    }
    answer = {
        "input": {"truncated": True},
        "spans": [{"label": "person", "start": 0, "end": 3, "probability": 0.8}],
    }
    runtime = {("pii_truncate", "long"): {"result": answer}}
    report = lp.compare_job(spec, legacy, runtime, lp.CPU_THRESHOLDS)
    assert report["compared"] == 1 and report["passed"]
    answer["input"]["truncated"] = False
    report = lp.compare_job(spec, legacy, runtime, lp.CPU_THRESHOLDS)
    assert report["disagreements"][0]["reason"] == "error" and not report["passed"]


def test_inputs_both_sides_reject_are_counted_not_compared(lp):
    spec = sequence_spec(["short", "long"])
    legacy = {
        ("domain", "short"): {"result": {"Probabilities": [0.2, 0.8]}},
        ("domain", "long"): {"error": "input exceeds 8192 tokens"},
    }
    runtime = {
        ("domain", "short"): {"result": {"probabilities": [0.2, 0.8]}},
        ("domain", "long"): {"result": {"error": "max_length_exceeded"}},
    }
    report = lp.compare_job(spec, legacy, runtime, lp.ROCM_THRESHOLDS)
    assert (report["compared"], report["both_rejected"]) == (1, 1)
    assert report["passed"]
