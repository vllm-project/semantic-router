"""The legacy parity driver's job selection and comparison rules (``tools/legacy_parity.py``)."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

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


def test_the_ab_summary_reports_paired_intervals_and_every_load_window(lp):
    legacy = [30e6, 32e6, 31e6, 29e6, 33e6]
    runtime = [10e6, 11e6, 10.5e6, 9.5e6, 11.5e6]
    windows = {"legacy": [40.0, 41.0], "exact": [110.0, 112.0]}
    report = {
        "domain": {
            "legacy": legacy,
            "runtime": runtime,
            "ratio": [a / b for a, b in zip(legacy, runtime, strict=True)],
            "throughput": windows,
        }
    }
    row = lp.ab_summary(report)["domain"]
    assert row["pairs"] == 5 and row["throughput_windows_per_s"] == windows
    assert row["throughput_per_s"] == {"legacy": 40.5, "exact": 111.0}
    for name in ("median_speedup", "p50_ratio", "p95_ratio"):
        low, high = row["ci95"][name]
        assert 2.8 < low <= high < 3.2
    assert row["ci95"] == lp.ab_summary(report)["domain"]["ci95"]


def test_ab_rounds_save_each_finished_round_and_resume_the_rotation(lp, tmp_path):
    specs = [sequence_spec(["a", "b"]), {**sequence_spec(["c"]), "Job": "pii"}]
    raw = tmp_path / "ab.raw.json"
    orders, windows = [], []

    def save(state):
        raw.write_text(json.dumps(state))

    async def pair(spec, item, runtime_first):
        if item["id"] == "b" and len(orders) == 4:
            raise RuntimeError("interrupted")
        orders.append((item["id"], runtime_first))
        return 3_000, 1_000

    async def window(side, spec):
        windows.append((spec["Job"], side))
        return 10.0

    def run(state, rounds, use_windows=True):
        asyncio.run(
            lp.ab_rounds(
                state,
                specs,
                rounds,
                pair,
                ["legacy", "exact", "batching"],
                window if use_windows else None,
                lambda: save(state),
            )
        )

    fresh = {"pair_rounds": 0, "window_rounds": 0, "report": {}}
    with pytest.raises(RuntimeError, match="interrupted"):
        run(fresh, 2)
    state = json.loads(raw.read_text())
    assert (state["pair_rounds"], state["window_rounds"]) == (1, 0)
    assert state["report"]["domain"]["legacy"] == [3_000, 3_000]
    run(state, 2, use_windows=False)
    assert orders == [
        ("a", False),
        ("b", False),
        ("c", False),
        ("a", True),
        ("a", True),
        ("b", True),
        ("c", True),
    ]
    assert not windows and state["report"]["domain"]["ratio"] == [3.0] * 4
    run(state, 3)
    assert orders[-3:] == [("a", False), ("b", False), ("c", False)]
    assert windows[:3] == [
        ("domain", "legacy"),
        ("domain", "exact"),
        ("domain", "batching"),
    ]
    assert windows[6:9] == [
        ("domain", "exact"),
        ("domain", "batching"),
        ("domain", "legacy"),
    ]
    assert windows[12:15] == [
        ("domain", "batching"),
        ("domain", "legacy"),
        ("domain", "exact"),
    ]
    assert json.loads(raw.read_text()) == state
    assert (state["pair_rounds"], state["window_rounds"]) == (3, 3)
    assert state["report"]["pii"]["throughput"] == {
        "legacy": [10.0] * 3,
        "exact": [10.0] * 3,
        "batching": [10.0] * 3,
    }


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


def test_corpus_refuses_to_substitute_edge_prompts_when_e2e_pool_is_empty(
    lp, monkeypatch
):
    monkeypatch.setattr(lp, "prompts", lambda: [])
    with pytest.raises(SystemExit, match="acceptance corpus is empty"):
        lp.corpus(0)


def test_ab_rounds_counts_failed_pairs_instead_of_dropping_them(lp):
    """Reproduction: a pair whose runtime side fails must be counted as failed,
    not silently excluded from every total."""
    specs = [sequence_spec(["a", "b", "c"])]
    report: dict[str, Any] = {}

    async def failing_pair(spec, item, runtime_first):
        return 1_000, -1

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, failing_pair, ["legacy"], None, lambda: None)
    )
    entry = report["domain"]
    assert entry["total_pairs"] == 3
    assert entry["failed_pairs"] == 3
    assert entry["legacy"] == [] and entry["runtime"] == [] and entry["ratio"] == []


def test_ab_rounds_records_successful_and_failed_pairs(lp):
    specs = [sequence_spec(["a", "b"])]
    report: dict[str, Any] = {}

    async def mixed_pair(spec, item, runtime_first):
        # "a" succeeds on both sides, "b" fails on the runtime side.
        return (3_000, 1_000) if item["id"] == "a" else (3_000, -1)

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, mixed_pair, ["legacy"], None, lambda: None)
    )
    entry = report["domain"]
    assert entry["total_pairs"] == 2
    assert entry["failed_pairs"] == 1
    assert entry["legacy"] == [3_000]
    assert entry["runtime"] == [1_000]
    assert entry["ratio"] == [3.0]


def test_ab_rounds_resume_preserves_pair_accounting(lp, tmp_path):
    specs = [sequence_spec(["a", "b"])]
    raw = tmp_path / "ab.raw.json"

    def save(state):
        raw.write_text(json.dumps(state))

    async def mixed_pair(spec, item, runtime_first):
        return (3_000, 1_000) if item["id"] == "a" else (3_000, -1)

    fresh = {"pair_rounds": 0, "window_rounds": 0, "report": {}}
    asyncio.run(
        lp.ab_rounds(fresh, specs, 1, mixed_pair, ["legacy"], None, lambda: save(fresh))
    )
    state = json.loads(raw.read_text())
    entry = state["report"]["domain"]
    assert entry["total_pairs"] == 2
    assert entry["failed_pairs"] == 1
    # Resuming continues accounting rather than resetting it.
    asyncio.run(
        lp.ab_rounds(state, specs, 2, mixed_pair, ["legacy"], None, lambda: save(state))
    )
    entry = state["report"]["domain"]
    assert entry["total_pairs"] == 4
    assert entry["failed_pairs"] == 2
    assert entry["ratio"] == [3.0, 3.0]


def test_ab_summary_all_failed_pairs_report_null_speedup(lp):
    """The issue's reproduction: every attempt fails, so the summary exposes
    the attempted/failed counts rather than only empty, NaN percentiles."""
    report = {
        "domain": {
            "legacy": [],
            "runtime": [],
            "ratio": [],
            "total_pairs": 3,
            "failed_pairs": 3,
        }
    }
    row = lp.ab_summary(report)["domain"]
    assert row["total_pairs"] == 3
    assert row["pairs"] == 0
    assert row["failed_pairs"] == 3
    assert row["pairs"] == 0 and row["total_pairs"] == 3
    assert row["failed_pairs"] / row["total_pairs"] == 1.0
    assert row["median_speedup"] is None
    assert row["legacy_ms"]["p50"] is None
    assert row["runtime_ms"]["p95"] is None
    assert "ci95" not in row


def test_ab_summary_mixed_success_and_failure(lp):
    report = {
        "domain": {
            "legacy": [2_000_000],
            "runtime": [1_000_000],
            "ratio": [2.0],
            "total_pairs": 4,
            "failed_pairs": 3,
        }
    }
    row = lp.ab_summary(report)["domain"]
    assert row["total_pairs"] == 4
    assert row["pairs"] == 1
    assert row["failed_pairs"] == 3
    assert row["pairs"] / row["total_pairs"] == pytest.approx(0.25)
    assert row["failed_pairs"] / row["total_pairs"] == pytest.approx(0.75)
    assert row["median_speedup"] == 2.0


def test_ab_summary_unknown_historical_accounting(lp):
    """Old checkpoints predate pair accounting, so attempted/failed totals are
    unknown rather than assumed zero; successful pairs stay derivable."""
    report = {
        "domain": {
            "legacy": [1_000_000],
            "runtime": [500_000],
            "ratio": [2.0],
        }
    }
    row = lp.ab_summary(report)["domain"]
    assert row["total_pairs"] is None
    assert row["failed_pairs"] is None
    assert row["pairs"] == 1
    assert row["median_speedup"] == 2.0


def test_ab_rounds_all_pairs_successful(lp):
    specs = [sequence_spec(["a", "b", "c"])]
    report: dict[str, Any] = {}

    async def ok_pair(spec, item, runtime_first):
        return 3_000, 1_000

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(lp.ab_rounds(state, specs, 1, ok_pair, ["legacy"], None, lambda: None))
    entry = report["domain"]
    assert entry["total_pairs"] == 3
    assert entry["failed_pairs"] == 0
    assert entry["new_total_pairs"] == 3
    assert entry["new_failed_pairs"] == 0
    assert entry["legacy"] == [3_000, 3_000, 3_000]
    assert entry["runtime"] == [1_000, 1_000, 1_000]
    assert entry["ratio"] == [3.0, 3.0, 3.0]


def test_ab_rounds_legacy_requests_failing(lp):
    specs = [sequence_spec(["a", "b"])]
    report: dict[str, Any] = {}

    async def legacy_fails(spec, item, runtime_first):
        return -1, 1_000

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, legacy_fails, ["legacy"], None, lambda: None)
    )
    entry = report["domain"]
    assert entry["total_pairs"] == 2
    assert entry["failed_pairs"] == 2
    assert entry["new_total_pairs"] == 2
    assert entry["new_failed_pairs"] == 2
    assert entry["legacy"] == []
    assert entry["runtime"] == []
    assert entry["ratio"] == []


def test_ab_rounds_both_sides_failing(lp):
    specs = [sequence_spec(["a", "b"])]
    report: dict[str, Any] = {}

    async def both_fail(spec, item, runtime_first):
        return -1, -1

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, both_fail, ["legacy"], None, lambda: None)
    )
    entry = report["domain"]
    assert entry["total_pairs"] == 2
    assert entry["failed_pairs"] == 2
    assert entry["legacy"] == []
    assert entry["runtime"] == []
    assert entry["ratio"] == []


def test_ab_rounds_zero_attempted_pairs(lp):
    specs = [sequence_spec([])]
    report: dict[str, Any] = {}

    async def unused_pair(spec, item, runtime_first):
        return 3_000, 1_000

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, unused_pair, ["legacy"], None, lambda: None)
    )
    entry = report["domain"]
    assert entry["total_pairs"] == 0
    assert entry["failed_pairs"] == 0
    assert entry["legacy"] == []
    assert entry["runtime"] == []
    assert entry["ratio"] == []


def test_ab_rounds_resume_historical_checkpoint_counts_new_attempts(lp, tmp_path):
    """Resuming an old checkpoint (no total_pairs/failed_pairs) must count new
    attempts without fabricating historical totals."""
    raw = tmp_path / "ab.raw.json"

    def save(state):
        raw.write_text(json.dumps(state))

    # Old checkpoint: has successful pairs but no pair accounting.
    old_state = {
        "pair_rounds": 1,
        "window_rounds": 0,
        "report": {
            "domain": {
                "legacy": [3_000],
                "runtime": [1_000],
                "ratio": [3.0],
            }
        },
    }
    raw.write_text(json.dumps(old_state))
    state = json.loads(raw.read_text())

    specs = [sequence_spec(["a", "b"])]

    async def mixed_pair(spec, item, runtime_first):
        return (3_000, 1_000) if item["id"] == "a" else (3_000, -1)

    asyncio.run(
        lp.ab_rounds(state, specs, 2, mixed_pair, ["legacy"], None, lambda: save(state))
    )
    entry = state["report"]["domain"]
    # Historical totals remain unknown (None).
    assert entry["total_pairs"] is None
    assert entry["failed_pairs"] is None
    # New attempts are counted separately.
    assert entry["new_total_pairs"] == 2
    assert entry["new_failed_pairs"] == 1
    # Successful pairs include both historical and new.
    assert entry["legacy"] == [3_000, 3_000]
    assert entry["runtime"] == [1_000, 1_000]
    assert entry["ratio"] == [3.0, 3.0]


def test_ab_rounds_save_resume_without_double_counting(lp, tmp_path):
    """Repeated checkpoint saves and resumes must not double-count attempts."""
    raw = tmp_path / "ab.raw.json"

    def save(state):
        raw.write_text(json.dumps(state))

    specs = [sequence_spec(["a", "b"])]

    async def mixed_pair(spec, item, runtime_first):
        return (3_000, 1_000) if item["id"] == "a" else (3_000, -1)

    # First run: 1 round.
    fresh = {"pair_rounds": 0, "window_rounds": 0, "report": {}}
    asyncio.run(
        lp.ab_rounds(fresh, specs, 1, mixed_pair, ["legacy"], None, lambda: save(fresh))
    )
    state = json.loads(raw.read_text())
    entry = state["report"]["domain"]
    assert entry["new_total_pairs"] == 2
    assert entry["new_failed_pairs"] == 1
    assert entry["total_pairs"] == 2
    assert entry["failed_pairs"] == 1

    # Resume: run 1 more round (round_index 1). Must not re-count round 0.
    asyncio.run(
        lp.ab_rounds(state, specs, 2, mixed_pair, ["legacy"], None, lambda: save(state))
    )
    entry = state["report"]["domain"]
    assert entry["new_total_pairs"] == 4
    assert entry["new_failed_pairs"] == 2
    assert entry["total_pairs"] == 4
    assert entry["failed_pairs"] == 2
    assert entry["legacy"] == [3_000, 3_000]
    assert entry["runtime"] == [1_000, 1_000]
    assert entry["ratio"] == [3.0, 3.0]


def test_ab_summary_json_contains_no_nan(lp):
    """Empty latency percentiles and median speedup serialize as JSON null,
    never NaN or Infinity."""
    report = {
        "domain": {
            "legacy": [],
            "runtime": [],
            "ratio": [],
            "total_pairs": 3,
            "failed_pairs": 3,
        }
    }
    row = lp.ab_summary(report)["domain"]
    payload = json.dumps(row)
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert row["legacy_ms"]["p50"] is None
    assert row["legacy_ms"]["p95"] is None
    assert row["runtime_ms"]["p50"] is None
    assert row["runtime_ms"]["p95"] is None
    assert row["median_speedup"] is None
    assert row["pairs"] == 0 and row["total_pairs"] == 3
    assert row["failed_pairs"] / row["total_pairs"] == 1.0


def test_ab_summary_all_successful(lp):
    report = {
        "domain": {
            "legacy": [3_000_000, 3_200_000, 3_100_000],
            "runtime": [1_000_000, 1_100_000, 1_050_000],
            "ratio": [3.0, 3.0, 3.0],
            "total_pairs": 3,
            "failed_pairs": 0,
        }
    }
    row = lp.ab_summary(report)["domain"]
    assert row["total_pairs"] == 3
    assert row["pairs"] == 3
    assert row["failed_pairs"] == 0
    assert row["pairs"] / row["total_pairs"] == 1.0
    assert row["median_speedup"] == 3.0


@pytest.mark.parametrize(
    "latencies",
    [
        (0, 1_000),
        (1_000, 0),
        (0, 0),
    ],
)
def test_ab_rounds_counts_zero_latency_as_failure(lp, latencies):
    specs = [sequence_spec(["a"])]
    report: dict[str, Any] = {}

    async def zero_latency_pair(spec, item, runtime_first):
        return latencies

    state = {"pair_rounds": 0, "window_rounds": 0, "report": report}
    asyncio.run(
        lp.ab_rounds(state, specs, 1, zero_latency_pair, ["legacy"], None, lambda: None)
    )

    entry = report["domain"]
    assert entry["total_pairs"] == 1
    assert entry["failed_pairs"] == 1
    assert entry["legacy"] == []
    assert entry["runtime"] == []
    assert entry["ratio"] == []
