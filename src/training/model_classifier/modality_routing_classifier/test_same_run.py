#!/usr/bin/env python3
"""Unit tests for the #3856 same-run helper.

No real-model download and no network access: the model-runtime adapter
tests use a tiny local `vllm-srun fixture` (seconds to generate, randomly
initialized) rather than the production Vela model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib import request as urllib_request

from same_run_harness import (
    host_identity,
    load_model_runtime_adapter,
    percentile,
    row_id_for,
    run_single_stream,
)
from same_run_pair import refuse_cross_host, refuse_mismatched_shape


def _make_qsl(n: int) -> list[dict]:
    return [
        {
            "qsl_index": i,
            "row_id": hex(i)[2:].zfill(16),
            "input_hash": hex(i)[2:].zfill(64),
            "label": "AR",
            "text": f"prompt {i}",
        }
        for i in range(n)
    ]


def _fast_classify(text: str) -> dict:
    return {
        "output": "AR",
        "tokenize_ns": 0,
        "forward_ns": 1_000_000,
        "e2e_ns": 1_000_000,
        "seq_len": len(text.split()),
    }


def _forward_stats(p99_ms: float) -> dict:
    """A run.forward-shaped dict with only p99_ms meaningfully set."""
    return {
        "n": 1,
        "mean_ms": p99_ms,
        "p50_ms": p99_ms,
        "p90_ms": p99_ms,
        "p95_ms": p99_ms,
        "p99_ms": p99_ms,
        "max_ms": p99_ms,
        "min_ms": p99_ms,
    }


def _run_fixture(
    row_ids: list[str],
    forward_p99_ms: float,
    max_length: int = 256,
    warmup_n: int = 20,
) -> dict:
    host = {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4}
    return {
        "host": host,
        "model": "stub-model",
        "run": {
            "binding": "hf",
            "max_length": max_length,
            "batch_size": 1,
            "warmup_n": warmup_n,
            "peak_rss_mb": 100.0,
            "cpu_s": 1.0,
            "forward": _forward_stats(forward_p99_ms),
        },
        "records": [
            {
                "row_id": row_id,
                "qsl_index": i,
                "input_hash": row_id,
                "label": "AR",
                "output": "AR",
                # Deliberately identical across runs: first-pass latency is
                # fast in both, so a buggy delta computed from records[]
                # alone would see no difference even though run.forward.p99_ms
                # (built from every pass) diverges sharply below.
                "forward_ms": 10.0,
                "e2e_ms": 10.0,
            }
            for i, row_id in enumerate(row_ids)
        ],
    }


class RowIdTests(unittest.TestCase):
    def test_hashes_text_only(self) -> None:
        prompt = "When was the 8088 processor released?"
        self.assertEqual(row_id_for(prompt), row_id_for(prompt))
        self.assertEqual(len(row_id_for(prompt)), 16)

    def test_stable_under_relabel(self) -> None:
        prompt = "How do I tie a bowline knot? Show me each step"
        # Gold changing AR → BOTH must not change the join key.
        self.assertEqual(row_id_for(prompt), row_id_for(prompt))


class HostPairTests(unittest.TestCase):
    def test_identity_requires_fields(self) -> None:
        with self.assertRaises(SystemExit):
            host_identity({})
        with self.assertRaises(SystemExit):
            host_identity(None)

    def test_same_host_pairs(self) -> None:
        host = {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4}
        refuse_cross_host({"host": host}, {"host": dict(host)})

    def test_cross_host_refused(self) -> None:
        left = {"host": {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4}}
        right = {"host": {"cpu_model": "AMD", "core_count": 16, "ram_gb": 64.0}}
        with self.assertRaises(SystemExit) as ctx:
            refuse_cross_host(left, right)
        self.assertIn("refusing to pair cross-host", str(ctx.exception))

    def test_pair_cli_refuses_cross_host(self) -> None:
        here = Path(__file__).resolve().parent
        baseline = {
            "host": {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4},
            "model": "bert",
            "run": {"peak_rss_mb": 1000, "cpu_s": 10, "binding": "hf"},
            "records": [],
        }
        candidate = {
            "host": {"cpu_model": "AMD", "core_count": 16, "ram_gb": 64.0},
            "model": "distil",
            "run": {"peak_rss_mb": 400, "cpu_s": 2, "binding": "hf"},
            "records": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "b.json"
            cand_path = Path(tmp) / "c.json"
            out_path = Path(tmp) / "p.json"
            base_path.write_text(json.dumps(baseline))
            cand_path.write_text(json.dumps(candidate))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(here / "same_run_pair.py"),
                    "--baseline",
                    str(base_path),
                    "--candidate",
                    str(cand_path),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing to pair cross-host", proc.stderr + proc.stdout)
            self.assertFalse(out_path.exists())


class PairShapeTests(unittest.TestCase):
    """Regression: same_run_pair must reject a non-fixed-shape comparison.

    Before this fix, pairing silently intersected row IDs and ignored
    run-setting mismatches, so a baseline with two rows at max_length=256
    "successfully" paired with a one-row candidate at max_length=128.
    """

    def test_same_shape_pairs(self) -> None:
        baseline = _run_fixture(["r1", "r2"], forward_p99_ms=10.0)
        candidate = _run_fixture(["r1", "r2"], forward_p99_ms=10.0)
        refuse_mismatched_shape(baseline, candidate)  # must not raise

    def test_mismatched_row_coverage_refused(self) -> None:
        baseline = _run_fixture(["r1", "r2"], forward_p99_ms=10.0)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0)
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("different row coverage", str(ctx.exception))

    def test_mismatched_max_length_refused(self) -> None:
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0, max_length=256)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0, max_length=128)
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("different settings", str(ctx.exception))

    def test_both_missing_max_length_refused(self) -> None:
        """Two absent values must not silently compare equal.

        A naive `baseline.get(key) != candidate.get(key)` check treats two
        runs that both omit max_length as "matching" (None == None), even
        though the comparison is actually unverifiable, not confirmed fair.
        """
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0)
        del baseline["run"]["max_length"]
        del candidate["run"]["max_length"]
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("missing", str(ctx.exception))

    def test_mismatched_warmup_n_refused(self) -> None:
        """Regression (Xunzhuo round 6): warmup_n was missing from
        RUN_SHAPE_KEYS entirely, so a --warmup 0 baseline could be paired
        against a --warmup 20 candidate and still emit a confident p99
        delta despite the two runs having different cold-start exposure.
        """
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0, warmup_n=0)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0, warmup_n=20)
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("different settings", str(ctx.exception))

    def test_both_missing_warmup_n_refused(self) -> None:
        """Same "both absent must not silently match" guard as
        test_both_missing_max_length_refused, for warmup_n."""
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0)
        del baseline["run"]["warmup_n"]
        del candidate["run"]["warmup_n"]
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("missing", str(ctx.exception))

    def test_one_sided_missing_warmup_n_refused(self) -> None:
        """Regression (Xunzhuo round 6, his exact repro): "or even a
        missing candidate value" -- only one side omits warmup_n, not both.
        """
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0, warmup_n=0)
        candidate = _run_fixture(["r1"], forward_p99_ms=10.0)
        del candidate["run"]["warmup_n"]
        with self.assertRaises(SystemExit) as ctx:
            refuse_mismatched_shape(baseline, candidate)
        self.assertIn("missing", str(ctx.exception))


class PairLatencyTests(unittest.TestCase):
    """Regression: paired latency deltas must use the full measurement
    window (run.forward), not the first-pass-only records[] join.

    Both fixtures below report identical 10 ms first-pass forward_ms, but
    the candidate's run.forward.p99_ms is 99 ms because of a slow later
    pass. Before this fix, delta_forward_p99_ms was computed by
    recomputing percentiles from the paired records -- 10 - 10 = 0 -- and
    completely missed the real +89 ms regression.
    """

    def test_delta_uses_full_measurement_window(self) -> None:
        here = Path(__file__).resolve().parent
        baseline = _run_fixture(["r1"], forward_p99_ms=10.0)
        candidate = _run_fixture(["r1"], forward_p99_ms=99.0)
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "b.json"
            cand_path = Path(tmp) / "c.json"
            out_path = Path(tmp) / "p.json"
            base_path.write_text(json.dumps(baseline))
            cand_path.write_text(json.dumps(candidate))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(here / "same_run_pair.py"),
                    "--baseline",
                    str(base_path),
                    "--candidate",
                    str(cand_path),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out_path.read_text())
            self.assertEqual(report["paired"]["delta_forward_p99_ms"], 89.0)


class MultiPassTests(unittest.TestCase):
    """Regression: latency_samples must span every measured pass, not first only."""

    def test_latency_samples_span_all_passes(self) -> None:
        """p99 reflects every measured pass, not only the first.

        A 2-row QSL with a classify function that returns 1 ms on pass 1 and
        100 ms on pass 2.  After two passes p99 must exceed 50 ms.  Before the
        fix it would be ≤ 1 ms because second-pass samples were discarded.
        """
        call_idx = [0]
        ns_per_call = [1_000_000, 1_000_000, 100_000_000, 100_000_000]

        def slow_second_pass(text: str) -> dict:
            ns = ns_per_call[min(call_idx[0], len(ns_per_call) - 1)]
            call_idx[0] += 1
            return {
                "output": "AR",
                "tokenize_ns": 0,
                "forward_ns": ns,
                "e2e_ns": ns,
                "seq_len": 5,
            }

        qsl = _make_qsl(2)
        # Mock perf_counter so the loop runs exactly two passes:
        # wall_before=0, end-of-pass-1 elapsed=0 (<1.0 → continue),
        # end-of-pass-2 elapsed=2.0 (≥1.0 → break), wall_s read=2.0.
        perf_seq = iter([0.0, 0.0, 2.0, 2.0])
        with mock.patch(
            "same_run_harness.time.perf_counter", side_effect=lambda: next(perf_seq)
        ):
            quality_records, latency_samples, meta = run_single_stream(
                slow_second_pass, qsl, warmup_n=0, min_duration_s=1.0
            )

        self.assertEqual(len(quality_records), 2, "first-pass quality rows preserved")
        self.assertEqual(
            len(latency_samples),
            4,
            "all-pass latency samples retained (2 rows x 2 passes)",
        )
        self.assertEqual(meta["scored_queries"], 4)
        self.assertEqual(meta["n_latency_samples"], 4)
        p99 = percentile([s["e2e_ms"] for s in latency_samples], 99)
        self.assertGreater(p99, 50.0, "p99 must reflect the slow second pass")

    def test_single_pass_quality_matches_latency(self) -> None:
        """When min_duration_s=0 the single pass produces matching counts."""
        qsl = _make_qsl(3)
        perf_seq = iter([0.0, 1.0, 1.0])
        with mock.patch(
            "same_run_harness.time.perf_counter", side_effect=lambda: next(perf_seq)
        ):
            quality_records, latency_samples, meta = run_single_stream(
                _fast_classify, qsl, warmup_n=0, min_duration_s=0.0
            )
        self.assertEqual(len(quality_records), 3)
        self.assertEqual(len(latency_samples), 3)
        self.assertEqual(meta["scored_queries"], 3)


class ModelRuntimeAdapterTests(unittest.TestCase):
    """Regression: load_model_runtime_adapter must talk to a real vllm-srun
    process, not a mocked HTTP shape -- same standard Xunzhuo held the old
    candle adapter to (a real compiled helper, not a phantom import).

    candle-binding was deleted upstream in #4512; every native model now
    runs behind the standalone vllm-srun HTTP service instead of in-process.
    `vllm-srun fixture --family task_heads --variant modality` writes a tiny,
    randomly-initialized model with the real AR/DIFFUSION/BOTH label set, no
    network access, loading in well under a second -- the real thing, just
    small, not a mock of it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("vllm-srun") is None:
            raise unittest.SkipTest(
                "vllm-srun not installed (pip install ./src/model-runtime)"
            )
        # The fixture is local-only; force offline so a test that would
        # otherwise attempt a network call fails fast instead of hanging.
        # load_model_runtime_adapter's subprocess inherits this.
        os.environ["HF_HUB_OFFLINE"] = "1"
        cls.tmp = tempfile.mkdtemp(prefix="model-runtime-fixture-")
        cls.fixture_dir = os.path.join(cls.tmp, "modality-fixture")
        subprocess.run(
            [
                "vllm-srun",
                "fixture",
                cls.fixture_dir,
                "--family",
                "task_heads",
                "--variant",
                "modality",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("HF_HUB_OFFLINE", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_missing_vllm_srun_raises_system_exit(self) -> None:
        """SystemExit with an install hint when the package is absent."""
        with mock.patch("shutil.which", return_value=None), self.assertRaises(
            SystemExit
        ) as ctx:
            load_model_runtime_adapter("any-model", 256)
        msg = str(ctx.exception)
        self.assertIn("vllm-srun", msg)

    def test_classify_round_trips_against_real_server(self) -> None:
        """A real vllm-srun fixture server produces a valid ClassifyResult."""
        classify = load_model_runtime_adapter(self.fixture_dir, max_length=256)
        try:
            result = classify("a short prompt")
            self.assertIn(result["output"], ("AR", "DIFFUSION", "BOTH"))
            self.assertGreater(result["e2e_ns"], 0)
            self.assertEqual(result["tokenize_ns"], 0)
        finally:
            classify.close()

    def test_no_proc_fs_skips_resource_attribution_honestly(self) -> None:
        """On macOS/Windows (no /proc), classify must still work, and must
        not report fabricated 0.0 resource numbers as if they were real.

        Before this fix, _proc_cpu_and_rss's own FileNotFoundError handling
        silently returned (0.0, 0.0) on macOS -- a measurement that looks
        real but isn't, the exact failure mode already found and fixed once
        before for this harness (round 2: a helper that actually burned
        0.5 CPU-seconds and 100 MiB reported as cpu_s: 0.0). Simulating
        _PROC_FS_AVAILABLE=False (rather than requiring an actual non-Linux
        box) and asserting helper_stats is never attached at all is the
        honest alternative: run_single_stream already treats a missing
        helper_stats the same as --binding hf, which has no subprocess.
        """
        with mock.patch("same_run_harness._PROC_FS_AVAILABLE", False):
            classify = load_model_runtime_adapter(self.fixture_dir, max_length=256)
            try:
                result = classify("a short prompt")
                self.assertIn(result["output"], ("AR", "DIFFUSION", "BOTH"))
                self.assertFalse(hasattr(classify, "helper_stats"))
            finally:
                classify.close()

    def test_run_single_stream_attributes_helper_resources(self) -> None:
        """Regression: the server subprocess's CPU/RSS must be folded into
        cpu_s/peak_rss_mb, read from /proc/<pid> now instead of a
        self-reporting JSON field (model-runtime's /metrics has no
        process-level CPU/RSS -- confirmed no ProcessCollector is attached).
        Same regression shape the round-2 fix proved for the old helper.
        """
        classify = load_model_runtime_adapter(self.fixture_dir, max_length=256)
        try:
            qsl = _make_qsl(3)
            _, _, meta = run_single_stream(
                classify, qsl, warmup_n=1, min_duration_s=0.0
            )
            self.assertTrue(meta["includes_helper_resources"])
            self.assertGreater(meta["cpu_s"], 0.0)
            self.assertGreater(meta["peak_rss_mb"], 0.0)
        finally:
            classify.close()

    def test_repeated_rows_do_not_hit_result_cache(self) -> None:
        """Regression (Xunzhuo round 5): the fixed QSL repeats every row
        across warmup and later measurement passes. With model-runtime's
        default 16,384-entry result cache left enabled, the second and
        third requests for the same text are cache hits -- measuring "how
        fast the cache answers," not a real forward pass. serve must be
        launched with --result-cache-entries 0, so the real
        vllm_srun_result_cache{...,outcome="hit"} series on the server's
        own /metrics endpoint stays absent (or zero) even after sending the
        exact same text three times in a row.
        """
        classify = load_model_runtime_adapter(self.fixture_dir, max_length=256)
        try:
            same_text = "a short prompt"
            for _ in range(3):
                result = classify(same_text)
                self.assertIn(result["output"], ("AR", "DIFFUSION", "BOTH"))

            request = urllib_request.Request(classify.base_url + "/metrics")
            with urllib_request.urlopen(request, timeout=5) as response:
                metrics_text = response.read().decode("utf-8")

            hit_lines = [
                line
                for line in metrics_text.splitlines()
                if line.startswith("vllm_srun_result_cache") and 'outcome="hit"' in line
            ]
            for line in hit_lines:
                value = float(line.rsplit(" ", 1)[-1])
                self.assertEqual(
                    value,
                    0.0,
                    f"result cache recorded a hit despite --result-cache-entries 0: {line!r}",
                )
        finally:
            classify.close()

    def test_zero_warmup_baseline_reflects_real_load_cost(self) -> None:
        """Regression (Xunzhuo round 5): with the supported --warmup 0,
        run_single_stream snapshots classify.helper_stats["cpu_s"] as its
        baseline before sending a single request. Before this fix that
        baseline was a hardcoded 0.0, so the ready child's already-spent
        model-load CPU (the real fixture showed 6.48s of child CPU before
        measurement) leaked into the very first measured request's delta.
        Asserting the baseline is already > 0 immediately after the
        adapter is constructed -- before classify() is ever called --
        proves it is seeded from a real reading, not a stale zero.
        """
        classify = load_model_runtime_adapter(self.fixture_dir, max_length=256)
        try:
            self.assertGreater(classify.helper_stats["cpu_s"], 0.0)
        finally:
            classify.close()


if __name__ == "__main__":
    unittest.main()
