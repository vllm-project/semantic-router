#!/usr/bin/env python3
"""Unit tests for the #3856 same-run helper (no model download)."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
import unittest.mock as mock
from pathlib import Path

from same_run_harness import (
    host_identity,
    load_candle_adapter,
    percentile,
    row_id_for,
    run_single_stream,
)
from same_run_pair import refuse_cross_host


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
            import subprocess

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
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing to pair cross-host", proc.stderr + proc.stdout)
            self.assertFalse(out_path.exists())


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
            "all-pass latency samples retained (2 rows × 2 passes)",
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


class CandleAdapterTests(unittest.TestCase):
    """Regression: load_candle_adapter must use the real Go helper, not a phantom import."""

    def _clear_helper_env(self) -> None:
        os.environ.pop("CANDLE_CLASSIFY_HELPER", None)

    def test_missing_helper_raises_system_exit(self) -> None:
        """SystemExit with a build hint when no binary is available."""
        self._clear_helper_env()
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(SystemExit) as ctx:
                load_candle_adapter("any-model", 256)
        msg = str(ctx.exception)
        self.assertIn("candle-classify", msg)
        self.assertIn("CANDLE_CLASSIFY_HELPER", msg)

    def test_helper_protocol_with_mock_binary(self) -> None:
        """A compliant Go helper binary produces a valid ClassifyResult."""
        helper_src = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys, json
            for line in sys.stdin:
                req = json.loads(line)
                resp = {{
                    "label": "AR",
                    "seq_len": len(req["text"].split()),
                    "tokenize_ns": 1_000_000,
                    "forward_ns": 5_000_000,
                    "helper_cpu_s": 0.25,
                    "helper_rss_mb": 210.5,
                }}
                print(json.dumps(resp), flush=True)
            """
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write(helper_src)
            helper_path = fh.name

        os.chmod(helper_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        try:
            os.environ["CANDLE_CLASSIFY_HELPER"] = helper_path
            classify = load_candle_adapter("stub-model", 256)
            try:
                result = classify("hello world test")
                self.assertEqual(result["output"], "AR")
                self.assertEqual(result["seq_len"], 3)
                self.assertEqual(result["tokenize_ns"], 1_000_000)
                self.assertGreater(result["e2e_ns"], 0)
                self.assertEqual(classify.helper_stats["cpu_s"], 0.25)
                self.assertEqual(classify.helper_stats["peak_rss_mb"], 210.5)
            finally:
                classify.close()
        finally:
            self._clear_helper_env()
            os.unlink(helper_path)

    def test_run_single_stream_attributes_helper_resources(self) -> None:
        """Regression: helper CPU/RSS must be folded into cpu_s/peak_rss_mb.

        Before this fix, run_single_stream measured only the parent process
        via time.process_time()/getrusage(RUSAGE_SELF). Since --binding
        candle runs inference in a child process, a helper that burns real
        CPU and allocates real memory was invisible: the parent only pays
        for pipe I/O, fractions of a millisecond. This reproduces Xunzhuo's
        review probe — a protocol-compatible helper that deliberately burns
        CPU and reports real RSS — and asserts the harness attributes it.
        """
        helper_src = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys, json, time
            for line in sys.stdin:
                req = json.loads(line)
                # Deliberately burn measurable CPU on every request.
                deadline = time.process_time() + 0.05
                while time.process_time() < deadline:
                    pass
                resp = {{
                    "label": "AR",
                    "seq_len": 0,
                    "tokenize_ns": 0,
                    "forward_ns": 1_000_000,
                    "helper_cpu_s": time.process_time(),
                    "helper_rss_mb": 137.0,
                }}
                print(json.dumps(resp), flush=True)
            """
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write(helper_src)
            helper_path = fh.name

        os.chmod(helper_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        try:
            os.environ["CANDLE_CLASSIFY_HELPER"] = helper_path
            classify = load_candle_adapter("stub-model", 256)
            try:
                qsl = _make_qsl(2)
                _, _, meta = run_single_stream(
                    classify, qsl, warmup_n=0, min_duration_s=0.0
                )
                self.assertTrue(meta["includes_helper_resources"])
                # 2 requests x ~0.05s deliberate CPU burn each. Before the
                # fix this was ~0.0 (parent pipe I/O only).
                self.assertGreater(meta["cpu_s"], 0.08)
                self.assertGreaterEqual(meta["peak_rss_mb"], 137.0)
            finally:
                classify.close()
        finally:
            self._clear_helper_env()
            os.unlink(helper_path)

    def test_helper_error_response_raises(self) -> None:
        """A helper that returns an error dict raises RuntimeError."""
        helper_src = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys, json
            for line in sys.stdin:
                print(json.dumps({{"error": "model not found"}}), flush=True)
            """
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
            fh.write(helper_src)
            helper_path = fh.name

        os.chmod(helper_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        try:
            os.environ["CANDLE_CLASSIFY_HELPER"] = helper_path
            classify = load_candle_adapter("bad-model", 256)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    classify("some text")
                self.assertIn("model not found", str(ctx.exception))
            finally:
                classify.close()
        finally:
            self._clear_helper_env()
            os.unlink(helper_path)


if __name__ == "__main__":
    unittest.main()
