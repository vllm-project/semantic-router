"""Deterministic scheduling and loopback checks for fixed-arrival workflows."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import threading
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from .__main__ import main
from .cases import MODELS
from .semantic_arrival_parity import TimedSemanticEvidence
from .semantic_arrivals import (
    _apply_timed_semantics,
    make_trace,
    run_arrival_wave,
    summarize_arrivals,
    trace_record,
)
from .semantic_audit import audit_case
from .semantic_cases import generate_cases, wire_bytes
from .semantic_transport import HttpSample, measure_http
from .test_semantic import _Handler
from .transport import Endpoint


class ArrivalSchedulerTests(TestCase):
    def setUp(self) -> None:
        self.model = MODELS[0]
        self.cases = generate_cases(
            self.model,
            self.model,
            question_count=1,
            state_count=3,
            variants=3,
            seed=17,
        )
        self.old = Endpoint("old", "http://127.0.0.1:1/v1/systemone", None)
        self.new_single = Endpoint("new", "http://127.0.0.1:2/v1/systemone", None)
        self.new_batch = Endpoint("new", "http://127.0.0.1:2/v1/decision/batches", None)

    def test_trace_is_fixed_before_either_arm_runs(self) -> None:
        first = make_trace(self.cases, 6, 19, 2_000_000)
        second = make_trace(self.cases, 6, 19, 2_000_000)
        self.assertEqual(first, second)
        self.assertEqual(
            [item.offset_ns for item in first], [i * 2_000_000 for i in range(6)]
        )
        row = trace_record(first, q=1, s=3, c=2, phase="throughput", round_=0)
        self.assertEqual(len(row["sha256"]), 64)
        self.assertEqual(
            row, trace_record(second, q=1, s=3, c=2, phase="throughput", round_=0)
        )
        changed = trace_record(
            make_trace(self.cases, 6, 19, 3_000_000),
            q=1,
            s=3,
            c=2,
            phase="throughput",
            round_=0,
        )
        self.assertNotEqual(row["sha256"], changed["sha256"])

    def test_old_state_dispatch_round_robins_arrived_workflows(self) -> None:
        submitted: list[int] = []

        def measure(
            endpoint,
            spec,
            *,
            case_id,
            concurrency,
            phase,
            round_number,
            sequence,
            timeout_seconds,
            capture_response,
        ):
            submitted.append(sequence)
            started = time.perf_counter_ns()
            return HttpSample(
                arm=endpoint.arm,
                phase=phase,
                round=round_number,
                sequence=sequence,
                case_id=case_id,
                concurrency=concurrency,
                state_id=spec.state_id,
                request_kind="single",
                request_sha256=spec.sha256,
                request_bytes=len(spec.body),
                decisions=spec.decisions,
                started_ns=started,
                ended_ns=started + 1_000_000,
                status_code=200,
                error_code=None,
                response_sha256="a" * 64,
            )

        trace = make_trace(self.cases, 3, 17, 0)
        samples, workflows, wave = run_arrival_wave(
            trace,
            arm="old",
            phase="throughput",
            round_=0,
            c=1,
            old=self.old,
            new_single=self.new_single,
            new_batch=self.new_batch,
            timeout=1,
            measure=measure,
        )
        self.assertEqual(submitted, [0, 1, 2] * 3)
        self.assertEqual(len(samples), 9)
        self.assertEqual([row["http_calls"] for row in workflows], [3, 3, 3])
        self.assertEqual(wave["peak_dispatched_http"], 1)
        self.assertEqual(wave["successful_decisions"], 9)
        self.assertTrue(
            all(row["latency_ms"] >= row["client_queue_ms"] for row in workflows)
        )

    def test_client_worker_error_is_a_failed_complete_workflow(self) -> None:
        def broken(*args, **kwargs):
            raise RuntimeError("do not expose this private error")

        trace = make_trace(self.cases, 1, 17, 0)
        samples, workflows, wave = run_arrival_wave(
            trace,
            arm="new",
            phase="throughput",
            round_=0,
            c=1,
            old=self.old,
            new_single=self.new_single,
            new_batch=self.new_batch,
            timeout=1,
            measure=broken,
        )
        self.assertEqual(samples[0].error_code, "client_worker_error")
        self.assertEqual(workflows[0]["error_codes"], ["client_worker_error"])
        self.assertEqual(wave["successful_decisions"], 0)
        summary = summarize_arrivals(
            workflows
            + [{**workflows[0], "arm": "old", "success": True, "error_codes": []}],
            [wave, {**wave, "arm": "old", "successful_decisions": 3}],
            max_jitter_ms=1000,
        )
        self.assertFalse(summary["comparison"]["eligible"])
        self.assertIsNone(
            summary["comparison"]["new_over_old_successful_decisions_per_second"]
        )

    def test_summary_uses_scheduled_window_and_jitter_suppresses_ratio(self) -> None:
        workflows = [
            {
                "arm": arm,
                "success": True,
                "decisions": 8,
                "latency_ms": latency,
                "client_queue_ms": 2.0,
                "arrival_jitter_ms": jitter,
            }
            for arm, latency, jitter in (("old", 20.0, 1.0), ("new", 10.0, 2.0))
        ]
        waves = [
            {
                "arm": "old",
                "window_seconds": 2.0,
                "max_arrival_jitter_ms": 1.0,
            },
            {
                "arm": "new",
                "window_seconds": 1.0,
                "max_arrival_jitter_ms": 2.0,
            },
        ]
        summary = summarize_arrivals(workflows, waves, max_jitter_ms=3.0)
        self.assertEqual(
            summary["comparison"]["new_over_old_successful_decisions_per_second"],
            2.0,
        )
        self.assertEqual(
            summary["comparison"]["old_over_new_p50_scheduled_arrival_latency"],
            2.0,
        )
        ineligible = summarize_arrivals(workflows, waves, max_jitter_ms=1.5)
        self.assertEqual(
            ineligible["comparison"]["reasons"], ["new_arrival_jitter_exceeded"]
        )
        self.assertIsNone(
            ineligible["comparison"]["new_over_old_successful_decisions_per_second"]
        )


class ArrivalLoopbackTests(TestCase):
    def test_cli_writes_same_sealed_trace_for_both_arms(self) -> None:
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
        threads = []
        for server in servers:
            server.bodies = []
            server.invalid_batch = False
            server.legacy_preview = False
            server.token_delta = 0
            server.probability_shift = 0.0
            server.choice_flip = False
            server.metrics_lock = threading.Lock()
            server.metrics = {}
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "arrival-receipt"
                args = [
                    "semantic-arrivals",
                    "--model",
                    MODELS[0],
                    "--old-url",
                    f"http://127.0.0.1:{servers[0].server_port}/v1/systemone",
                    "--new-url",
                    f"http://127.0.0.1:{servers[1].server_port}/v1/systemone",
                    "--old-source-ref",
                    "a" * 40,
                    "--new-source-ref",
                    "b" * 40,
                    "--old-model-revision",
                    "c" * 40,
                    "--new-model-revision",
                    "c" * 40,
                    "--old-hardware",
                    "test GPU x1",
                    "--new-hardware",
                    "test GPU x1",
                    "--old-network-scope",
                    "loopback",
                    "--new-network-scope",
                    "loopback",
                    "--old-physical-batch-size",
                    "1",
                    "--new-physical-batch-size",
                    "8",
                    "--question-counts",
                    "3",
                    "--state-counts",
                    "1,4",
                    "--concurrencies",
                    "1,4",
                    "--variants",
                    "2",
                    "--warmup",
                    "1",
                    "--workflows",
                    "4",
                    "--rounds",
                    "2",
                    "--arrival-spacing-ms",
                    "0",
                    "--max-arrival-jitter-ms",
                    "1000",
                    "--output-dir",
                    str(output),
                ]
                self.assertEqual(main(args), 0)
                receipt = json.loads((output / "receipt.json").read_text())
                traces = json.loads((output / "arrival-traces.json").read_text())
                self.assertEqual(receipt["status"], "measured")
                self.assertEqual(receipt["audit"]["status"], "passed")
                self.assertEqual(
                    receipt["timed_semantic_evidence"]["archived_http"], 112
                )
                archive = output / "timed-semantic.jsonl.gz"
                self.assertEqual(
                    hashlib.sha256(archive.read_bytes()).hexdigest(),
                    receipt["timed_semantic_evidence"]["sha256"],
                )
                with gzip.open(archive, "rt", encoding="utf-8") as handle:
                    evidence_rows = [json.loads(line) for line in handle]
                self.assertEqual(len(evidence_rows), 112)
                self.assertEqual({row["arm"] for row in evidence_rows}, {"old", "new"})
                self.assertEqual(
                    {row["validation_status"] for row in evidence_rows}, {"passed"}
                )
                self.assertEqual(len(receipt["shapes"]), 4)
                self.assertEqual(len(traces["traces"]), 12)
                self.assertTrue(
                    all(
                        row["summary"]["comparison"]["eligible"]
                        for row in receipt["shapes"]
                    )
                )
                workflows = [
                    json.loads(line)
                    for line in (output / "workflows.jsonl").read_text().splitlines()
                ]
                for trace in traces["traces"]:
                    pair = [
                        row
                        for row in workflows
                        if (row["phase"], row["round"], row["arrival_trace_sha256"])
                        == (trace["phase"], trace["round"], trace["sha256"])
                    ]
                    self.assertEqual({row["arm"] for row in pair}, {"old", "new"})
                    for arm in ("old", "new"):
                        self.assertEqual(
                            [row["case_id"] for row in pair if row["arm"] == arm],
                            [item["case_id"] for item in trace["arrivals"]],
                        )
                        self.assertTrue(all(row["success"] for row in pair))
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

    def test_timed_old_and_new_drift_make_workflows_ineligible(self) -> None:
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
        threads = []
        for index, server in enumerate(servers):
            server.bodies = []
            server.invalid_batch = False
            server.legacy_preview = index == 0
            server.token_delta = 0
            server.probability_shift = 0.0
            server.choice_flip = False
            server.metrics_lock = threading.Lock()
            server.metrics = {}
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        try:
            old = Endpoint(
                "old",
                f"http://127.0.0.1:{servers[0].server_port}/v1/systemone",
                None,
                "legacy_preview",
            )
            new_single = Endpoint(
                "new",
                f"http://127.0.0.1:{servers[1].server_port}/v1/systemone",
                None,
            )
            new_batch = Endpoint(
                "new",
                f"http://127.0.0.1:{servers[1].server_port}/v1/decision/batches",
                None,
            )
            case = generate_cases(
                MODELS[0],
                MODELS[0],
                question_count=3,
                state_count=2,
                variants=1,
                seed=17,
            )[0]
            audit, _ = audit_case(
                case,
                old,
                new_single,
                new_batch,
                timeout=2,
                probability_tolerance=0.01,
            )
            self.assertTrue(audit["passed"])
            trace = make_trace((case,), 1, 17, 0)
            old_samples = [
                measure_http(
                    old,
                    spec,
                    case_id=case.id,
                    concurrency=2,
                    phase="throughput",
                    round_number=0,
                    sequence=0,
                    timeout_seconds=2,
                    capture_response=True,
                )
                for spec in case.old_singles
            ]
            new_sample = measure_http(
                new_batch,
                case.new_request,
                case_id=case.id,
                concurrency=2,
                phase="throughput",
                round_number=0,
                sequence=0,
                timeout_seconds=2,
                capture_response=True,
            )
            evidence = TimedSemanticEvidence(io.StringIO())
            kwargs = {
                "model_id": MODELS[0],
                "old_model_id": MODELS[0],
                "old_response_mode": "legacy_preview",
            }
            old_statuses, _ = evidence.validate_wave(
                old_samples, trace, {case.id: audit}, arm="old", **kwargs
            )
            new_statuses, _ = evidence.validate_wave(
                [new_sample], trace, {case.id: audit}, arm="new", **kwargs
            )
            self.assertEqual(set(old_statuses.values()), {"passed"})
            self.assertEqual(set(new_statuses.values()), {"passed"})

            old_body = json.loads(old_samples[0].response_body)
            old_body["answers"]["q0000"]["input_tokens"] += 1
            old_body["answers"]["q0001"]["input_tokens"] -= 1
            old_bytes = wire_bytes(old_body)
            changed_old = replace(
                old_samples[0],
                response_body=old_bytes,
                response_sha256=hashlib.sha256(old_bytes).hexdigest(),
            )
            new_body = json.loads(new_sample.response_body)
            new_body["results"][0]["answers"]["q0000"]["noul"] += 0.02
            new_bytes = wire_bytes(new_body)
            changed_new = replace(
                new_sample,
                response_body=new_bytes,
                response_sha256=hashlib.sha256(new_bytes).hexdigest(),
            )
            old_statuses, _ = evidence.validate_wave(
                [changed_old, old_samples[1]],
                trace,
                {case.id: audit},
                arm="old",
                **kwargs,
            )
            new_statuses, summary = evidence.validate_wave(
                [changed_new], trace, {case.id: audit}, arm="new", **kwargs
            )
            self.assertIn("timed_semantic_mismatch", old_statuses.values())
            self.assertIn("timed_semantic_mismatch", new_statuses.values())
            self.assertEqual(summary["validated_http"], 0)
            workflow = {
                "sequence": 0,
                "http_calls": 1,
                "success": True,
                "decisions": case.decisions,
                "error_codes": [],
            }
            wave = {"successful_workflows": 1, "successful_decisions": case.decisions}
            _apply_timed_semantics([workflow], wave, new_statuses, summary)
            self.assertFalse(workflow["success"])
            self.assertEqual(wave["successful_decisions"], 0)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)
