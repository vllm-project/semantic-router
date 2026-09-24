"""Loopback contract tests for synthetic workflow and batch measurements."""

from __future__ import annotations

import json
import base64
import gzip
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

# cases initializes the source checkout's runtime-contract path.
# isort: off
from .cases import MODELS
from decision_runtime.confidence import choice_confidence, score_confidence

# isort: on

from decision_runtime.contracts import (
    SystemOneRequest,
    SystemOneResponse,
    validate_response_for_request,
)

from . import semantic_runner
from .__main__ import main
from .legacy_projection import project_legacy_preview
from .semantic_cases import generate_cases
from .semantic_metrics import MetricCapture, MetricsError, _parse_snapshot
from .semantic_report import build_semantic_matrix
from .semantic_runner import (
    MAX_TIMED_SEMANTIC_COMPRESSED_BYTES,
    MAX_TIMED_SEMANTIC_EVIDENCE_BYTES,
    TIMED_SEMANTIC_CONCURRENCIES,
)
from tools.ci.decision_timed_semantics import (
    MAX_ARCHIVE_BYTES,
    MAX_COMPRESSED_ARCHIVE_BYTES,
    TIMED_CONCURRENCIES,
    canonical_request_bodies,
)


def _answer(question, *, probability_shift=0.0, choice_flip=False):
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.5 + probability_shift}
    if question["type"] == "choice":
        keys = list(question["criteria"])
        probabilities = dict.fromkeys(keys, 1 / len(keys))
        if choice_flip:
            probabilities[keys[0]] -= 0.002
            probabilities[keys[1]] += 0.002
        return {
            "type": "choice",
            "choice": keys[1] if choice_flip else keys[0],
            "confidence": choice_confidence(tuple(probabilities.values())),
            "probabilities": probabilities,
        }
    levels = question["criteria"]
    probabilities = {str(index): 1 / len(levels) for index in range(len(levels))}
    return {
        "type": "score",
        "score": (len(levels) - 1) / 2,
        "confidence": score_confidence(tuple(probabilities.values())),
        "legend": {str(index): level for index, level in enumerate(levels)},
        "probabilities": probabilities,
    }


def _legacy_answer(question):
    answer = _answer(question)
    answer["input_tokens"] = 10
    if question["type"] != "noul":
        peak = max(answer["probabilities"].values())
        answer["top_probability"] = peak
        answer["confidence"] = peak
    return answer


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.bodies.append((self.path, body))
        request = json.loads(body)
        rows = len(request["questions"]) * len(request.get("states", [None]))
        with self.server.metrics_lock:
            counters = self.server.metrics.setdefault(
                request["model"],
                {
                    "row_preparation_seconds": 0.0,
                    "row_preparations": 0,
                    "physical_batches": 0,
                    "physical_batch_rows": 0,
                    "buckets": dict.fromkeys(
                        ("1", "2", "4", "8", "16", "32", "64", "+Inf"), 0
                    ),
                },
            )
            counters["row_preparation_seconds"] += rows / 1000
            counters["row_preparations"] += rows
            counters["physical_batches"] += 1
            counters["physical_batch_rows"] += rows
            for le in counters["buckets"]:
                if le == "+Inf" or rows <= int(le):
                    counters["buckets"][le] += 1
        if self.path == "/v1/systemone":
            legacy = self.server.legacy_preview
            response = {
                "model": request["model"],
                "answers": {
                    key: (
                        _legacy_answer(value)
                        if legacy
                        else _answer(
                            value,
                            probability_shift=self.server.probability_shift,
                            choice_flip=self.server.choice_flip,
                        )
                    )
                    for key, value in request["questions"].items()
                },
                "usage": {
                    "input_tokens": 10 * len(request["questions"])
                    + self.server.token_delta,
                    "output_tokens": 0 if legacy else 1,
                },
            }
            if legacy:
                response.update(
                    profile={"confidence_definition": "max(p); old preview"},
                    timing={"inference_ms": 1.0},
                    source="live_native",
                )
        elif self.path == "/v1/decision/batches":
            results = [
                {
                    "id": state["id"],
                    "answers": {
                        key: _answer(
                            value,
                            probability_shift=self.server.probability_shift,
                            choice_flip=self.server.choice_flip,
                        )
                        for key, value in request["questions"].items()
                    },
                    "usage": {
                        "input_tokens": 10 * len(request["questions"])
                        + self.server.token_delta,
                        "output_tokens": 1,
                    },
                }
                for state in request["states"]
            ]
            if self.server.invalid_batch:
                results.reverse()
            response = {
                "model": request["model"],
                "results": results,
                "usage": {
                    "input_tokens": (
                        10 * len(request["questions"]) + self.server.token_delta
                    )
                    * len(results),
                    "output_tokens": len(results),
                },
            }
        else:
            self.send_error(404)
            return
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        names = {
            "row_preparation_seconds": "decision_runtime_row_preparation_duration_seconds_total",
            "row_preparations": "decision_runtime_row_preparations_total",
            "physical_batches": "decision_runtime_physical_batches_total",
            "physical_batch_rows": "decision_runtime_physical_batch_rows_total",
        }
        lines = []
        with self.server.metrics_lock:
            for model, counters in self.server.metrics.items():
                for key, name in names.items():
                    lines.append(f'{name}{{model="{model}"}} {counters[key]}')
                for le, count in counters["buckets"].items():
                    lines.append(
                        "decision_runtime_physical_batch_size_bucket"
                        f'{{model="{model}",le="{le}"}} {count}'
                    )
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class GraphMetricTests(TestCase):
    def test_graph_replay_is_counted_and_bad_events_fail(self) -> None:
        model = "llm-semantic-router/Decision-1.0-Sol-2B"
        base = [
            f'decision_runtime_row_preparation_duration_seconds_total{{model="{model}"}} 1',
            f'decision_runtime_row_preparations_total{{model="{model}"}} 1',
            f'decision_runtime_physical_batches_total{{model="{model}"}} 1',
            f'decision_runtime_physical_batch_rows_total{{model="{model}"}} 8',
            f'decision_runtime_physical_batch_size_bucket{{model="{model}",le="+Inf"}} 1',
        ]
        before = _parse_snapshot("\n".join(base).encode(), "a" * 64, model)
        self.assertEqual(
            before.graph_events, dict.fromkeys(("capture", "replay", "fallback"), 0)
        )
        event = f'decision_runtime_qwen_rocm_graph_events_total{{model="{model}",event="replay"}} 3'
        after = _parse_snapshot("\n".join([*base, event]).encode(), "b" * 64, model)
        capture = MetricCapture("new", 8, 8, 32, 0, before, after, None)
        self.assertEqual(capture.public_record()["delta"]["graph_events"]["replay"], 3)
        for bad in (
            event.replace('event="replay"', 'event="unreviewed"'),
            event.replace(" 3", " 3.5"),
            "\n".join((event, event)),
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(
                    MetricsError, "metrics_invalid_graph_events"
                ):
                    _parse_snapshot("\n".join([*base, bad]).encode(), "c" * 64, model)


class SemanticTests(TestCase):
    def test_timed_gate_canonical_requests_match_benchmark_generator(self):
        for model in MODELS:
            for questions, states in ((32, 1), (8, 8), (32, 32)):
                with self.subTest(model=model, questions=questions, states=states):
                    cases = generate_cases(
                        model,
                        "old-decision-model",
                        question_count=questions,
                        state_count=states,
                        variants=4,
                        seed=17,
                    )
                    self.assertEqual(
                        {case.id: case.new_request.body for case in cases},
                        canonical_request_bodies(model, questions, states),
                    )

    def test_timed_all_concurrency_archive_contains_exact_http_bodies(self):
        self.assertEqual(TIMED_SEMANTIC_CONCURRENCIES, TIMED_CONCURRENCIES)
        self.assertEqual(MAX_ARCHIVE_BYTES, 120 * 1024 * 1024)
        self.assertEqual(MAX_COMPRESSED_ARCHIVE_BYTES, 24 * 1024 * 1024)
        self.assertEqual(MAX_TIMED_SEMANTIC_EVIDENCE_BYTES, MAX_ARCHIVE_BYTES)
        self.assertEqual(
            MAX_TIMED_SEMANTIC_COMPRESSED_BYTES,
            MAX_COMPRESSED_ARCHIVE_BYTES,
        )
        servers, threads = self._servers()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--concurrencies",
                    "1,8,32",
                    "--timed-semantic-evidence",
                )
                self.assertEqual(main(args), 0)
                with gzip.open(output / "timed-semantic.jsonl.gz", "rt") as handle:
                    rows = [json.loads(line) for line in handle]
                self.assertEqual(len(rows), 4 * 2 * 2 * 3)
                self.assertEqual({row["concurrency"] for row in rows}, {1, 8, 32})
                samples = [
                    json.loads(line)
                    for line in (output / "samples.jsonl").read_text().splitlines()
                ]
                for row in rows:
                    request_wire = base64.b64decode(
                        row["request_base64"], validate=True
                    )
                    self.assertEqual(
                        hashlib.sha256(request_wire).hexdigest(), row["request_sha256"]
                    )
                    wire = base64.b64decode(row["response_base64"], validate=True)
                    self.assertEqual(
                        hashlib.sha256(wire).hexdigest(), row["response_sha256"]
                    )
                    matched = [
                        sample
                        for sample in samples
                        if sample["arm"] == "new"
                        and sample["phase"] == "throughput"
                        and all(
                            sample[field] == row[field]
                            for field in (
                                "case_id",
                                "concurrency",
                                "round",
                                "sequence",
                                "request_sha256",
                                "response_sha256",
                            )
                        )
                    ]
                    self.assertEqual(len(matched), 1)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_timed_c1_archive_writer_fails_closed_at_raw_limit(self):
        servers, threads = self._servers()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--concurrencies",
                    "1",
                    "--timed-semantic-evidence",
                )
                with patch.object(
                    semantic_runner, "MAX_TIMED_SEMANTIC_EVIDENCE_BYTES", 1
                ):
                    self.assertEqual(main(args), 2)
                self.assertFalse((output / "receipt.json").exists())
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_timed_c1_archive_writer_fails_closed_at_compressed_limit(self):
        servers, threads = self._servers()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--concurrencies",
                    "1",
                    "--timed-semantic-evidence",
                )
                with patch.object(
                    semantic_runner, "MAX_TIMED_SEMANTIC_COMPRESSED_BYTES", 1
                ):
                    self.assertEqual(main(args), 2)
                self.assertFalse((output / "receipt.json").exists())
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_legacy_preview_projection_checks_old_statistic_and_distribution(self):
        request = SystemOneRequest.model_validate(
            {
                "model": "decision-nano-preview",
                "state": "Please refund the duplicate charge.",
                "questions": {
                    "refund": {
                        "type": "noul",
                        "instructions": "Is a refund requested?",
                    },
                    "category": {
                        "type": "choice",
                        "instructions": "Classify the request.",
                        "criteria": {"billing": None, "technical": None},
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate urgency.",
                        "criteria": ["low", "medium", "high"],
                    },
                },
            }
        )
        answers = {
            key: _legacy_answer(question.model_dump())
            for key, question in request.questions.items()
        }
        legacy = {
            "model": request.model,
            "answers": answers,
            "usage": {"input_tokens": 30, "output_tokens": 0},
            "profile": {"confidence_definition": "max(p); old preview"},
            "timing": {"inference_ms": 1.0},
            "source": "live_native",
        }
        projected = project_legacy_preview(legacy, request)
        validate_response_for_request(
            request, SystemOneResponse.model_validate(projected)
        )
        self.assertNotIn("profile", projected)
        self.assertNotIn("input_tokens", projected["answers"]["refund"])
        self.assertEqual(projected["answers"]["category"]["confidence"], 0.0)

        legacy["answers"]["category"]["confidence"] = 0.9
        with self.assertRaisesRegex(ValueError, "max-probability"):
            project_legacy_preview(legacy, request)

    def test_generator_is_deterministic_and_respects_contract(self):
        cases = generate_cases(
            MODELS[0], MODELS[0], question_count=9, state_count=4, variants=2, seed=17
        )
        repeated = generate_cases(
            MODELS[0], MODELS[0], question_count=9, state_count=4, variants=2, seed=17
        )
        self.assertEqual(cases, repeated)
        self.assertEqual(cases[0].decisions, 36)
        self.assertEqual(len(cases[0].old_singles), 4)
        self.assertFalse(cases[0].same_wire_bytes)
        self.assertEqual(
            {
                question.type
                for question in cases[0].new_request.request.questions.values()
            },
            {"noul", "choice", "score"},
        )
        single = generate_cases(
            MODELS[0], MODELS[0], question_count=3, state_count=1, variants=1, seed=17
        )[0]
        self.assertTrue(single.same_wire_bytes)
        self.assertEqual(single.old_singles[0].sha256, single.new_request.sha256)
        maximum = generate_cases(
            MODELS[0], MODELS[0], question_count=32, state_count=32, variants=1, seed=17
        )[0]
        self.assertEqual(maximum.decisions, 1024)
        self.assertEqual(len(maximum.old_singles), 32)
        with self.assertRaisesRegex(ValueError, "decision limit"):
            generate_cases(
                MODELS[0],
                MODELS[0],
                question_count=33,
                state_count=32,
                variants=1,
                seed=17,
            )

    def _servers(self, invalid_batch=False):
        servers = [ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
        for server in servers:
            server.bodies = []
            server.invalid_batch = invalid_batch
            server.legacy_preview = False
            server.token_delta = 0
            server.probability_shift = 0.0
            server.choice_flip = False
            server.metrics_lock = threading.Lock()
            server.metrics = {}
        servers[0].invalid_batch = False
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in servers
        ]
        for thread in threads:
            thread.start()
        return servers, threads

    def _run_args(self, servers, output, *extra):
        return [
            "semantic",
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
            "--latency-workflows",
            "2",
            "--throughput-workflows",
            "4",
            "--rounds",
            "2",
            "--output-dir",
            str(output),
            *extra,
        ]

    def test_fake_services_validate_singles_and_batches_and_record_shape(self):
        servers, threads = self._servers()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--old-metrics-url",
                    f"http://127.0.0.1:{servers[0].server_port}/metrics",
                    "--new-metrics-url",
                    f"http://127.0.0.1:{servers[1].server_port}/metrics",
                )
                self.assertEqual(main(args), 0)
                receipt_text = (output / "receipt.json").read_text()
                receipt = json.loads(receipt_text)
                metrics = [
                    json.loads(line)
                    for line in (output / "metrics.jsonl").read_text().splitlines()
                ]
                samples = [
                    json.loads(line)
                    for line in (output / "samples.jsonl").read_text().splitlines()
                ]
                workflows = [
                    json.loads(line)
                    for line in (output / "workflows.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(receipt["shapes"]), 4)
                self.assertEqual(receipt["audit"]["status"], "passed")
                self.assertTrue(receipt["audit"]["comparison_eligible"])
                audit_text = (output / "audit.jsonl").read_text()
                audits = [json.loads(line) for line in audit_text.splitlines()]
                self.assertEqual(len(audits), 4)
                self.assertTrue(all(item["passed"] for item in audits))
                self.assertEqual(
                    receipt["audit"]["provenance"]["old"]["model_revision"], "c" * 40
                )
                self.assertTrue(audits[0]["old_http"][0]["request_sha256"])
                self.assertTrue(audits[0]["new_http"]["response_sha256"])
                self.assertTrue(
                    all(
                        item["old_input_tokens_total"] == item["new_input_tokens_total"]
                        for item in audits
                    )
                )
                self.assertEqual(len(metrics), 16)
                self.assertTrue(all(item["error_code"] is None for item in metrics))
                self.assertTrue(all(sample["success"] for sample in samples))
                self.assertTrue(all(item["success"] for item in workflows))
                self.assertEqual({item["concurrency"] for item in samples}, {1, 4})
                self.assertEqual({item["concurrency"] for item in workflows}, {1, 4})
                self.assertEqual(
                    {path for path, _ in servers[0].bodies}, {"/v1/systemone"}
                )
                self.assertEqual(
                    {path for path, _ in servers[1].bodies},
                    {"/v1/systemone", "/v1/decision/batches"},
                )
                for row in receipt["shapes"]:
                    summary = row["summary"]
                    comparison = summary["comparison"]
                    self.assertTrue(comparison["eligible"])
                    self.assertGreater(
                        summary["arms"]["new"]["throughput"][
                            "successful_decisions_per_second"
                        ],
                        0,
                    )
                    self.assertIsNotNone(
                        summary["arms"]["old"]["latency"]["workflow_latency"]["p99_ms"]
                    )
                    self.assertEqual(
                        comparison["wire_bytes_identical"], row["state_count"] == 1
                    )
                    if row["state_count"] > 1:
                        self.assertEqual(
                            summary["arms"]["old"]["throughput"][
                                "successful_decisions"
                            ],
                            96,
                        )
                        self.assertEqual(
                            summary["arms"]["new"]["throughput"][
                                "successful_decisions"
                            ],
                            96,
                        )
                        self.assertEqual(
                            summary["arms"]["old"]["throughput"]["http_attempts"],
                            32,
                        )
                        self.assertEqual(
                            summary["arms"]["new"]["throughput"]["http_attempts"],
                            8,
                        )
                        telemetry = summary["telemetry"]
                        self.assertTrue(telemetry["comparison"]["available"])
                        self.assertEqual(
                            telemetry["arms"]["old"]["counter_deltas"][
                                "row_preparations"
                            ],
                            96,
                        )
                        self.assertEqual(
                            telemetry["arms"]["new"]["counter_deltas"][
                                "physical_batches"
                            ],
                            8,
                        )
                        self.assertEqual(
                            telemetry["arms"]["old"][
                                "observed_rows_per_physical_batch"
                            ],
                            3,
                        )
                        self.assertEqual(
                            telemetry["arms"]["new"][
                                "observed_rows_per_physical_batch"
                            ],
                            12,
                        )
                self.assertNotIn("127.0.0.1", receipt_text)
                self.assertNotIn("http://", receipt_text)
                self.assertNotIn("127.0.0.1", audit_text)
                self.assertNotIn("http://", audit_text)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_bad_batch_fails_complete_workflow(self):
        servers, threads = self._servers(invalid_batch=True)
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(servers, output, "--parity-policy", "report")
                args[args.index("--state-counts") + 1] = "4"
                args[args.index("--concurrencies") + 1] = "2"
                args[args.index("--warmup") + 1] = "0"
                args[args.index("--rounds") + 1] = "1"
                self.assertEqual(main(args), 1)
                receipt = json.loads((output / "receipt.json").read_text())
                row = receipt["shapes"][0]
                self.assertEqual(
                    row["summary"]["arms"]["new"]["throughput"]["successful_decisions"],
                    0,
                )
                self.assertEqual(
                    row["summary"]["arms"]["new"]["throughput"]["errors"],
                    {"response_contract": 4},
                )
                self.assertFalse(row["summary"]["comparison"]["eligible"])
                self.assertIn(
                    "exploratory_parity_report_mode",
                    row["summary"]["comparison"]["reasons"],
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_model_id_adapter_is_explicitly_nonidentical(self):
        servers, threads = self._servers()
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers, output, "--old-model-id", "legacy/decision"
                )
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                args[args.index("--warmup") + 1] = "0"
                args[args.index("--rounds") + 1] = "1"
                self.assertEqual(main(args), 0)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["adapter"]["kind"], "model_id_only")
                comparison = receipt["shapes"][0]["summary"]["comparison"]
                self.assertEqual(
                    comparison["type"], "single_request_model_id_adapter_workflow"
                )
                self.assertFalse(comparison["wire_bytes_identical"])
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_legacy_preview_adapter_is_recorded_and_runs_after_timing(self):
        servers, threads = self._servers()
        servers[0].legacy_preview = True
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--old-model-id",
                    "decision-nano-preview",
                    "--old-response-mode",
                    "legacy_preview",
                )
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                args[args.index("--warmup") + 1] = "0"
                args[args.index("--rounds") + 1] = "1"
                self.assertEqual(main(args), 0)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(
                    receipt["adapter"]["kind"], "legacy_preview_and_model_id"
                )
                self.assertTrue(receipt["adapter"]["applied_outside_timed_interval"])
                audit = json.loads((output / "audit.jsonl").read_text().splitlines()[0])
                self.assertEqual(audit["old_input_tokens_total"], 30)
                self.assertEqual(audit["new_input_tokens_total"], 30)
                self.assertEqual(
                    audit["states"][0]["old_answer_source"], "raw_legacy_preview"
                )
                self.assertEqual(
                    set(audit["states"][0]["old_question_input_tokens"].values()),
                    {10},
                )
                self.assertFalse(
                    receipt["shapes"][0]["summary"]["comparison"][
                        "wire_bytes_identical"
                    ]
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_token_mismatch_blocks_timing_and_reports_separately(self):
        servers, threads = self._servers()
        servers[1].token_delta = 1
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(servers, output)
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                self.assertEqual(main(args), 1)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["status"], "audit_failed")
                self.assertEqual(receipt["shapes"], [])
                self.assertFalse((output / "samples.jsonl").exists())
                self.assertEqual(
                    receipt["audit"]["mismatch_counts"], {"input_token_mismatch": 2}
                )
                audit = json.loads((output / "audit.jsonl").read_text().splitlines()[0])
                self.assertEqual(audit["input_token_total_delta"], 1)
                self.assertEqual(
                    audit["states"][0]["answers"][0]["absolute_probability_delta"], 0
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_probability_and_categorical_mismatch_block_timing(self):
        servers, threads = self._servers()
        servers[1].probability_shift = 0.02
        servers[1].choice_flip = True
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(servers, output)
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                self.assertEqual(main(args), 1)
                receipt = json.loads((output / "receipt.json").read_text())
                counts = receipt["audit"]["mismatch_counts"]
                self.assertGreater(counts["probability_tolerance"], 0)
                self.assertGreater(counts["categorical_outcome_mismatch"], 0)
                self.assertGreaterEqual(
                    receipt["audit"]["absolute_probability_delta"]["max"], 0.02
                )
                self.assertFalse((output / "samples.jsonl").exists())
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_noul_threshold_crossing_is_diagnostic_only(self):
        servers, threads = self._servers()
        servers[1].probability_shift = -0.004
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(servers, output)
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                args[args.index("--warmup") + 1] = "0"
                args[args.index("--rounds") + 1] = "1"
                self.assertEqual(main(args), 0)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["audit"]["status"], "passed")
                self.assertGreater(
                    receipt["audit"][
                        "noul_threshold_0_5_disagreements_diagnostic_only"
                    ],
                    0,
                )
                audit = json.loads((output / "audit.jsonl").read_text().splitlines()[0])
                self.assertTrue(
                    any(
                        answer.get("threshold_0_5_outcome_mismatch_diagnostic")
                        for state in audit["states"]
                        for answer in state["answers"]
                    )
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_small_probability_drift_passes_but_report_mode_has_no_ratios(self):
        servers, threads = self._servers()
        servers[1].probability_shift = 0.00424
        try:
            with TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = self._run_args(
                    servers,
                    output,
                    "--parity-policy",
                    "report",
                    "--old-metrics-url",
                    f"http://127.0.0.1:{servers[0].server_port}/metrics",
                    "--new-metrics-url",
                    f"http://127.0.0.1:{servers[1].server_port}/metrics",
                )
                args[args.index("--state-counts") + 1] = "1"
                args[args.index("--concurrencies") + 1] = "1"
                args[args.index("--warmup") + 1] = "0"
                args[args.index("--rounds") + 1] = "1"
                self.assertEqual(main(args), 0)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["audit"]["status"], "passed")
                self.assertAlmostEqual(
                    receipt["audit"]["absolute_probability_delta"]["max"], 0.00424
                )
                comparison = receipt["shapes"][0]["summary"]["comparison"]
                self.assertFalse(comparison["eligible"])
                self.assertIsNone(comparison["old_over_new_p50_workflow_latency"])
                self.assertIsNone(
                    comparison["new_over_old_successful_decisions_per_second"]
                )
                self.assertIn("exploratory_parity_report_mode", comparison["reasons"])
                telemetry = receipt["shapes"][0]["summary"]["telemetry"]["comparison"]
                self.assertFalse(telemetry["available"])
                self.assertIsNone(
                    telemetry["old_over_new_row_preparation_seconds_per_decision"]
                )
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_semantic_matrix_requires_six_receipts(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            build_semantic_matrix([])
