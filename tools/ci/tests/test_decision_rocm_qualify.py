"""The ROCm receipt producer must derive pass claims from live observations."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_rocm_qualify as qualify
from decision_runtime.confidence import score_confidence

SOURCE = "a" * 40
ARTIFACT = "sha256:" + "b" * 64
MANIFEST = "c" * 64
CANDIDATE = (
    "ghcr.io/example/semantic-router/decision-runtime-rocm-staging@sha256:" + "d" * 64
)


def _answers(questions: dict) -> dict:
    answers = {}
    for key, question in questions.items():
        kind = question["type"]
        if kind == "noul":
            answers[key] = {"type": kind, "noul": 0.75}
        elif kind == "choice":
            answers[key] = {
                "type": kind,
                "choice": "billing",
                "confidence": 0.6,
                "probabilities": {"billing": 0.8, "technical": 0.2},
            }
        else:
            probabilities = [0.2, 0.3, 0.5]
            answers[key] = {
                "type": kind,
                "score": 1.3,
                "confidence": score_confidence(probabilities),
                "legend": {"0": "routine", "1": "important", "2": "urgent"},
                "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
            }
    return answers


class FakeRuntimeIO:
    """A strict fake service; no GPU, Docker, registry, or HF access."""

    def __init__(
        self,
        *,
        bad_status: bool = False,
        bad_row_limit: bool = False,
        bad_response: bool = False,
        missing_forward: bool = False,
        no_coalescing: bool = False,
    ):
        self.bad_status = bad_status
        self.bad_row_limit = bad_row_limit
        self.bad_response = bad_response
        self.missing_forward = missing_forward
        self.no_coalescing = no_coalescing
        self.commands: list[list[str]] = []
        self.paths: list[str] = []
        self.active_model: str | None = None
        self.active_instance: str | None = None
        self.rows = 0
        self.batches = 0
        self._lock = threading.Lock()

    def command(self, arguments: list[str], *, timeout: int) -> str:
        del timeout
        self.commands.append(arguments)
        if arguments[1:3] == ["drun", "run"]:
            assert self.active_model is None
            self.active_model = arguments[3]
            self.active_instance = arguments[arguments.index("--instance-name") + 1]
            self.rows = self.batches = 0
            port = arguments[arguments.index("--port") + 1]
            return (
                "Decision runtime ready\n"
                f"  Instance: {self.active_instance}\n"
                f"  Model: {self.active_model}@{SOURCE}\n"
                f"  Endpoint: http://127.0.0.1:{port}/v1/systemone\n"
                "  Backend: rocm (bfloat16)\n"
                f"  Artifact: {ARTIFACT}\n"
                "  Mode: detached\n"
            )
        assert arguments[1:3] == ["drun", "stop"]
        assert arguments[3] == self.active_instance
        self.active_model = self.active_instance = None
        return "Stopped Decision runtime instance.\n"

    def http(
        self, method: str, url: str, body: bytes | None, *, timeout: int
    ) -> qualify.HTTPObservation:
        del timeout
        assert self.active_model is not None
        path = url.split(":", 2)[-1].split("/", 1)[-1]
        with self._lock:
            self.paths.append("/" + path)
        if method == "GET":
            assert body is None
            if url.endswith("/ready"):
                response = {"ready": True}
            elif url.endswith("/api/status"):
                response = {
                    "status": "ready",
                    "models": [self.active_model],
                    "artifact": {
                        "model": self.active_model,
                        "revision": "e" * 40 if self.bad_status else SOURCE,
                        "manifest_sha256": MANIFEST,
                        "content_sha256": ARTIFACT.removeprefix("sha256:"),
                    },
                    "scheduler": [
                        {
                            "model": self.active_model,
                            "running": 0,
                            "queued": 0,
                            "max_concurrency": qualify.MAX_CONCURRENCY,
                            "max_queue": qualify.MAX_QUEUE,
                            "max_active_rows": (
                                1024 if self.bad_row_limit else qualify.MAX_ACTIVE_ROWS
                            ),
                        }
                    ],
                }
            else:
                assert url.endswith("/metrics")
                rows = 0 if self.missing_forward else self.rows
                batches = 0 if self.missing_forward else self.batches
                name = self.active_model
                metrics = (
                    f'decision_runtime_physical_batches_total{{model="{name}"}} '
                    f"{batches}\n"
                    f'decision_runtime_physical_batch_rows_total{{model="{name}"}} '
                    f"{rows}\n"
                    "decision_runtime_physical_batch_duration_seconds_total"
                    f'{{model="{name}"}} {rows * 0.0001:.9f}\n'
                )
                return qualify.HTTPObservation(200, metrics.encode(), 1.0)
            return qualify.HTTPObservation(200, json.dumps(response).encode(), 1.0)

        assert method == "POST" and body is not None
        request = json.loads(body)
        assert request["model"] == self.active_model
        questions = request["questions"]
        if url.endswith("/v1/systemone"):
            answers = _answers(questions)
            if self.bad_response:
                answers.pop(next(iter(answers)))
            response = {
                "model": self.active_model,
                "answers": answers,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
            decisions = len(questions)
        else:
            assert url.endswith("/v1/decision/batches")
            results = [
                {
                    "id": state["id"],
                    "answers": _answers(questions),
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
                for state in request["states"]
            ]
            response = {
                "model": self.active_model,
                "results": results,
                "usage": {"input_tokens": len(results), "output_tokens": 0},
            }
            decisions = len(results) * len(questions)
        with self._lock:
            self.rows += decisions
            self.batches += (
                decisions
                if self.no_coalescing
                else (decisions + qualify.PHYSICAL_BATCH - 1) // qualify.PHYSICAL_BATCH
            )
        return qualify.HTTPObservation(200, json.dumps(response).encode(), 4.0)


class ROCmQualificationTests(unittest.TestCase):
    def _options(self, root: Path) -> qualify.Options:
        return qualify.Options(
            owner="example",
            candidate_ref=CANDIDATE,
            output_dir=root / "qualification",
            port=43177,
            gpu_device=0,
        )

    def test_six_models_generate_hashed_evidence_from_actual_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = self._options(root)
            io = FakeRuntimeIO()
            inspected = []
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
            ):
                receipt_path = qualify.qualify_all(
                    options,
                    io=io,
                    inspect_candidate=lambda record: inspected.append(record.copy()),
                )
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(len(inspected), 1)
            self.assertEqual(len(receipt["models"]), 6)
            self.assertEqual(
                {row["id"] for row in receipt["models"]}, qualify.MODEL_IDS
            )
            self.assertEqual(len(io.commands), 12)
            self.assertIsNone(io.active_model)
            for command in io.commands[::2]:
                self.assertIn(CANDIDATE, command)
                self.assertEqual(command[command.index("--max-batch") + 1], "8")
                self.assertEqual(command[command.index("--gpu-device") + 1], "0")
            for row in receipt["models"]:
                evidence = json.loads(
                    (options.output_dir / row["evidence_file"]).read_text()
                )
                self.assertEqual(evidence["performance"]["decisions"], 64)
                self.assertEqual(evidence["performance"]["physical_rows"], 64)
                self.assertGreater(evidence["performance"]["p95_ms"], 0)
                self.assertTrue(all(evidence["checks"].values()))
                self.assertEqual(len(evidence["raw_sha256"]), 29)
                for relative, digest in evidence["raw_sha256"].items():
                    self.assertEqual(
                        qualify._digest((options.output_dir / relative).read_bytes()),
                        digest,
                    )
            self.assertIn("/v1/systemone", io.paths)
            self.assertIn("/v1/decision/batches", io.paths)

    def test_bad_live_identity_does_not_emit_promotable_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(bad_status=True)
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
                self.assertRaisesRegex(qualify.QualificationError, "live status"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertEqual([command[2] for command in io.commands], ["run", "stop"])

    def test_wrong_active_row_credit_limit_does_not_qualify(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(bad_row_limit=True)
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
                self.assertRaisesRegex(qualify.QualificationError, "live status"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertEqual([command[2] for command in io.commands], ["run", "stop"])

    def test_missing_physical_forwards_fails_even_when_http_answers_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(missing_forward=True)
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
                self.assertRaisesRegex(qualify.QualificationError, "physical forward"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertIsNone(io.active_model)

    def test_contract_failure_fails_closed_and_stops_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(bad_response=True)
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
                self.assertRaisesRegex(qualify.QualificationError, "request contract"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertEqual([command[2] for command in io.commands], ["run", "stop"])

    def test_singleton_only_forwards_do_not_qualify_physical_batching(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(no_coalescing=True)
            with (
                patch.object(qualify, "source_sha", return_value=SOURCE),
                patch.object(qualify, "_require_clean_source"),
                patch.object(qualify, "_require_free_port"),
                self.assertRaisesRegex(qualify.QualificationError, "physical forward"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertIsNone(io.active_model)

    def test_dirty_source_is_rejected_before_registry_or_runtime_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO()
            with (
                patch.object(
                    qualify.subprocess,
                    "check_output",
                    return_value=" M tools/ci/decision_rocm_qualify.py\n",
                ),
                self.assertRaisesRegex(qualify.QualificationError, "clean source"),
            ):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertFalse(options.output_dir.exists())
            self.assertEqual(io.commands, [])

    def test_mutable_candidate_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            options = replace(
                options, candidate_ref=CANDIDATE.rsplit("@", 1)[0] + ":latest"
            )
            io = FakeRuntimeIO()
            with self.assertRaisesRegex(qualify.QualificationError, "staging digest"):
                qualify.qualify_all(options, io=io, inspect_candidate=lambda _: None)
            self.assertEqual(io.commands, [])
            self.assertFalse(options.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
