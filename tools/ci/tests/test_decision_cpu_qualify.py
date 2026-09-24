"""A CPU release default requires three live, exact-digest model probes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_cpu_qualify as cpu
from decision_rocm_qualify import HTTPObservation
from decision_runtime.confidence import score_confidence

SOURCE = "a" * 40
ARTIFACT = "sha256:" + "b" * 64
MANIFEST = "c" * 64
REF = "ghcr.io/example/semantic-router/decision-runtime-cpu@sha256:" + "d" * 64


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
    """No Docker, model download, GPU, listener, or network access."""

    def __init__(
        self,
        *,
        bad_status: bool = False,
        bad_response: bool = False,
        bad_image: bool = False,
        partial_launch: bool = False,
    ):
        self.bad_status = bad_status
        self.bad_response = bad_response
        self.bad_image = bad_image
        self.partial_launch = partial_launch
        self.commands: list[list[str]] = []
        self.paths: list[str] = []
        self.active_model: str | None = None
        self.active_instance: str | None = None

    def command(self, arguments: list[str], *, timeout: int) -> str:
        del timeout
        self.commands.append(arguments)
        if arguments[:3] == ["docker", "image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": "sha256:" + "e" * 64,
                        "RepoDigests": [REF],
                    }
                ]
            )
        if arguments[:2] == ["docker", "inspect"]:
            assert self.active_instance is not None
            assert arguments[2] == f"vllm-sr-drun-{self.active_instance}"
            return json.dumps(
                [
                    {
                        "Image": "sha256:" + ("f" if self.bad_image else "e") * 64,
                        "Config": {
                            "Image": REF,
                            "Cmd": [
                                "python",
                                "-m",
                                "decision_runtime.entrypoint",
                                "--backend",
                                "cpu",
                            ],
                            "Labels": {
                                "ai.vllm-sr.drun.managed": "true",
                                "ai.vllm-sr.drun.instance": self.active_instance,
                                "ai.vllm-sr.drun.image": REF,
                            },
                        },
                        "HostConfig": {"Devices": [], "DeviceRequests": None},
                        "State": {"Status": "running"},
                    }
                ]
            )
        if arguments[1:3] == ["drun", "run"]:
            assert self.active_model is None
            self.active_model = arguments[3]
            self.active_instance = arguments[arguments.index("--instance-name") + 1]
            port = arguments[arguments.index("--port") + 1]
            if self.partial_launch:
                raise cpu.QualificationError("simulated CLI timeout after launch")
            return (
                "Decision runtime ready\n"
                f"  Instance: {self.active_instance}\n"
                f"  Model: {self.active_model}@{SOURCE}\n"
                f"  Endpoint: http://127.0.0.1:{port}/v1/systemone\n"
                "  Backend: cpu (float32)\n"
                f"  Artifact: {ARTIFACT}\n"
                "  Mode: detached\n"
            )
        assert arguments[1:3] == ["drun", "stop"]
        assert arguments[3] == self.active_instance
        self.active_model = self.active_instance = None
        return "Stopped Decision runtime instance.\n"

    def http(
        self, method: str, url: str, body: bytes | None, *, timeout: int
    ) -> HTTPObservation:
        del timeout
        assert self.active_model is not None
        path = "/" + url.split(":", 2)[-1].split("/", 1)[-1]
        self.paths.append(path)
        if method == "GET":
            assert body is None
            if url.endswith("/ready"):
                response = {"ready": True}
            else:
                assert url.endswith("/api/status")
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
                            "max_concurrency": 8,
                            "max_queue": 32,
                            "max_active_rows": 4096,
                        }
                    ],
                }
            return HTTPObservation(200, json.dumps(response).encode(), 1.0)
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
        return HTTPObservation(200, json.dumps(response).encode(), 4.0)


class CPUQualificationTests(unittest.TestCase):
    def _options(self, root: Path) -> cpu.Options:
        return cpu.Options(
            owner="example",
            published_ref=REF,
            output_dir=root / "cpu-qualification",
            port=43177,
        )

    def _qualify(self, options: cpu.Options, io: FakeRuntimeIO) -> Path:
        with (
            patch.object(cpu, "source_sha", return_value=SOURCE),
            patch.object(cpu, "_require_clean_source"),
            patch.object(cpu, "_require_free_port"),
        ):
            return cpu.qualify_all(
                options,
                io=io,
                inspect_image=lambda reference, source: self.assertEqual(
                    (reference, source), (REF, SOURCE)
                ),
            )

    def test_three_live_models_and_exact_published_digest_are_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO()
            receipt_path = self._qualify(options, io)
            receipt = cpu.validate_receipt(
                receipt_path, owner="example", revision=SOURCE, published_image=REF
            )
            self.assertEqual({row["id"] for row in receipt["models"]}, cpu.MODEL_IDS)
            self.assertEqual(len(io.commands), 12)
            self.assertIsNone(io.active_model)
            for command in (cmd for cmd in io.commands if cmd[1:3] == ["drun", "run"]):
                self.assertEqual(command[command.index("--backend") + 1], "cpu")
                self.assertEqual(command[command.index("--image") + 1], REF)
                self.assertEqual(command[command.index("--runtime") + 1], "docker")
                self.assertNotIn("--gpu-device", command)
            self.assertIn("/v1/systemone", io.paths)
            self.assertIn("/v1/decision/batches", io.paths)
            for row in receipt["models"]:
                evidence = json.loads(
                    (options.output_dir / row["evidence_file"]).read_text()
                )
                self.assertEqual(len(evidence["raw_sha256"]), len(cpu.RAW_FILES))
                self.assertTrue(all(evidence["checks"].values()))

    def test_wrong_live_status_or_response_never_writes_release_receipt(self):
        for variant in ("bad_status", "bad_response"):
            with self.subTest(
                variant=variant
            ), tempfile.TemporaryDirectory() as temporary:
                options = self._options(Path(temporary))
                io = FakeRuntimeIO(**{variant: True})
                with self.assertRaises((ValueError, RuntimeError)):
                    self._qualify(options, io)
                self.assertFalse((options.output_dir / "qualification.json").exists())
                self.assertIsNone(io.active_model)
                self.assertEqual(io.commands[-1][1:3], ["drun", "stop"])

    def test_partial_launch_still_attempts_owned_stop_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(partial_launch=True)
            with self.assertRaisesRegex(cpu.QualificationError, "CLI timeout"):
                self._qualify(options, io)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertIsNone(io.active_instance)
            self.assertEqual(io.commands[-1][1:3], ["drun", "stop"])

    def test_live_container_must_run_the_pulled_digest_image_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            io = FakeRuntimeIO(bad_image=True)
            with self.assertRaisesRegex(cpu.QualificationError, "published digest"):
                self._qualify(options, io)
            self.assertFalse((options.output_dir / "qualification.json").exists())
            self.assertIsNone(io.active_instance)

    def test_receipt_rejects_missing_model_wrong_digest_and_tampered_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            path = self._qualify(options, FakeRuntimeIO())
            record = json.loads(path.read_text())
            with self.assertRaisesRegex(ValueError, "published image"):
                cpu.validate_receipt(
                    path,
                    owner="example",
                    revision=SOURCE,
                    published_image=REF.replace("d" * 64, "e" * 64),
                )
            with self.assertRaisesRegex(ValueError, "published image and source"):
                cpu.validate_receipt(
                    path, owner="example", revision="f" * 40, published_image=REF
                )
            path.write_text(json.dumps({**record, "models": record["models"][:-1]}))
            with self.assertRaisesRegex(ValueError, "exactly three"):
                cpu.validate_receipt(
                    path, owner="example", revision=SOURCE, published_image=REF
                )
            path.write_text(json.dumps(record))
            raw_name = next(
                iter(
                    json.loads(
                        (
                            options.output_dir / record["models"][0]["evidence_file"]
                        ).read_text()
                    )["raw_sha256"]
                )
            )
            (options.output_dir / raw_name).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "raw evidence changed"):
                cpu.validate_receipt(
                    path, owner="example", revision=SOURCE, published_image=REF
                )

    def test_missing_or_escaping_raw_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            path = self._qualify(options, FakeRuntimeIO())
            record = json.loads(path.read_text())
            evidence_path = options.output_dir / record["models"][0]["evidence_file"]
            evidence = json.loads(evidence_path.read_text())
            raw_name = next(iter(evidence["raw_sha256"]))
            raw_path = options.output_dir / raw_name
            raw_path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or escapes"):
                cpu.validate_receipt(
                    path, owner="example", revision=SOURCE, published_image=REF
                )
            raw_path.symlink_to(Path(temporary) / "outside")
            with self.assertRaisesRegex(ValueError, "missing or escapes"):
                cpu.validate_receipt(
                    path, owner="example", revision=SOURCE, published_image=REF
                )

    def test_raw_evidence_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self._options(Path(temporary))
            path = self._qualify(options, FakeRuntimeIO())
            record = json.loads(path.read_text())
            evidence = json.loads(
                (options.output_dir / record["models"][0]["evidence_file"]).read_text()
            )
            raw_name = next(iter(evidence["raw_sha256"]))
            (options.output_dir / raw_name).write_bytes(b"x" * (cpu.MAX_RAW_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size is invalid"):
                cpu.validate_receipt(
                    path, owner="example", revision=SOURCE, published_image=REF
                )


if __name__ == "__main__":
    unittest.main()
