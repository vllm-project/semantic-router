"""Protected paired evidence must bind live services to immutable images."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_perf_release_producer as producer

from tools.ci.tests.test_decision_perf_release_gate import _fixture

REF = "ghcr.io/example/decision@sha256:" + "a" * 64
IMAGE = "sha256:" + "b" * 64
CONTAINER = "c" * 64
SOURCE = "d" * 40


class DecisionPairedProducerTests(unittest.TestCase):
    def test_compacts_raw_shape_and_hashes_all_measured_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, report = _fixture(root / "fixture")
            model = report["models"][0]
            source_shape = model["shapes"][0]
            work = root / "work"
            measured = work / "measured"
            measured.mkdir(parents=True)
            raw_dir = root / "fixture" / Path(source_shape["raw_receipt_path"]).parent
            for name in producer.RAW_FILES:
                origin = raw_dir / name
                if origin.exists():
                    shutil.copyfile(origin, measured / name)
                else:
                    (measured / name).write_text("{}\n", encoding="utf-8")
            (work / "preflight-summary.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "mismatch_counts": {},
                        "absolute_probability_delta": {"max": 0.001},
                    }
                ),
                encoding="utf-8",
            )
            (work / "preflight-audit.jsonl").write_text("{}\n", encoding="utf-8")
            output = root / "output"
            with patch.object(
                producer, "_harness_digest", return_value=report["harness_sha256"]
            ):
                shape = producer._shape(
                    {"model_id": model["model_id"]},
                    32,
                    1,
                    work,
                    output,
                    report["source_sha"],
                )
            self.assertEqual(
                [cell["concurrency"] for cell in shape["cells"]], [1, 8, 32]
            )
            for name in (
                "receipt",
                "preflight",
                "preflight_audit",
                "audit",
                "samples",
                "workflows",
                "metrics",
            ):
                path = output / shape[f"raw_{name}_path"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    shape[f"raw_{name}_sha256"],
                )

    def test_only_exact_loopback_service_urls_are_accepted(self) -> None:
        self.assertEqual(
            producer._loopback(
                "http://127.0.0.1:8123/metrics", "metrics", path="/metrics"
            ),
            "http://127.0.0.1:8123/metrics",
        )
        for url in (
            "http://127.0.0.2:8123/metrics",
            "http://127.0.0.1:8123/metrics?token=secret",
            "http://example.invalid:8123/metrics",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                producer._loopback(url, "metrics", path="/metrics")

    def test_container_must_run_the_exact_digest_image_on_its_loopback_port(
        self,
    ) -> None:
        arm = {
            "image_ref": REF,
            "container_id": CONTAINER,
            "url": "http://127.0.0.1:8123/v1/systemone",
            "metrics_url": "http://127.0.0.1:8123/metrics",
        }
        image = {
            "Id": IMAGE,
            "Config": {
                "Labels": {
                    "ai.vllm-sr.decision.backend": "rocm",
                    "ai.vllm-sr.decision.source-state": "clean",
                    "org.opencontainers.image.revision": SOURCE,
                }
            },
        }
        container = {
            "Id": CONTAINER,
            "Image": IMAGE,
            "Config": {"Env": ["ROCR_VISIBLE_DEVICES=0"]},
            "State": {"Running": True},
            "HostConfig": {
                "Devices": [
                    {"PathOnHost": "/dev/kfd", "PathInContainer": "/dev/kfd"},
                    {"PathOnHost": "/dev/dri", "PathInContainer": "/dev/dri"},
                ]
            },
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8123"}]}
            },
        }

        def inspect(kind: str, identity: str) -> dict:
            self.assertEqual(identity, REF if kind == "image" else CONTAINER)
            return image if kind == "image" else container

        with patch.object(producer, "_docker_inspect", side_effect=inspect):
            self.assertEqual(
                producer._container_image(arm, source_sha=SOURCE, gpu_device="0"), IMAGE
            )
            arm["metrics_url"] = "http://127.0.0.1:9999/metrics"
            with self.assertRaisesRegex(
                producer.ProducerError, "candidate metrics must use the API listener"
            ):
                producer._container_image(arm, source_sha=SOURCE, gpu_device="0")
            # An old service may expose /metrics on a separate mapped port,
            # but that port must belong to the inspected old container.
            with self.assertRaisesRegex(
                producer.ProducerError, "metrics URL is not this container"
            ):
                producer._container_image(arm, source_sha=None, gpu_device="0")
            container["NetworkSettings"]["Ports"]["9000/tcp"] = [
                {"HostIp": "127.0.0.1", "HostPort": "9999"}
            ]
            self.assertEqual(
                producer._container_image(arm, source_sha=None, gpu_device="0"), IMAGE
            )
            arm["metrics_url"] = "http://127.0.0.1:8123/metrics"
            container["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"] = "8124"
            container["NetworkSettings"]["Ports"]["9000/tcp"] = [
                {"HostIp": "127.0.0.1", "HostPort": "8123"}
            ]
            with self.assertRaisesRegex(
                producer.ProducerError, "approved runtime listener"
            ):
                producer._container_image(arm, source_sha=SOURCE, gpu_device="0")
            container["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"] = "8123"
            container["Image"] = "sha256:" + "e" * 64
            with self.assertRaisesRegex(producer.ProducerError, "declared image"):
                producer._container_image(arm, source_sha=SOURCE, gpu_device="0")

    def test_candidate_image_source_label_must_match_current_commit(self) -> None:
        arm = {
            "image_ref": REF,
            "container_id": CONTAINER,
            "url": "http://127.0.0.1:8123/v1/systemone",
        }
        image = {
            "Id": IMAGE,
            "Config": {
                "Labels": {
                    "ai.vllm-sr.decision.backend": "rocm",
                    "ai.vllm-sr.decision.source-state": "clean",
                    "org.opencontainers.image.revision": "e" * 40,
                }
            },
        }
        with (
            patch.object(producer, "_docker_inspect", return_value=image),
            self.assertRaisesRegex(producer.ProducerError, "source or backend"),
        ):
            producer._container_image(arm, source_sha=SOURCE, gpu_device="0")

    def test_candidate_batch_size_is_taken_from_exact_running_launch(self) -> None:
        container = {"Config": {"Entrypoint": ["python"], "Cmd": ["--max-batch", "16"]}}
        self.assertEqual(producer._declared_launch_batch(container), 16)
        container["Config"]["Cmd"].extend(["--max-batch=8"])
        with self.assertRaisesRegex(producer.ProducerError, "ambiguous"):
            producer._declared_launch_batch(container)
        container["Config"]["Cmd"] = ["--max-batch", "untrusted"]
        with self.assertRaisesRegex(producer.ProducerError, "ambiguous"):
            producer._declared_launch_batch(container)

    def test_candidate_must_execute_the_approved_drun_process(self) -> None:
        model_id = "llm-semantic-router/Decision-1.0-Kai-0.6B"
        revision = "1" * 40
        artifact = "2" * 64
        command = [
            "/opt/vllm-sr/venvs/vela/bin/python",
            "-m",
            "decision_runtime.entrypoint",
            "--model",
            model_id,
            "--revision",
            revision,
            "--backend",
            "rocm",
            "--artifact-root",
            "/opt/vllm-sr/decision-artifact",
            "--artifact-content-id",
            artifact,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-batch",
            "8",
            "--max-concurrency",
            "4",
            "--max-queue",
            "32",
        ]
        image = {
            "Config": {
                "Entrypoint": [],
                "Env": ["TOKENIZERS_PARALLELISM=false"],
                "WorkingDir": "/opt/vllm-sr",
                "User": "root",
            }
        }
        container = {
            "Config": {
                "Entrypoint": [],
                "Cmd": command[:],
                "Env": ["TOKENIZERS_PARALLELISM=false", "ROCR_VISIBLE_DEVICES=0"],
                "WorkingDir": "/opt/vllm-sr",
                "User": "root",
            }
        }
        with (
            patch.object(
                producer,
                "resolve_decision_runtime_model",
                return_value=SimpleNamespace(profile=SimpleNamespace(family="vela")),
            ),
            patch.object(producer, "_running_command", return_value=command),
        ):
            producer._candidate_process(
                container,
                image,
                model_id=model_id,
                revision=revision,
                artifact_content_id=artifact,
                physical_batch_size=8,
                gpu_device="0",
            )
            container["Config"]["Cmd"] = [
                "python3",
                "-c",
                "print('unapproved')",
                "--max-batch",
                "8",
            ]
            with self.assertRaisesRegex(
                producer.ProducerError, "approved drun entrypoint"
            ):
                producer._candidate_process(
                    container,
                    image,
                    model_id=model_id,
                    revision=revision,
                    artifact_content_id=artifact,
                    physical_batch_size=8,
                    gpu_device="0",
                )

    def test_old_mounted_code_must_match_the_protected_core_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old-core.py"
            source.write_text("old code\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            artifact_source = Path(temporary) / "artifact"
            artifact_source.mkdir()
            (artifact_source / "weights.bin").write_bytes(b"weights")
            artifact_tree_digest = producer._mount_digest(artifact_source)
            content_id = "f" * 64
            manifest_sha = "e" * 64
            artifact_row = {
                "model_id": "example/model",
                "revision": "1" * 40,
                "artifact_content_id": content_id,
                "artifact_manifest_sha256": manifest_sha,
            }
            arm = {
                "image_ref": IMAGE,
                "container_id": CONTAINER,
                "url": "http://127.0.0.1:8123/v1/systemone",
                "mounts": [
                    {"destination": "/old-core.py", "sha256": digest},
                    {
                        "destination": "/artifact",
                        "sha256": artifact_tree_digest,
                    },
                ],
                "artifact_mount_destination": "/artifact",
                "artifact_locator": {"kind": "argument", "flag": "--model"},
                "core_source_kind": "mounted",
                "core_mount_destination": "/old-core.py",
                "core_locator": {"kind": "command_path", "path": "/old-core.py"},
            }
            image = {"Id": IMAGE, "Config": {"Labels": {}}}
            container = {
                "Id": CONTAINER,
                "Image": IMAGE,
                "Config": {
                    "Env": ["ROCR_VISIBLE_DEVICES=0"],
                    "Cmd": ["python", "/old-core.py", "--model", "/artifact"],
                },
                "State": {"Running": True, "Pid": 12345},
                "HostConfig": {
                    "Devices": [
                        {"PathOnHost": "/dev/kfd", "PathInContainer": "/dev/kfd"},
                        {"PathOnHost": "/dev/dri", "PathInContainer": "/dev/dri"},
                    ]
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "RW": False,
                        "Source": str(source),
                        "Destination": "/old-core.py",
                    },
                    {
                        "Type": "bind",
                        "RW": False,
                        "Source": str(artifact_source),
                        "Destination": "/artifact",
                    },
                ],
                "NetworkSettings": {
                    "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8123"}]}
                },
            }
            with patch.object(
                producer,
                "_docker_inspect",
                side_effect=lambda kind, identity: (
                    image if kind == "image" else container
                ),
            ), patch.object(
                producer, "resolve_decision_runtime_model", return_value=object()
            ), patch.object(
                producer,
                "_running_command",
                side_effect=lambda inspected: inspected["Config"]["Cmd"],
            ), patch.object(
                producer,
                "open_verified_artifact",
                return_value=SimpleNamespace(
                    repository_id=artifact_row["model_id"],
                    revision=artifact_row["revision"],
                    content_id=content_id,
                    manifest=SimpleNamespace(sha256=manifest_sha),
                ),
            ) as verified:
                self.assertEqual(
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    ),
                    IMAGE,
                )
                verified.assert_called_once()
                arm["core_source_kind"] = "baked"
                with self.assertRaisesRegex(
                    producer.ProducerError, "independent protected process/source"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )
                arm["core_source_kind"] = "mounted"
                with self.assertRaisesRegex(
                    producer.ProducerError, "running process script or executable"
                ):
                    producer._validate_locator(
                        {"kind": "environment", "name": "UNUSED_CORE"},
                        label="core",
                        destination="/old-core.py",
                    )
                container["Config"]["Cmd"] = [
                    "python",
                    "baked.py",
                    "--core",
                    "/old-core.py",
                    "--model",
                    "/artifact",
                ]
                with self.assertRaisesRegex(
                    producer.ProducerError, "not the executed process source"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )
                container["Config"]["Cmd"] = [
                    "python",
                    "/old-core.py",
                    "--model",
                    "/other",
                ]
                with self.assertRaisesRegex(
                    producer.ProducerError, "launch does not name"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )
                container["Config"]["Cmd"] = [
                    "python",
                    "other.py",
                    "--model",
                    "/artifact",
                ]
                with self.assertRaisesRegex(
                    producer.ProducerError, "not the executed process source"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )
                container["Config"]["Cmd"] = [
                    "python",
                    "/old-core.py",
                    "--model",
                    "/artifact",
                ]
                with patch.object(
                    producer,
                    "_running_command",
                    return_value=["python", "baked.py", "--model", "/artifact"],
                ), self.assertRaisesRegex(
                    producer.ProducerError, "running process differs"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )
                container["Config"]["Cmd"] = [
                    "python",
                    "/old-core.py",
                    "--model",
                    "/artifact",
                ]
                source.write_text("changed code\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    producer.ProducerError, "mounted source changed"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=digest,
                        old_artifact=artifact_row,
                    )


if __name__ == "__main__":
    unittest.main()
