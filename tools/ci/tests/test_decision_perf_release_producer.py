"""Protected paired evidence must bind live services to immutable images."""

from __future__ import annotations

import hashlib
import io
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


def _full_snapshot_fixture(
    root: Path, *, code_filename: str = "model.py"
) -> tuple[Path, Path, dict, SimpleNamespace]:
    source = root / "snapshot"
    core = root / "core"
    (source / "native").mkdir(parents=True)
    core.mkdir()
    weight = b"qualified model data"
    historical_code = b"raise RuntimeError('repository Python must not execute')\n"
    (source / "native" / "weights.bin").write_bytes(weight)
    (source / "native" / code_filename).write_bytes(historical_code)

    def identity(payload: bytes) -> dict:
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    manifest = {
        "files": {
            "weights.bin": identity(weight),
            code_filename: identity(historical_code),
        }
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (source / "native" / "MANIFEST.json").write_bytes(manifest_bytes)
    manifest_identity = identity(manifest_bytes)
    binding = {
        "repo": "example/model",
        "revision": "1" * 40,
        "manifest_file": "native/MANIFEST.json",
        "manifest_sha256": manifest_identity["sha256"],
        "files": {
            "native/MANIFEST.json": manifest_identity,
            f"native/{code_filename}": identity(historical_code),
            "native/weights.bin": identity(weight),
        },
    }
    (core / "BINDINGS.json").write_text(json.dumps({"old/model": binding}))
    receipt = {
        "schema_version": 2,
        "repository_id": binding["repo"],
        "revision": binding["revision"],
        "manifest": {
            "path": binding["manifest_file"],
            "sha256": manifest_identity["sha256"],
            "size_bytes": manifest_identity["bytes"],
        },
        "files": [
            {
                "path": "native/weights.bin",
                "sha256": identity(weight)["sha256"],
                "size_bytes": len(weight),
            }
        ],
    }
    content_id = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = SimpleNamespace(
        repository_id=binding["repo"],
        revision=binding["revision"],
        content_id=content_id,
        manifest=SimpleNamespace(
            path=binding["manifest_file"],
            sha256=manifest_identity["sha256"],
            size_bytes=manifest_identity["bytes"],
        ),
        files=(
            SimpleNamespace(
                repository_path="native/weights.bin",
                sha256=identity(weight)["sha256"],
                size_bytes=len(weight),
            ),
        ),
    )
    row = {
        "model_id": binding["repo"],
        "old_model_id": "old/model",
        "revision": binding["revision"],
        "artifact_content_id": content_id,
        "artifact_manifest_sha256": manifest_identity["sha256"],
    }
    return source, core, row, artifact


class DecisionPairedProducerTests(unittest.TestCase):
    def test_full_snapshot_accepts_renamed_model_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, core, row, artifact = _full_snapshot_fixture(
                Path(temporary), code_filename="modeling_decision.py"
            )
            with (
                patch.object(
                    producer, "resolve_decision_runtime_model", return_value=object()
                ),
                patch.object(producer, "open_verified_artifact", return_value=artifact),
            ):
                self.assertEqual(
                    producer._verified_old_snapshot(source, core, row),
                    row["artifact_content_id"],
                )

    def test_full_snapshot_selected_data_matches_qualified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, core, row, artifact = _full_snapshot_fixture(Path(temporary))
            with (
                patch.object(
                    producer, "resolve_decision_runtime_model", return_value=object()
                ),
                patch.object(producer, "open_verified_artifact", return_value=artifact),
            ):
                self.assertEqual(
                    producer._verified_old_snapshot(source, core, row),
                    row["artifact_content_id"],
                )
                (source / "native" / "weights.bin").write_bytes(b"substituted")
                with self.assertRaisesRegex(producer.ProducerError, "old binding"):
                    producer._verified_old_snapshot(source, core, row)

    def test_full_snapshot_rejects_extra_code_symlinks_and_revision_drift(self) -> None:
        for change, message in (
            ("extra", "file roster"),
            ("code", "old binding"),
            ("symlink", "symlink"),
            ("revision", "model revision"),
        ):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                source, core, row, artifact = _full_snapshot_fixture(Path(temporary))
                if change == "extra":
                    (source / "unexpected.py").write_text("pass\n")
                elif change == "code":
                    (source / "native" / "model.py").write_text("pass\n")
                elif change == "symlink":
                    (source / "alias").symlink_to(
                        source / "native", target_is_directory=True
                    )
                else:
                    bindings = json.loads((core / "BINDINGS.json").read_text())
                    bindings["old/model"]["revision"] = "2" * 40
                    (core / "BINDINGS.json").write_text(json.dumps(bindings))
                with (
                    patch.object(
                        producer,
                        "resolve_decision_runtime_model",
                        return_value=object(),
                    ),
                    patch.object(
                        producer, "open_verified_artifact", return_value=artifact
                    ),
                    self.assertRaisesRegex(producer.ProducerError, message),
                ):
                    producer._verified_old_snapshot(source, core, row)

    def test_full_snapshot_rejects_selected_data_or_manifest_identity_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, core, row, artifact = _full_snapshot_fixture(Path(temporary))
            artifact.files = (
                SimpleNamespace(
                    repository_path="native/weights.bin",
                    sha256="0" * 64,
                    size_bytes=artifact.files[0].size_bytes,
                ),
            )
            with (
                patch.object(
                    producer, "resolve_decision_runtime_model", return_value=object()
                ),
                patch.object(producer, "open_verified_artifact", return_value=artifact),
                self.assertRaisesRegex(producer.ProducerError, "old selected data"),
            ):
                producer._verified_old_snapshot(source, core, row)
        with tempfile.TemporaryDirectory() as temporary:
            source, core, row, artifact = _full_snapshot_fixture(Path(temporary))
            changed_code = b"raise RuntimeError('different inert code')\n"
            (source / "native" / "model.py").write_bytes(changed_code)
            bindings = json.loads((core / "BINDINGS.json").read_text())
            bindings["old/model"]["files"]["native/model.py"] = {
                "bytes": len(changed_code),
                "sha256": hashlib.sha256(changed_code).hexdigest(),
            }
            (core / "BINDINGS.json").write_text(json.dumps(bindings))
            with (
                patch.object(
                    producer, "resolve_decision_runtime_model", return_value=object()
                ),
                patch.object(producer, "open_verified_artifact", return_value=artifact),
                self.assertRaisesRegex(producer.ProducerError, "model-manifest file"),
            ):
                producer._verified_old_snapshot(source, core, row)
        with tempfile.TemporaryDirectory() as temporary:
            source, core, row, artifact = _full_snapshot_fixture(Path(temporary))
            (source / "native" / "MANIFEST.json").write_text('{"files":{}}')
            with (
                patch.object(
                    producer, "resolve_decision_runtime_model", return_value=object()
                ),
                patch.object(producer, "open_verified_artifact", return_value=artifact),
                self.assertRaisesRegex(producer.ProducerError, "manifest differs"),
            ):
                producer._verified_old_snapshot(source, core, row)

    def test_tree_hash_frames_paths_types_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = root / "one"
            two = root / "two"
            one.mkdir()
            two.mkdir()
            (one / "a").write_bytes(b"x")
            (one / "b").write_bytes(b"y")
            (two / "a").write_bytes(b"x\0F\0b\0y")

            def legacy_stream(tree: Path) -> bytes:
                return b"".join(
                    b"F\0" + path.name.encode() + b"\0" + path.read_bytes() + b"\0"
                    for path in sorted(tree.iterdir())
                )

            legacy_one = legacy_stream(one)
            legacy_two = legacy_stream(two)
            self.assertEqual(legacy_one, legacy_two)
            self.assertNotEqual(
                producer._mount_digest(one), producer._mount_digest(two)
            )
            alias = root / "alias"
            alias.symlink_to(one, target_is_directory=True)
            with self.assertRaisesRegex(producer.ProducerError, "not canonical"):
                producer._mount_digest(alias / "a")

    def test_bind_identity_matches_the_live_mounted_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter.py"
            source.write_text("adapter\n", encoding="utf-8")
            proc_root = root / "proc"
            mounted = proc_root / "123" / "root" / "adapter.py"
            mounted.parent.mkdir(parents=True)
            mounted.hardlink_to(source)
            container = {"State": {"Pid": 123}}

            producer._verify_bind_inode(
                container, source, "/adapter.py", proc_root=proc_root
            )
            mounted.unlink()
            mounted.write_text("different inode\n", encoding="utf-8")
            with self.assertRaisesRegex(producer.ProducerError, "live mounted inode"):
                producer._verify_bind_inode(
                    container, source, "/adapter.py", proc_root=proc_root
                )

    def test_old_baseline_v1_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(producer.ProducerError, "baseline schema"):
            producer._validate_config(
                {"schema_version": "decision-paired-baseline-v1"},
                candidate_ref=REF,
                qualification={},
            )

    def test_full_snapshot_config_requires_three_typed_mounts_and_exact_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            content_id = "f" * 64
            manifest_sha = "e" * 64
            revision = "d" * 40
            metadata = cache_root / "sha256" / content_id / ".vllm-sr-artifact.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps(
                    {
                        "repository_id": "example/model",
                        "revision": revision,
                        "manifest": {"sha256": manifest_sha},
                    }
                )
            )
            row = {
                "model_id": "example/model",
                "revision": revision,
                "artifact_content_id": content_id,
                "artifact_metadata_sha256": producer._digest(metadata),
                "artifact_manifest_sha256": manifest_sha,
                "old_core_source_sha256": "a" * 64,
                "old_adapter_source_sha256": "b" * 64,
                "old_physical_batch_size": 8,
                "new_physical_batch_size": 8,
                "old_arm_overlay": "none",
                "old": {
                    "url": "http://127.0.0.1:8123/v1/systemone",
                    "image_ref": IMAGE,
                    "container_id": CONTAINER,
                    "core_source_kind": "mounted_adapter",
                    "artifact_layout": producer.OLD_ARTIFACT_LAYOUT,
                    "artifact_mount_destination": "/snapshot",
                    "artifact_locator": {"kind": "argument", "flag": "--artifact-root"},
                    "adapter_mount_destination": "/adapter.py",
                    "adapter_locator": {"kind": "command_path", "path": "/adapter.py"},
                    "core_mount_destination": "/core",
                    "core_locator": {
                        "kind": "python_import",
                        "path": "/core/old_core.py",
                        "module": "old_core",
                    },
                    "api_container_port": 8000,
                    "mounts": [
                        {
                            "destination": "/adapter.py",
                            "sha256": "b" * 64,
                            "kind": "file",
                        },
                        {"destination": "/core", "sha256": "a" * 64, "kind": "tree"},
                        {
                            "destination": "/snapshot",
                            "sha256": "c" * 64,
                            "kind": "tree",
                        },
                    ],
                },
                "new": {
                    "url": "http://127.0.0.1:8234/v1/systemone",
                    "metrics_url": "http://127.0.0.1:8234/metrics",
                    "image_ref": REF,
                    "container_id": "d" * 64,
                },
            }
            config = {
                "schema_version": "decision-paired-baseline-v2",
                "hardware": "AMD Instinct MI300X",
                "gpu_device": "0",
                "gpu_exclusivity": "dedicated_gpu_no_unrelated_compute",
                "models": [row],
            }
            qualification = {
                "models": [
                    {
                        "id": row["model_id"],
                        "revision": revision,
                        "artifact_content_id": "sha256:" + content_id,
                    }
                ]
            }
            with (
                patch.object(producer, "MODEL_IDS", {row["model_id"]}),
                patch.object(
                    producer, "default_artifact_cache_root", return_value=cache_root
                ),
            ):
                self.assertEqual(
                    producer._validate_config(
                        config, candidate_ref=REF, qualification=qualification
                    ),
                    [row],
                )
                row["old"]["mounts"][0]["kind"] = "tree"
                with self.assertRaisesRegex(
                    producer.ProducerError, "adapter or imported core"
                ):
                    producer._validate_config(
                        config, candidate_ref=REF, qualification=qualification
                    )
                row["old"]["mounts"][0]["kind"] = "file"
                row["old"]["adapter_locator"]["path"] = "/adapter.py/other.py"
                with self.assertRaisesRegex(producer.ProducerError, "exact file mount"):
                    producer._validate_config(
                        config, candidate_ref=REF, qualification=qualification
                    )
                row["old"]["adapter_locator"]["path"] = "/adapter.py"
                row["old"]["mounts"].append(
                    {"destination": "/other", "sha256": "9" * 64, "kind": "tree"}
                )
                with self.assertRaisesRegex(
                    producer.ProducerError, "adapter or imported core"
                ):
                    producer._validate_config(
                        config, candidate_ref=REF, qualification=qualification
                    )

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
                "timed_semantic",
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

    def test_direct_mounted_core_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core.py"
            snapshot = root / "snapshot"
            core.write_text("old core\n")
            snapshot.mkdir()
            (snapshot / "weights.bin").write_bytes(b"weights")
            core_sha = producer._mount_digest(core)
            arm = {
                "image_ref": IMAGE,
                "container_id": CONTAINER,
                "url": "http://127.0.0.1:8123/v1/systemone",
                "core_source_kind": "mounted",
                "core_mount_destination": "/core.py",
                "artifact_mount_destination": "/snapshot",
                "mounts": [
                    {"destination": "/core.py", "sha256": core_sha, "kind": "file"},
                    {
                        "destination": "/snapshot",
                        "sha256": producer._mount_digest(snapshot),
                        "kind": "tree",
                    },
                ],
            }
            container = {
                "Id": CONTAINER,
                "Image": IMAGE,
                "Config": {"Env": ["ROCR_VISIBLE_DEVICES=0"]},
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
                        "Source": str(core),
                        "Destination": "/core.py",
                    },
                    {
                        "Type": "bind",
                        "RW": False,
                        "Source": str(snapshot),
                        "Destination": "/snapshot",
                    },
                ],
            }
            with (
                patch.object(
                    producer,
                    "_docker_inspect",
                    side_effect=lambda kind, identity: (
                        {"Id": IMAGE, "Config": {"Labels": {}}}
                        if kind == "image"
                        else container
                    ),
                ),
                self.assertRaisesRegex(producer.ProducerError, "direct old core"),
            ):
                producer._container_image(
                    arm,
                    source_sha=None,
                    gpu_device="0",
                    old_core_sha256=core_sha,
                    old_artifact={"model_id": "example/model"},
                )

    def test_imported_core_requires_a_separate_adapter_and_live_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter.py"
            core = root / "core"
            artifact = root / "artifact"
            adapter.write_text("adapter\n")
            core.mkdir()
            (core / "core.py").write_text("core\n")
            artifact.mkdir()
            (artifact / "weights.bin").write_bytes(b"weights")
            adapter_sha = producer._mount_digest(adapter)
            core_sha = producer._mount_digest(core)
            arm = {
                "image_ref": IMAGE,
                "container_id": CONTAINER,
                "url": "http://127.0.0.1:8123/v1/systemone",
                "api_container_port": 8000,
                "core_source_kind": "mounted_adapter",
                "artifact_layout": producer.OLD_ARTIFACT_LAYOUT,
                "adapter_mount_destination": "/adapter.py",
                "adapter_locator": {"kind": "command_path", "path": "/adapter.py"},
                "core_mount_destination": "/core",
                "core_locator": {
                    "kind": "python_import",
                    "path": "/core/core.py",
                    "module": "old_core",
                },
                "artifact_mount_destination": "/artifact",
                "artifact_locator": {"kind": "argument", "flag": "--artifact-root"},
                "mounts": [
                    {
                        "destination": "/adapter.py",
                        "sha256": adapter_sha,
                        "kind": "file",
                    },
                    {"destination": "/core", "sha256": core_sha, "kind": "tree"},
                    {
                        "destination": "/artifact",
                        "sha256": producer._mount_digest(artifact),
                        "kind": "tree",
                    },
                ],
            }
            row = {
                "model_id": "example/model",
                "revision": "1" * 40,
                "artifact_content_id": "f" * 64,
                "artifact_manifest_sha256": "e" * 64,
                "old_core_source_sha256": core_sha,
                "old_adapter_source_sha256": adapter_sha,
            }
            command = ["python", "/adapter.py", "--artifact-root", "/artifact"]
            container = {
                "Id": CONTAINER,
                "Image": IMAGE,
                "Config": {"Env": ["ROCR_VISIBLE_DEVICES=0"], "Cmd": command},
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
                        "Destination": dest,
                    }
                    for source, dest in (
                        (adapter, "/adapter.py"),
                        (core, "/core"),
                        (artifact, "/artifact"),
                    )
                ],
                "NetworkSettings": {
                    "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8123"}]}
                },
            }
            evidence: list[dict] = []
            snapshot_proofs: list[dict] = []
            with (
                patch.object(
                    producer,
                    "_docker_inspect",
                    side_effect=lambda kind, identity: (
                        {"Id": IMAGE, "Config": {"Labels": {}}}
                        if kind == "image"
                        else container
                    ),
                ),
                patch.object(producer, "_running_command", return_value=command),
                patch.object(producer, "_verify_bind_inode") as bind_identity,
                patch.object(
                    producer,
                    "_verified_old_snapshot",
                    return_value=row["artifact_content_id"],
                ),
                patch.object(
                    producer, "_old_live_attestation", return_value={"live": True}
                ) as live,
            ):
                self.assertEqual(
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                        old_attestations=evidence,
                        old_snapshot_proofs=snapshot_proofs,
                    ),
                    IMAGE,
                )
                self.assertEqual(evidence, [{"live": True}])
                self.assertEqual(bind_identity.call_count, 9)
                self.assertEqual(
                    snapshot_proofs,
                    [
                        {
                            "full_snapshot_sha256": producer._mount_digest(artifact),
                            "selected_data_content_id": row["artifact_content_id"],
                        }
                    ],
                )
                live.assert_called_once()
                arm["mounts"][0]["kind"] = "tree"
                with self.assertRaisesRegex(
                    producer.ProducerError, "mounted source kind"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
                arm["mounts"][0]["kind"] = "file"
                container["Mounts"][2]["Destination"] = "/core"
                with self.assertRaisesRegex(
                    producer.ProducerError, "unattested or writable mount"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
                container["Mounts"][2]["Destination"] = "/artifact"
                # Identical static mount digests cannot establish that the
                # running adapter imported the mounted core at all.
                live.side_effect = producer.ProducerError("core was not imported")
                with self.assertRaisesRegex(producer.ProducerError, "not imported"):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
                live.side_effect = None
                container["NetworkSettings"]["Ports"]["8000/tcp"] = [
                    {"HostIp": "127.0.0.1", "HostPort": "8124"}
                ]
                container["NetworkSettings"]["Ports"]["9000/tcp"] = [
                    {"HostIp": "127.0.0.1", "HostPort": "8123"}
                ]
                with self.assertRaisesRegex(
                    producer.ProducerError, "declared container port"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
                container["NetworkSettings"]["Ports"]["8000/tcp"] = [
                    {"HostIp": "127.0.0.1", "HostPort": "8123"}
                ]
                container["NetworkSettings"]["Ports"].pop("9000/tcp")
                container["Mounts"][0]["RW"] = True
                with self.assertRaisesRegex(
                    producer.ProducerError, "unattested or writable"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
                container["Mounts"][0]["RW"] = False
                (core / "core.py").write_text("substituted core\n")
                with self.assertRaisesRegex(
                    producer.ProducerError, "mounted source changed"
                ):
                    producer._container_image(
                        arm,
                        source_sha=None,
                        gpu_device="0",
                        old_core_sha256=core_sha,
                        old_adapter_sha256=adapter_sha,
                        old_artifact=row,
                    )
            with self.assertRaisesRegex(producer.ProducerError, "imported core path"):
                producer._import_locator(
                    {
                        "kind": "python_import",
                        "path": "/other/core.py",
                        "module": "old_core",
                    },
                    "/core",
                )

    def test_live_proof_rejects_stale_challenge_wrong_import_and_artifact(self) -> None:
        arm = {
            "url": "http://127.0.0.1:8123/v1/systemone",
            "adapter_locator": {"path": "/adapter.py"},
            "core_locator": {"module": "old_core", "path": "/core.py"},
            "artifact_mount_destination": "/artifact",
        }
        row = {
            "old_adapter_source_sha256": "a" * 64,
            "old_core_source_sha256": "b" * 64,
            "artifact_content_id": "c" * 64,
            "model_id": "example/model",
            "revision": "d" * 40,
        }
        expected = {
            "schema_version": producer.OLD_ATTESTATION_SCHEMA,
            "challenge": "e" * 64,
            "pid": 1,
            "process_start_ticks": 123456,
            "adapter_path": "/adapter.py",
            "adapter_sha256": row["old_adapter_source_sha256"],
            "imported_module": "old_core",
            "imported_core_path": "/core.py",
            "core_mount_sha256": row["old_core_source_sha256"],
            "loaded_artifact_root": "/artifact",
            "loaded_artifact_content_id": row["artifact_content_id"],
            "model_id": row["model_id"],
            "revision": row["revision"],
        }

        class AttestationResponse(io.BytesIO):
            def __init__(
                self,
                value: dict,
                *,
                status: int = 200,
                media_type: str = "application/json",
            ) -> None:
                super().__init__(json.dumps(value).encode())
                self.status = status
                self.headers = {"Content-Type": media_type}

        def serve(
            value: dict, *, status: int = 200, media_type: str = "application/json"
        ) -> SimpleNamespace:
            def open_proof(request, timeout):
                self.assertEqual(
                    request.full_url,
                    "http://127.0.0.1:8123/api/decision-baseline-attestation?challenge="
                    + "e" * 64,
                )
                self.assertEqual(timeout, 30)
                return AttestationResponse(value, status=status, media_type=media_type)

            return SimpleNamespace(open=open_proof)

        with (
            patch.object(producer, "_process_start_ticks", return_value=123456),
            patch.object(
                producer, "_running_command", return_value=["python", "/adapter.py"]
            ),
            patch.object(producer.secrets, "token_hex", return_value="e" * 64),
        ):
            for field, bad in (
                ("challenge", "f" * 64),
                ("imported_core_path", "/unrelated.py"),
                ("loaded_artifact_content_id", "f" * 64),
                ("process_start_ticks", 123457),
            ):
                with self.subTest(field=field):
                    with patch.object(
                        producer, "LOOPBACK_OPENER", serve({**expected, field: bad})
                    ):
                        with self.assertRaisesRegex(
                            producer.ProducerError, "identity differs"
                        ):
                            producer._old_live_attestation(
                                {"State": {"Pid": 12345}}, arm, row
                            )
            with (
                patch.object(
                    producer, "LOOPBACK_OPENER", serve({**expected, "pid": True})
                ),
                self.assertRaisesRegex(producer.ProducerError, "invalid type"),
            ):
                producer._old_live_attestation({"State": {"Pid": 12345}}, arm, row)
            with patch.object(producer, "LOOPBACK_OPENER", serve(expected)):
                archived = {
                    key: value
                    for key, value in expected.items()
                    if key
                    not in (
                        "adapter_path",
                        "imported_module",
                        "imported_core_path",
                        "loaded_artifact_root",
                    )
                }
                for field in (
                    "adapter_path",
                    "imported_module",
                    "imported_core_path",
                    "loaded_artifact_root",
                ):
                    archived[field + "_sha256"] = hashlib.sha256(
                        expected[field].encode()
                    ).hexdigest()
                self.assertEqual(
                    producer._old_live_attestation({"State": {"Pid": 12345}}, arm, row),
                    archived,
                )
            for status, media_type in ((302, "application/json"), (200, "text/plain")):
                with self.subTest(status=status, media_type=media_type):
                    with patch.object(
                        producer,
                        "LOOPBACK_OPENER",
                        serve(expected, status=status, media_type=media_type),
                    ):
                        with self.assertRaisesRegex(
                            producer.ProducerError, "JSON 200 response"
                        ):
                            producer._old_live_attestation(
                                {"State": {"Pid": 12345}}, arm, row
                            )
            with (
                patch.object(producer, "LOOPBACK_OPENER", serve(expected)),
                patch.object(
                    producer, "_process_start_ticks", side_effect=[123456, 123457]
                ),
                self.assertRaisesRegex(producer.ProducerError, "changed during"),
            ):
                producer._old_live_attestation({"State": {"Pid": 12345}}, arm, row)

    def test_live_process_identity_requires_container_pid_one(self) -> None:
        stat_line = "12345 (python) " + " ".join(["S"] + ["0"] * 18 + ["45678"])
        container = {"State": {"Pid": 12345}}
        with patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=lambda path, **kwargs: (
                stat_line if path.name == "stat" else "NSpid:\t12345\t1\n"
            ),
        ):
            self.assertEqual(producer._process_start_ticks(container), 45678)
        with (
            patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=lambda path, **kwargs: (
                    stat_line if path.name == "stat" else "NSpid:\t12345\n"
                ),
            ),
            self.assertRaisesRegex(producer.ProducerError, "not PID 1"),
        ):
            producer._process_start_ticks(container)


if __name__ == "__main__":
    unittest.main()
