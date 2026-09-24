"""Release lock checks bind the wheel inventory to qualified OCI digests."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_image_lock_release as release
import decision_cpu_qualify as cpu
import decision_rocm_promotion as rocm

REVISION = "a" * 40
CPU_DIGEST = "sha256:" + "b" * 64
ROCM_DIGEST = "sha256:" + "c" * 64
ARTIFACT = "sha256:" + "d" * 64
OWNER = "example"
CPU_REF = f"ghcr.io/{OWNER}/semantic-router/decision-runtime-cpu@{CPU_DIGEST}"
ROCM_REF = f"ghcr.io/{OWNER}/semantic-router/decision-runtime-rocm@{ROCM_DIGEST}"
CANDIDATE = (
    f"ghcr.io/{OWNER}/semantic-router/decision-runtime-rocm-staging@{ROCM_DIGEST}"
)


def _fixture(directory: Path) -> tuple[Path, Path, Path]:
    digest_file = directory / "published-digest.txt"
    digest_file.write_text(CPU_DIGEST + "\n", encoding="ascii")
    cpu_base, _, _ = cpu.DECISION_RUNTIME_BASES[cpu.IMAGE]
    cpu_record = {
        "schema": 1,
        "image": cpu.IMAGE,
        "source_sha": REVISION,
        "base_image": cpu_base,
        "published_ref": CPU_REF,
        "backend": "cpu",
        "platform": "linux/amd64",
        "models": [],
    }
    for index, model_id in enumerate(sorted(cpu.MODEL_IDS)):
        slug = model_id.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
        hashes = {}
        for filename in cpu.RAW_FILES:
            relative = f"raw/{slug}/{filename}"
            file = directory / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                json.dumps(
                    {
                        "published_ref": CPU_REF,
                        "local_image_id": "sha256:" + "e" * 64,
                    }
                ).encode()
                if filename == "image-attestation.json"
                else f"{model_id}:{filename}".encode()
            )
            file.write_bytes(payload)
            hashes[relative] = cpu._digest(payload)
        evidence = {
            "image_ref": CPU_REF,
            "source_sha": REVISION,
            "model_id": model_id,
            "revision": f"{index}" * 40,
            "artifact_content_id": ARTIFACT,
            "backend": "cpu",
            "device": "cpu",
            "result": "passed",
            "checks": dict.fromkeys(cpu.REQUIRED_CHECKS, True),
            "raw_sha256": hashes,
        }
        relative = f"cpu-models/{slug}.json"
        payload = json.dumps(evidence, sort_keys=True).encode()
        file = directory / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(payload)
        cpu_record["models"].append(
            {
                "id": model_id,
                "revision": f"{index}" * 40,
                "artifact_content_id": ARTIFACT,
                "evidence_file": relative,
                "evidence_sha256": cpu._digest(payload),
            }
        )
    (directory / "cpu-qualification.json").write_text(json.dumps(cpu_record))
    lock_path = directory / "decision-images.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": REVISION,
                "images": {"cpu": CPU_REF, "rocm": ROCM_REF},
            }
        ),
        encoding="utf-8",
    )
    base, _, _ = rocm.DECISION_RUNTIME_BASES[rocm.IMAGE]
    receipt = {
        "schema": 1,
        "image": rocm.IMAGE,
        "source_sha": REVISION,
        "base_image": base,
        "candidate_ref": CANDIDATE,
        "backend": "rocm",
        "platform": "linux/amd64",
        "models": [],
    }
    for index, model_id in enumerate(sorted(rocm.MODEL_IDS)):
        revision = f"{index}" * 40
        slug = model_id.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
        raw_hashes = {}
        for filename in sorted(rocm.RAW_FILES):
            relative = f"raw/{slug}/{filename}"
            raw_path = directory / relative
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_bytes = f"{model_id}:{filename}".encode()
            raw_path.write_bytes(raw_bytes)
            raw_hashes[relative] = rocm._digest(raw_bytes)
        evidence = {
            "image_ref": CANDIDATE,
            "source_sha": REVISION,
            "model_id": model_id,
            "revision": revision,
            "artifact_content_id": ARTIFACT,
            "backend": "rocm",
            "device": "rocm",
            "result": "passed",
            "checks": dict.fromkeys(rocm.REQUIRED_CHECKS, True),
            "raw_sha256": raw_hashes,
        }
        filename = f"model-{index}.json"
        raw = json.dumps(evidence, sort_keys=True).encode()
        (directory / filename).write_bytes(raw)
        receipt["models"].append(
            {
                "id": model_id,
                "revision": revision,
                "artifact_content_id": ARTIFACT,
                "evidence_file": filename,
                "evidence_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }
        )
    receipt_path = directory / "qualification.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return lock_path, digest_file, receipt_path


class DecisionImageLockReleaseTests(unittest.TestCase):
    def _validate(self, lock: Path, digest: Path, receipt: Path):
        return release.validate_release_lock(
            lock,
            owner=OWNER,
            revision=REVISION,
            cpu_digest_file=digest,
            cpu_receipt=digest.parent / "cpu-qualification.json",
            rocm_receipt=receipt,
            rocm_published_ref=ROCM_REF,
        )

    def test_exact_six_model_receipt_and_published_digests_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            with (
                patch.object(release, "validate_registry_candidate") as staging,
                patch.object(release, "inspect_published_image") as published,
            ):
                inventory = self._validate(lock, digest, receipt)
            self.assertEqual(inventory.source_sha, REVISION)
            self.assertEqual(dict(inventory.images), {"cpu": CPU_REF, "rocm": ROCM_REF})
            staging.assert_called_once()
            self.assertEqual(published.call_count, 2)
            published.assert_any_call(CPU_REF, revision=REVISION, backend="cpu")
            published.assert_any_call(ROCM_REF, revision=REVISION, backend="rocm")

    def test_lock_and_digest_artifact_must_match_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "different source commit"):
                release.validate_release_lock(
                    lock,
                    owner=OWNER,
                    revision="e" * 40,
                    cpu_digest_file=digest,
                    cpu_receipt=digest.parent / "cpu-qualification.json",
                    rocm_receipt=receipt,
                    rocm_published_ref=ROCM_REF,
                )
            digest.write_text("sha256:" + "f" * 64)
            with self.assertRaisesRegex(ValueError, "CPU image differs"):
                self._validate(lock, digest, receipt)

    def test_cpu_default_requires_live_receipt_for_exact_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            cpu_receipt = digest.parent / "cpu-qualification.json"
            cpu_receipt.unlink()
            with self.assertRaises((OSError, ValueError)):
                self._validate(lock, digest, receipt)
            with self.assertRaises((OSError, ValueError)):
                release.generate_release_lock(
                    digest.parent / "unqualified-lock.json",
                    owner=OWNER,
                    revision=REVISION,
                    cpu_digest_file=digest,
                    cpu_receipt=cpu_receipt,
                    rocm_receipt=receipt,
                    rocm_published_ref=ROCM_REF,
                )
            self.assertFalse((digest.parent / "unqualified-lock.json").exists())
            _fixture(Path(temporary))
            document = json.loads(cpu_receipt.read_text())
            document["published_ref"] = CPU_REF.replace(CPU_DIGEST, ROCM_DIGEST)
            cpu_receipt.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "published image"):
                self._validate(lock, digest, receipt)

    def test_rocm_promotion_must_preserve_qualified_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "qualified digest"):
                release.validate_release_lock(
                    lock,
                    owner=OWNER,
                    revision=REVISION,
                    cpu_digest_file=digest,
                    cpu_receipt=digest.parent / "cpu-qualification.json",
                    rocm_receipt=receipt,
                    rocm_published_ref=ROCM_REF.replace(ROCM_DIGEST, CPU_DIGEST),
                )
            record = json.loads(receipt.read_text())
            receipt.write_text(json.dumps({**record, "models": record["models"][:-1]}))
            with self.assertRaisesRegex(ValueError, "exactly six"):
                self._validate(lock, digest, receipt)
            receipt.write_text(json.dumps(record))
            evidence = Path(temporary) / "model-0.json"
            evidence.write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence content changed"):
                self._validate(lock, digest, receipt)

    def test_lock_rejects_missing_backend_and_mutable_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            document = json.loads(lock.read_text())
            document["images"].pop("rocm")
            lock.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "needs CPU and ROCm"):
                self._validate(lock, digest, receipt)
            document["images"]["rocm"] = ROCM_REF.rsplit("@", 1)[0] + ":latest"
            lock.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "invalid rocm image"):
                self._validate(lock, digest, receipt)

    def test_registry_manifest_digest_and_labels_are_checked(self):
        manifest = b"OCI manifest bytes"
        reference = release._published_ref(
            OWNER, "cpu", "sha256:" + hashlib.sha256(manifest).hexdigest()
        )
        base, _, _ = release.DECISION_RUNTIME_BASES["decision-runtime-cpu"]
        config = {
            "os": "linux",
            "architecture": "amd64",
            "config": {
                "Labels": {
                    "org.opencontainers.image.base.name": base,
                    "org.opencontainers.image.revision": REVISION,
                    "ai.vllm-sr.decision.source-state": "clean",
                    "ai.vllm-sr.decision.backend": "cpu",
                }
            },
        }
        with patch.object(
            release.subprocess,
            "check_output",
            side_effect=[manifest, json.dumps(config).encode()],
        ):
            release.inspect_published_image(reference, revision=REVISION, backend="cpu")
        config["config"]["Labels"]["ai.vllm-sr.decision.backend"] = "rocm"
        with (
            patch.object(
                release.subprocess,
                "check_output",
                side_effect=[manifest, json.dumps(config).encode()],
            ),
            self.assertRaisesRegex(ValueError, "labels differ"),
        ):
            release.inspect_published_image(reference, revision=REVISION, backend="cpu")
        with (
            patch.object(release.subprocess, "check_output", return_value=b"wrong"),
            self.assertRaisesRegex(ValueError, "content differs"),
        ):
            release.inspect_published_image(reference, revision=REVISION, backend="cpu")

    def test_generate_is_deterministic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            lock.unlink()
            arguments = {
                "owner": OWNER,
                "revision": REVISION,
                "cpu_digest_file": digest,
                "cpu_receipt": digest.parent / "cpu-qualification.json",
                "rocm_receipt": receipt,
                "rocm_published_ref": ROCM_REF,
            }
            with (
                patch.object(release, "validate_registry_candidate"),
                patch.object(release, "inspect_published_image"),
            ):
                generated = release.generate_release_lock(lock, **arguments)
                first = lock.read_bytes()
                release.generate_release_lock(lock, **arguments)
                self.assertEqual(first, lock.read_bytes())
                self.assertEqual(
                    dict(generated.images), {"cpu": CPU_REF, "rocm": ROCM_REF}
                )
                expected = {
                    "schema_version": 1,
                    "source_sha": REVISION,
                    "images": {"cpu": CPU_REF, "rocm": ROCM_REF},
                }
                self.assertEqual(
                    first,
                    (
                        json.dumps(expected, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode(),
                )
                lock.write_bytes(b"existing unrelated file")
                with self.assertRaisesRegex(ValueError, "already differs"):
                    release.generate_release_lock(lock, **arguments)
                self.assertEqual(lock.read_bytes(), b"existing unrelated file")

    def test_generate_checks_evidence_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "qualified digest"):
                release.generate_release_lock(
                    lock,
                    owner=OWNER,
                    revision=REVISION,
                    cpu_digest_file=digest,
                    cpu_receipt=digest.parent / "cpu-qualification.json",
                    rocm_receipt=receipt,
                    rocm_published_ref=ROCM_REF.replace(ROCM_DIGEST, CPU_DIGEST),
                )
            self.assertFalse(lock.exists())
            with (
                patch.object(
                    release,
                    "validate_registry_candidate",
                    side_effect=ValueError("registry mismatch"),
                ),
                self.assertRaisesRegex(ValueError, "registry mismatch"),
            ):
                release.generate_release_lock(
                    lock,
                    owner=OWNER,
                    revision=REVISION,
                    cpu_digest_file=digest,
                    cpu_receipt=digest.parent / "cpu-qualification.json",
                    rocm_receipt=receipt,
                    rocm_published_ref=ROCM_REF,
                )
            self.assertFalse(lock.exists())

    def test_generate_command_uses_checkout_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock, digest, receipt = _fixture(Path(temporary))
            lock.unlink()
            arguments = [
                "decision_image_lock_release.py",
                "generate",
                "--output",
                str(lock),
                "--owner",
                OWNER,
                "--cpu-digest-file",
                str(digest),
                "--cpu-receipt",
                str(digest.parent / "cpu-qualification.json"),
                "--rocm-receipt",
                str(receipt),
                "--rocm-published-ref",
                ROCM_REF,
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(release, "source_sha", return_value=REVISION) as checkout,
                patch.object(release, "validate_registry_candidate"),
                patch.object(release, "inspect_published_image"),
            ):
                release.main()
            checkout.assert_called_once()
            self.assertEqual(
                dict(release._read_lock(lock).images),
                {"cpu": CPU_REF, "rocm": ROCM_REF},
            )

    def test_distribution_contains_exact_lock_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            lock, _, _ = _fixture(directory)
            wheel = directory / "candidate.whl"
            sdist = directory / "candidate.tar.gz"
            expected = lock.read_bytes()
            with zipfile.ZipFile(wheel, "w") as package:
                package.writestr(release._PACKAGE_LOCK_PATH, expected)
            with tarfile.open(sdist, "w:gz") as package:
                member = tarfile.TarInfo("vllm_sr-0.3.0/" + release._PACKAGE_LOCK_PATH)
                member.size = len(expected)
                package.addfile(member, io.BytesIO(expected))
            self.assertEqual(
                release.verify_distribution_lock(lock, wheel, sdist),
                "sha256:" + hashlib.sha256(expected).hexdigest(),
            )
            with zipfile.ZipFile(wheel, "w") as package:
                package.writestr(release._PACKAGE_LOCK_PATH, b"tampered")
            with self.assertRaisesRegex(ValueError, "wheel must contain"):
                release.verify_distribution_lock(lock, wheel, sdist)
            with zipfile.ZipFile(wheel, "w") as package:
                package.writestr(release._PACKAGE_LOCK_PATH, expected)
            with tarfile.open(sdist, "w:gz") as package:
                member = tarfile.TarInfo("vllm_sr-0.3.0/" + release._PACKAGE_LOCK_PATH)
                member.size = len(expected)
                package.addfile(member, io.BytesIO(b"X" + expected[1:]))
            with self.assertRaisesRegex(ValueError, "differs from release input"):
                release.verify_distribution_lock(lock, wheel, sdist)

    def test_package_declares_injected_lock_resource(self):
        package_root = Path(__file__).resolve().parents[3] / "src" / "vllm-sr"
        pyproject = package_root / "pyproject.toml"
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        self.assertIn(
            "decision-images.lock.json",
            metadata["tool"]["setuptools"]["package-data"]["cli.decision_runtime"],
        )
        self.assertIn(
            "include cli/decision_runtime/decision-images.lock.json",
            (package_root / "MANIFEST.in").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
