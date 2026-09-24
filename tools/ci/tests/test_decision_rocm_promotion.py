"""External ROCm promotion must reject an unqualified or mutable candidate."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_rocm_promotion as rocm

REVISION = "a" * 40
DIGEST = "sha256:" + "b" * 64
ARTIFACT = "sha256:" + "c" * 64
CANDIDATE = "ghcr.io/example/semantic-router/decision-runtime-rocm-staging@" + DIGEST


def receipt(directory: Path) -> Path:
    base, _, _ = rocm.DECISION_RUNTIME_BASES[rocm.IMAGE]
    record = {
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
            "revision": str(index) * 40,
            "artifact_content_id": ARTIFACT,
            "backend": "rocm",
            "device": "rocm",
            "result": "passed",
            "checks": dict.fromkeys(rocm.REQUIRED_CHECKS, True),
            "raw_sha256": raw_hashes,
        }
        filename = f"model-{index}.json"
        raw = json.dumps(evidence).encode()
        (directory / filename).write_bytes(raw)
        record["models"].append(
            {
                "id": model_id,
                "revision": evidence["revision"],
                "artifact_content_id": ARTIFACT,
                "evidence_file": filename,
                "evidence_sha256": rocm._digest(raw),
            }
        )
    path = directory / "qualification.json"
    path.write_text(json.dumps(record))
    return path


class ROCmPromotionTests(unittest.TestCase):
    def test_six_models_have_source_bound_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = receipt(Path(tmp))
            result = rocm.validate_receipt(path, owner="Example", revision=REVISION)
            self.assertEqual(len(result["models"]), 6)
            candidate = json.loads(path.read_text())
            candidate["models"].pop()
            path.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError, "exactly six"):
                rocm.validate_receipt(path, owner="Example", revision=REVISION)

    def test_bad_digest_tag_evidence_and_revision_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = receipt(Path(tmp))
            original = json.loads(path.read_text())
            for field, value in (
                ("candidate_ref", CANDIDATE.rsplit("@", 1)[0] + ":latest"),
                ("source_sha", "d" * 40),
                ("base_image", "python:latest"),
            ):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**original, field: value}))
                    with self.assertRaises(ValueError):
                        rocm.validate_receipt(path, owner="example", revision=REVISION)
            path.write_text(json.dumps(original))
            (Path(tmp) / original["models"][0]["evidence_file"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "content changed"):
                rocm.validate_receipt(path, owner="example", revision=REVISION)

    def test_raw_evidence_must_exist_match_hash_and_stay_in_receipt_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = receipt(directory)
            record = json.loads(path.read_text())
            first = directory / record["models"][0]["evidence_file"]
            evidence = json.loads(first.read_text())
            relative = next(iter(evidence["raw_sha256"]))
            raw_path = directory / relative
            original = raw_path.read_bytes()
            raw_path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "raw evidence content changed"):
                rocm.validate_receipt(path, owner="example", revision=REVISION)
            raw_path.write_bytes(original)
            raw_path.unlink()
            with self.assertRaisesRegex(ValueError, "raw evidence file is missing"):
                rocm.validate_receipt(path, owner="example", revision=REVISION)
            raw_path.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "raw evidence content changed"):
                rocm.validate_receipt(path, owner="example", revision=REVISION)

    def test_raw_evidence_inventory_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = receipt(directory)
            record = json.loads(path.read_text())
            first = directory / record["models"][0]["evidence_file"]
            evidence = json.loads(first.read_text())
            evidence["raw_sha256"].pop(next(iter(evidence["raw_sha256"])))
            updated = json.dumps(evidence).encode()
            first.write_bytes(updated)
            record["models"][0]["evidence_sha256"] = rocm._digest(updated)
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(
                ValueError, "raw evidence inventory is incomplete"
            ):
                rocm.validate_receipt(path, owner="example", revision=REVISION)

    def test_raw_evidence_symlink_cannot_escape_receipt_directory(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as external,
        ):
            directory = Path(tmp)
            path = receipt(directory)
            record = json.loads(path.read_text())
            first = directory / record["models"][0]["evidence_file"]
            evidence = json.loads(first.read_text())
            relative = next(iter(evidence["raw_sha256"]))
            raw_path = directory / relative
            external_file = Path(external) / "copied-raw.txt"
            external_file.write_bytes(raw_path.read_bytes())
            raw_path.unlink()
            raw_path.symlink_to(external_file)
            with self.assertRaisesRegex(ValueError, "escapes the receipt directory"):
                rocm.validate_receipt(path, owner="example", revision=REVISION)

    def test_registry_digest_and_source_labels_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = rocm.validate_receipt(
                receipt(Path(tmp)), owner="example", revision=REVISION
            )
            config = {
                "os": "linux",
                "architecture": "amd64",
                "config": {
                    "Labels": {
                        "org.opencontainers.image.base.name": record["base_image"],
                        "org.opencontainers.image.revision": REVISION,
                        "ai.vllm-sr.decision.source-state": "clean",
                        "ai.vllm-sr.decision.backend": "rocm",
                    }
                },
            }
            with (
                patch.object(
                    rocm.subprocess,
                    "check_output",
                    side_effect=[b"wrong manifest", json.dumps(config).encode()],
                ),
                self.assertRaisesRegex(ValueError, "registry content differs"),
            ):
                rocm.validate_registry_candidate(record)
            with (
                patch.object(rocm, "_digest", return_value=DIGEST),
                patch.object(
                    rocm.subprocess,
                    "check_output",
                    side_effect=[b"manifest", json.dumps(config).encode()],
                ),
            ):
                rocm.validate_registry_candidate(record)
            config["config"]["Labels"]["ai.vllm-sr.decision.backend"] = "cpu"
            with (
                patch.object(rocm, "_digest", return_value=DIGEST),
                patch.object(
                    rocm.subprocess,
                    "check_output",
                    side_effect=[b"manifest", json.dumps(config).encode()],
                ),
                self.assertRaisesRegex(ValueError, "configuration differs"),
            ):
                rocm.validate_registry_candidate(record)

    def test_promotion_requires_protected_exact_source_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = rocm.validate_receipt(
                receipt(Path(tmp)), owner="example", revision=REVISION
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "GITHUB_REF": "refs/heads/feature",
                        "GITHUB_REPOSITORY": "example/semantic-router",
                        "GITHUB_EVENT_NAME": "push",
                        "GITHUB_SHA": REVISION,
                    },
                ),
                self.assertRaisesRegex(ValueError, "protected exact-source push"),
            ):
                rocm.promote(record, owner="example")
            with (
                patch.dict(
                    os.environ,
                    {
                        "GITHUB_REF": "refs/tags/v1.2.3",
                        "GITHUB_REPOSITORY": "example/semantic-router",
                        "GITHUB_EVENT_NAME": "push",
                        "GITHUB_SHA": "d" * 40,
                    },
                ),
                self.assertRaisesRegex(ValueError, "protected exact-source push"),
            ):
                rocm.promote(record, owner="example")

            def copied(arguments, *, check):
                self.assertTrue(check)
                self.assertEqual(
                    arguments[0:4], ["skopeo", "copy", "--all", "--preserve-digests"]
                )
                self.assertEqual(arguments[-2], "docker://" + CANDIDATE)
                self.assertEqual(
                    arguments[-1],
                    "docker://ghcr.io/example/semantic-router/decision-runtime-rocm:"
                    + REVISION,
                )
                Path(arguments[arguments.index("--digestfile") + 1]).write_text(DIGEST)

            for ref in ("refs/heads/main", "refs/tags/v1.2.3"):
                with (
                    patch.dict(
                        os.environ,
                        {
                            "GITHUB_REF": ref,
                            "GITHUB_REPOSITORY": "example/semantic-router",
                            "GITHUB_EVENT_NAME": "push",
                            "GITHUB_SHA": REVISION,
                            "RUNNER_TEMP": tmp,
                        },
                    ),
                    patch.object(rocm.subprocess, "run", side_effect=copied),
                ):
                    self.assertEqual(
                        rocm.promote(record, owner="example"),
                        "ghcr.io/example/semantic-router/decision-runtime-rocm@"
                        + DIGEST,
                    )


if __name__ == "__main__":
    unittest.main()
