"""Only the current plan's artifact names may supply qualification evidence."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

import check_ci_gate as gate  # noqa: E402
from test_ci_gate import completed  # noqa: E402
from verification_catalog import load_catalog  # noqa: E402

PATHS = [
    "tools/make/build-run-test.mk",
    "e2e/profiles/standalone/profile.go",
    "tools/ci/workflow_evidence.py",
]
BATCHES = (
    ("component_batches", "ci-result-components-"),
    ("platform_batches", "ci-result-"),
    ("e2e_batches", "ci-result-"),
)


def artifact_names(plan):
    names = {}
    for field, prefix in BATCHES:
        for batch in plan[field]:
            for record in batch["verifications"]:
                names[record["id"]] = prefix + batch["id"]
    for record in plan["verifications"]:
        names.setdefault(record["id"], "ci-result-" + record["id"])
    return names


def write_receipts(directory, plan, receipts):
    names = artifact_names(plan)
    for receipt in receipts:
        destination = directory / names[receipt["id"]] / f"{receipt['id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt))


def write_builds(directory, builds):
    for build in builds:
        name = build["id"].removeprefix("image:")
        destination = directory / f"ci-build-image-{name}" / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "images": [{"platform": "linux/amd64"}],
                    **build,
                    "id": name,
                }
            )
        )


class CIArtifactSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Components default to separate jobs. Exercise the supported shared
        # policy here so stale component batches remain a real regression case.
        with patch.dict(
            load_catalog()["batching"]["executor_defaults"]["tools"],
            {"batch_policy": "shared-components"},
        ):
            cls.old_plan, cls.old_receipts, cls.old_builds = completed(PATHS)
            cls.plan, cls.receipts, cls.builds = completed(PATHS, full=True)

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def populate_rerun(self):
        write_receipts(self.directory, self.old_plan, self.old_receipts)
        write_receipts(self.directory, self.plan, self.receipts)
        write_builds(self.directory, self.old_builds)
        write_builds(self.directory, self.builds)

    def verdict(self, plan=None):
        plan = self.plan if plan is None else plan
        return gate.evaluate_gate(
            plan,
            gate.load_receipts(self.directory, plan),
            builds=gate.load_builds(self.directory, plan["images"]),
        )

    def test_affected_to_full_rerun_ignores_old_component_platform_and_e2e_batches(
        self,
    ):
        self.populate_rerun()
        for field, prefix in BATCHES:
            with self.subTest(family=field):
                old = {prefix + batch["id"] for batch in self.old_plan[field]}
                current = {prefix + batch["id"] for batch in self.plan[field]}
                self.assertTrue(old - current, "fixture must retain an old batch")
                self.assertTrue(all((self.directory / name).is_dir() for name in old))
        selected = gate.load_receipts(self.directory, self.plan)
        self.assertCountEqual(
            [receipt["id"] for receipt in selected],
            self.plan["expected_verification_ids"],
        )
        verdict = self.verdict()
        self.assertTrue(verdict.passed, verdict.errors)

    def test_old_batch_cannot_replace_a_missing_receipt_in_the_current_batch(self):
        old_names = artifact_names(self.old_plan)
        current_names = artifact_names(self.plan)
        for field, _ in BATCHES:
            with self.subTest(family=field):
                self.populate_rerun()
                identity = self.old_plan[field][0]["verifications"][0]["id"]
                self.assertNotEqual(old_names[identity], current_names[identity])
                old_receipt = self.directory / old_names[identity] / f"{identity}.json"
                self.assertTrue(old_receipt.exists())
                current_receipt = (
                    self.directory / current_names[identity] / f"{identity}.json"
                )
                current_receipt.unlink()
                verdict = self.verdict()
                self.assertFalse(verdict.passed)
                self.assertIn(
                    f"required verification {identity}: missing", verdict.errors
                )

    def test_duplicate_receipts_inside_a_current_artifact_still_fail_the_gate(self):
        self.populate_rerun()
        receipt = self.receipts[0]
        name = artifact_names(self.plan)[receipt["id"]]
        (self.directory / name / "duplicate.json").write_text(json.dumps(receipt))
        verdict = self.verdict()
        self.assertFalse(verdict.passed)
        self.assertIn(f"duplicate result for {receipt['id']}", verdict.errors)

    def test_unselected_old_images_and_receipts_are_ignored_before_parsing(self):
        self.populate_rerun()
        plan, receipts, builds = completed(["README.md"])
        self.assertEqual(plan["images"], [])
        write_receipts(self.directory, plan, receipts)
        write_builds(self.directory, builds)
        old_image = self.directory / "ci-build-image-retired" / "manifest.json"
        old_image.parent.mkdir()
        old_image.write_text("not valid JSON")
        old_result = self.directory / "ci-result-retired" / "result.json"
        old_result.parent.mkdir()
        old_result.write_text("not valid JSON")
        self.assertEqual(gate.load_builds(self.directory, plan["images"]), [])
        verdict = self.verdict(plan)
        self.assertTrue(verdict.passed, verdict.errors)

    def test_missing_current_image_cannot_be_supplied_by_an_old_artifact_name(self):
        self.populate_rerun()
        image = self.plan["images"][0]
        current = self.directory / f"ci-build-image-{image}"
        old = self.directory / "ci-build-image-retired"
        shutil.copytree(current, old)
        shutil.rmtree(current)
        selected = gate.load_builds(self.directory, self.plan["images"])
        self.assertNotIn(f"image:{image}", [build["id"] for build in selected])
        verdict = self.verdict()
        self.assertFalse(verdict.passed)
        self.assertTrue(
            any(
                "build inventory mismatch" in error and f"image:{image}" in error
                for error in verdict.errors
            )
        )


if __name__ == "__main__":
    unittest.main()
