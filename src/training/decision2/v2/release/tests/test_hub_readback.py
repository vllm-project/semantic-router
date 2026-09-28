"""hub readback collection expectations, including a new revision of a collected release (stdlib, fake API)."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from v2.release import hub, layout

REPO = "llm-semantic-router/DEV2.0-0.8B"
REVISION = "b" * 40
RELEASE = Path(__file__).resolve().parents[1] / "release.sh"


class FakeApi:
    def __init__(self, package: Path, collected: bool):
        self.package = package
        self.collected = collected

    def model_info(self, repo, revision=None, files_metadata=False):
        siblings = [
            SimpleNamespace(
                rfilename=name,
                lfs=None,
                blob_id=hub.git_blob_sha1(self.package / name),
            )
            for name in ("README.md", layout.MANIFEST_NAME)
        ]
        card = {"license": "apache-2.0", "base_model": "org/base"}
        return SimpleNamespace(
            private=True,
            sha=revision,
            siblings=siblings,
            card_data=SimpleNamespace(to_dict=lambda: card),
        )

    def get_collection(self, slug):
        items = (
            [SimpleNamespace(item_id=REPO, item_type="model")] if self.collected else []
        )
        return SimpleNamespace(private=True, title=hub.COLLECTION_TITLE, items=items)


class HubReadbackTest(unittest.TestCase):
    def readback(self, kind: str, collected: bool, **flags) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch) / "DEV2.0-0.8B"
            package.mkdir()
            (package / "README.md").write_text(
                "---\nlicense: apache-2.0\nbase_model: org/base\n---\n\n# card\n"
            )
            manifest = {
                "kind": kind,
                "model_name": "DEV2.0-0.8B",
                "repo_id": REPO,
                "files_sha256": {"README.md": layout.sha_file(package / "README.md")},
            }
            (package / layout.MANIFEST_NAME).write_text(json.dumps(manifest))
            args = argparse.Namespace(
                repo=REPO,
                revision=REVISION,
                package=package,
                collection=hub.COLLECTION,
                expect_collected=flags.get("expect_collected", False),
                already_collected=flags.get("already_collected", False),
            )
            with mock.patch.object(
                hub, "_api", return_value=FakeApi(package, collected)
            ), mock.patch("v2.release.card.check_rendered", return_value=[]):
                return hub.readback(args)

    def test_first_release_must_not_be_collected_before_its_seal(self):
        self.assertTrue(self.readback("release", collected=False)["passed"])
        self.assertFalse(self.readback("release", collected=True)["passed"])

    def test_new_revision_of_a_collected_release_expects_its_item(self):
        result = self.readback("release", collected=True, already_collected=True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["collection"]["expected_repo"])
        self.assertFalse(
            self.readback("release", collected=False, already_collected=True)["passed"]
        )

    def test_staging_is_never_expected_in_the_collection(self):
        self.assertFalse(
            self.readback("staging", collected=True, already_collected=True)["passed"]
        )
        self.assertFalse(
            self.readback("staging", collected=True, expect_collected=True)["passed"]
        )

    def test_launcher_accepts_already_collected_only_with_collect(self):
        run = subprocess.run(
            [
                "bash",
                str(RELEASE),
                "--spec",
                "s.json",
                "--src",
                "x",
                "--work",
                "/data/dev2/runs/release/test-never-created",
                "--upload",
                "--already-collected",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("--already-collected needs --collect", run.stderr)
        text = RELEASE.read_text(encoding="utf-8")
        self.assertIn(
            '"${readback_args[@]}" --output "$work/receipts/readback.json"', text
        )
        self.assertIn(
            '--expect-collected --output "$work/receipts/readback-collected.json"', text
        )


if __name__ == "__main__":
    unittest.main()
