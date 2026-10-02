"""hub upload leaves exactly the package on the Hub (stdlib, fake API)."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from v2.release import hub, layout


class FakeApi:
    def __init__(self, private: bool = True):
        self.calls = []
        self.private = private

    def model_info(self, repo, revision=None):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(oid="a" * 40, commit_url=None)


class HubUploadTest(unittest.TestCase):
    def upload(self, kind: str, repo: str, api: FakeApi) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch) / "Decision-2.0-Kai-0.6B"
            package.mkdir()
            (package / "README.md").write_text("card\n")
            manifest = {
                "kind": kind,
                "model_name": "Decision-2.0-Kai-0.6B",
                "repo_id": repo,
                "files_sha256": {"README.md": layout.sha_file(package / "README.md")},
            }
            (package / layout.MANIFEST_NAME).write_text(json.dumps(manifest))
            args = argparse.Namespace(repo=repo, package=package, message="m")
            with mock.patch.object(hub, "_api", return_value=api):
                return hub.upload(args)

    def test_upload_deletes_remote_files_the_package_no_longer_has(self):
        api = FakeApi()
        result = self.upload("staging", "vllm-sr/dev2-release-staging", api)
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(api.calls[0]["delete_patterns"], ["*"])

    def test_staging_stays_private_and_releases_are_public(self):
        with self.assertRaises(RuntimeError):
            self.upload(
                "staging", "vllm-sr/dev2-release-staging", FakeApi(private=False)
            )
        release = "vllm-sr/Decision-2.0-Kai-0.6B"
        self.assertEqual(
            self.upload("release", release, FakeApi(private=False))["revision"],
            "a" * 40,
        )
        with self.assertRaises(RuntimeError):
            self.upload("release", release, FakeApi(private=True))


if __name__ == "__main__":
    unittest.main()
