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
    def __init__(self):
        self.calls = []

    def model_info(self, repo, revision=None):
        return SimpleNamespace(private=True)

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(oid="a" * 40, commit_url=None)


class HubUploadTest(unittest.TestCase):
    def test_upload_deletes_remote_files_the_package_no_longer_has(self):
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch) / "DEV2.0-0.6B"
            package.mkdir()
            (package / "README.md").write_text("card\n")
            manifest = {
                "kind": "staging",
                "model_name": "DEV2.0-0.6B",
                "repo_id": "llm-semantic-router/dev2-release-staging",
                "files_sha256": {"README.md": layout.sha_file(package / "README.md")},
            }
            (package / layout.MANIFEST_NAME).write_text(json.dumps(manifest))
            api = FakeApi()
            args = argparse.Namespace(
                repo="llm-semantic-router/dev2-release-staging",
                package=package,
                message="m",
            )
            with mock.patch.object(hub, "_api", return_value=api):
                result = hub.upload(args)
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(api.calls[0]["delete_patterns"], ["*"])


if __name__ == "__main__":
    unittest.main()
