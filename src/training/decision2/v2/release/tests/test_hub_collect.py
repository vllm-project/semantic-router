"""hub collect finds the Decision 2.0 collection by its pinned slug, whatever its title (stdlib, fake Hub)."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from v2.release import hub, layout

REPO = "vllm-sr/Decision-2.0-Vega-27B"
REVISION = "a" * 40


class FakeApi:
    def __init__(self, slug: str, title: str, collection_private: bool = False):
        self.slug, self.title, self.items = slug, title, []
        self.collection_private = collection_private

    def model_info(self, repo, revision=None):
        return SimpleNamespace(private=False, sha=revision)

    def get_collection(self, slug):
        return SimpleNamespace(
            slug=self.slug,
            title=self.title,
            private=self.collection_private,
            items=[SimpleNamespace(item_id=i, item_type="model") for i in self.items],
        )

    def add_collection_item(self, slug, item_id, item_type, exists_ok):
        if item_id not in self.items:
            self.items.append(item_id)


class HubCollectTest(unittest.TestCase):
    def collect(self, api: FakeApi, collection: str = hub.COLLECTION) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch) / "Decision-2.0-Vega-27B"
            package.mkdir()
            (package / layout.MANIFEST_NAME).write_text(json.dumps({"kind": "release"}))
            gate = Path(scratch) / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "schema": "dev2-release-gate/1",
                        "decision": "release",
                        "repo_id": REPO,
                        "revision": REVISION,
                        "manifest_sha256": layout.sha_file(
                            package / layout.MANIFEST_NAME
                        ),
                    }
                )
            )
            args = argparse.Namespace(
                repo=REPO,
                revision=REVISION,
                package=package,
                gate=gate,
                collection=collection,
            )
            with mock.patch.object(hub, "_api", return_value=api):
                return hub.collect(args)

    def test_a_renamed_collection_is_found_by_slug(self):
        result = self.collect(FakeApi(hub.COLLECTION, "🎲 Decision 2.0"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["collection"]["title"], "🎲 Decision 2.0")
        self.assertEqual(result["collection"]["items"], [REPO])

    def test_a_private_collection_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.collect(
                FakeApi(hub.COLLECTION, hub.COLLECTION_TITLE, collection_private=True)
            )

    def test_any_other_collection_is_refused(self):
        other = "vllm-sr/scratch-0123"
        with self.assertRaises(RuntimeError):
            self.collect(FakeApi(other, hub.COLLECTION_TITLE), collection=other)


if __name__ == "__main__":
    unittest.main()
