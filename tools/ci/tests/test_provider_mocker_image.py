"""Fixture publications are keyed by build inputs, independent of product commits."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provider_mocker_image as mocker
from ci_plan import digest, github_outputs, make_plan, resolve_image_sources
from image_artifacts import publication_tags


class FakeRegistry:
    def __init__(self, fingerprint):
        self.documents = {}
        config = {
            "os": "linux",
            "architecture": "amd64",
            "config": {
                "Labels": {
                    mocker.INPUT_LABEL: fingerprint,
                    mocker.CHANNEL_LABEL: "main",
                    "org.opencontainers.image.revision": "a" * 40,
                    "org.opencontainers.image.source": "https://github.com/"
                    + mocker.REPOSITORY,
                }
            },
        }
        self.config_digest = self.add("blobs", config)
        manifest = {"config": {"digest": self.config_digest}, "layers": []}
        native = self.add("manifests", manifest)
        index = {"manifests": [{"digest": native}]}
        self.digest = self.add("manifests", index)
        self.documents["manifests", mocker.input_tag(fingerprint)] = (
            index,
            self.digest,
        )

    def add(self, kind, value):
        digest = "sha256:" + hashlib.sha256(json.dumps(value).encode()).hexdigest()
        self.documents[kind, digest] = (value, digest)
        return digest

    def document(self, kind, reference):
        return self.documents[kind, reference]


class FixturePublicationTests(unittest.TestCase):
    def test_only_actual_build_inputs_invalidate_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / mocker.CONTEXT
            (context / "provider_mocker").mkdir(parents=True)
            paths = [
                context / ".dockerignore",
                context / "Dockerfile",
                context / "requirements.txt",
                context / "provider_mocker/app.py",
            ]
            for path in paths:
                path.write_text("original")
            baseline = mocker.input_fingerprint(root)
            for filename in (
                "README.md",
                "tests/test_app.py",
                "provider_mocker/__pycache__/app.pyc",
            ):
                path = context / filename
                path.parent.mkdir(exist_ok=True)
                path.write_text("does not enter image")
                self.assertEqual(mocker.input_fingerprint(root), baseline)
                self.assertEqual(
                    mocker.acquisition([str(path.relative_to(root))], root)["source"],
                    "published",
                )
            for path in paths:
                path.write_text("changed")
                self.assertNotEqual(mocker.input_fingerprint(root), baseline)
                self.assertEqual(
                    mocker.acquisition([str(path.relative_to(root))], root)["source"],
                    "candidate",
                )
                path.write_text("original")
            paths[-1].chmod(0o755)
            self.assertNotEqual(mocker.input_fingerprint(root), baseline)

    def test_registry_checks_input_channel_and_original_source(self):
        record = {"inputs_sha256": "b" * 64, "id": mocker.IMAGE, "source": "published"}
        registry = FakeRegistry(record["inputs_sha256"])
        result = mocker.resolve_published(record, registry)
        self.assertEqual(result["ref"], mocker.REGISTRY + "@" + registry.digest)
        self.assertEqual(result["image_source_sha"], "a" * 40)
        mocker.validate_acquisition(result, result)
        for field in (
            mocker.INPUT_LABEL,
            mocker.CHANNEL_LABEL,
            "org.opencontainers.image.revision",
            "org.opencontainers.image.source",
        ):
            bad = copy.deepcopy(registry)
            bad.documents["blobs", bad.config_digest][0]["config"]["Labels"][
                field
            ] = "wrong"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "identity"),
            ):
                mocker.resolve_published(record, bad)
        for field in (
            "ref",
            "registry_digest",
            "image_source_sha",
            "inputs_sha256",
            "images",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                mocker.validate_acquisition({**result, field: "wrong"}, result)

    def test_only_missing_tag_manifest_is_unavailable(self):
        fingerprint = "b" * 64
        record = {"inputs_sha256": fingerprint}
        registry = FakeRegistry(fingerprint)
        tag = mocker.input_tag(fingerprint)
        missing = HTTPError("url", 404, "missing", {}, None)
        with patch.object(registry, "document", side_effect=missing) as lookup:
            with self.assertRaises(mocker.PublicationUnavailableError):
                mocker.resolve_published(record, registry)
            lookup.assert_called_once_with("manifests", tag)

        for code in (401, 403, 404):
            with (
                patch.object(
                    mocker,
                    "RegistryClient",
                    side_effect=HTTPError("url", code, "missing", {}, None),
                ),
                self.assertRaises(HTTPError),
            ):
                mocker.resolve_published(record)
        for error in (
            HTTPError("url", 401, "unauthorized", {}, None),
            HTTPError("url", 403, "forbidden", {}, None),
            URLError("network unavailable"),
        ):
            with (
                patch.object(registry, "document", side_effect=error),
                self.assertRaises(type(error)),
            ):
                mocker.resolve_published(record, registry)

        original_document = registry.document

        def missing_child(kind, reference):
            if reference == tag:
                return original_document(kind, reference)
            raise missing

        with (
            patch.object(registry, "document", side_effect=missing_child),
            self.assertRaises(HTTPError),
        ):
            mocker.resolve_published(record, registry)

    def test_missing_fixture_tag_builds_exact_candidate_and_main_publishes(self):
        for profile in ("pr", "main", "nightly", "release"):
            with self.subTest(profile=profile):
                plan = make_plan(
                    ["e2e/testing/run_memory_integration.sh"],
                    source_sha="c" * 40,
                    profile=profile,
                )
                original = copy.deepcopy(plan)
                with patch(
                    "ci_plan.resolve_published",
                    side_effect=mocker.PublicationUnavailableError("tag missing"),
                ):
                    resolved = resolve_image_sources(plan)
                self.assertEqual(plan, original)
                self.assertEqual(resolved["paths"], original["paths"])
                self.assertEqual(resolved["verifications"], original["verifications"])
                self.assertEqual(
                    resolved["image_sources"][mocker.IMAGE],
                    {
                        **original["image_sources"][mocker.IMAGE],
                        "source": "candidate",
                    },
                )
                self.assertIn(mocker.IMAGE, resolved["build_images"])
                self.assertEqual(
                    mocker.IMAGE in resolved["publish_images"], profile == "main"
                )
                self.assertEqual(
                    resolved["multiarch"], bool(resolved["publish_images"])
                )
                self.assertEqual(
                    resolved["plan_sha256"],
                    digest(
                        {
                            key: value
                            for key, value in resolved.items()
                            if key != "plan_sha256"
                        }
                    ),
                )
                outputs = {
                    key: json.loads(value)
                    for key, value in github_outputs(resolved).items()
                }
                fixture = outputs["image_producers"]["image-fixtures"]
                self.assertEqual(fixture["build_images"], [mocker.IMAGE])
                self.assertEqual(fixture["published_images"], [])

    def test_valid_publication_is_reused_and_invalid_one_still_fails(self):
        plan = make_plan(
            ["e2e/testing/run_memory_integration.sh"],
            source_sha="c" * 40,
        )
        original = copy.deepcopy(plan)
        fingerprint = plan["image_sources"][mocker.IMAGE]["inputs_sha256"]
        published = mocker.resolve_published(
            plan["image_sources"][mocker.IMAGE], FakeRegistry(fingerprint)
        )
        with patch("ci_plan.resolve_published", return_value=published):
            resolved = resolve_image_sources(plan)
        self.assertEqual(plan, original)
        self.assertEqual(resolved["image_sources"][mocker.IMAGE], published)
        self.assertNotIn(mocker.IMAGE, resolved["build_images"])
        self.assertEqual(
            json.loads(github_outputs(resolved)["published_images"]), [published]
        )
        with (
            patch("ci_plan.resolve_published", side_effect=ValueError("bad labels")),
            self.assertRaisesRegex(ValueError, "bad labels"),
        ):
            resolve_image_sources(plan)
        self.assertEqual(plan, original)

    def test_fixture_is_reused_for_all_unrelated_ci_entrypoints(self):
        for profile, full in (
            ("pr", False),
            ("pr", True),
            ("main", False),
            ("nightly", False),
            ("release", False),
        ):
            plan = make_plan(
                ["e2e/testing/run_memory_integration.sh"],
                source_sha="c" * 40,
                profile=profile,
                full=full,
            )
            with self.subTest(profile=profile, full=full):
                self.assertIn(mocker.IMAGE, plan["images"])
                self.assertNotIn(mocker.IMAGE, plan["build_images"])
                self.assertNotIn(mocker.IMAGE, plan["publish_images"])
                self.assertEqual(
                    plan["image_sources"][mocker.IMAGE]["source"], "published"
                )

    def test_scheduled_inventory_is_not_a_change_to_fixture_inputs(self):
        # Scheduled events have no base SHA: git_changed_files returns all tracked
        # paths to qualify the product. That inventory cannot invalidate a fixture.
        for profile in ("nightly", "release"):
            plan = make_plan(
                [
                    mocker.CONTEXT + "/Dockerfile",
                    mocker.CONTEXT + "/provider_mocker/app.py",
                ],
                source_sha="c" * 40,
                profile=profile,
            )
            self.assertNotIn(mocker.IMAGE, plan["build_images"])
            self.assertEqual(plan["image_sources"][mocker.IMAGE]["source"], "published")

    def test_changed_fixture_is_qualified_in_pr_and_published_only_on_main(self):
        for profile in ("pr", "main"):
            plan = make_plan(
                [mocker.CONTEXT + "/provider_mocker/app.py"],
                source_sha="c" * 40,
                profile=profile,
            )
            self.assertIn(mocker.IMAGE, plan["build_images"])
            self.assertEqual(mocker.IMAGE in plan["publish_images"], profile == "main")
            self.assertEqual(plan["image_sources"][mocker.IMAGE]["source"], "candidate")
        with patch.object(mocker, "input_fingerprint", return_value="d" * 64):
            self.assertEqual(
                publication_tags(mocker.IMAGE, "main", "", False, ""),
                ["inputs-" + "d" * 64],
            )
        for profile in ("pr", "release", "nightly"):
            with self.assertRaises(ValueError):
                publication_tags(mocker.IMAGE, profile, "", False, "")


if __name__ == "__main__":
    unittest.main()
