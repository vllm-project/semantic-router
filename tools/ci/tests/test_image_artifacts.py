import copy
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_artifacts as images
from check_ci_gate import load_builds


def archive_image(path, architectures=("amd64",)):
    contents, descriptors = {}, []

    def blob(value):
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        contents["blobs/sha256/" + digest] = raw
        return {"digest": "sha256:" + digest, "size": len(raw)}

    for arch in architectures:
        config = blob(
            {
                "os": "linux",
                "architecture": arch,
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "a" * 64]},
                "config": {"Cmd": ["router"], "Labels": {"empty-label": ""}},
            }
        )
        descriptors.append(blob({"config": config, "layers": []}))
    contents["index.json"] = json.dumps({"manifests": descriptors}).encode()
    with tarfile.open(path, "w") as archive:
        for name, content in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


class ImageArtifactTests(unittest.TestCase):
    def test_multiarchitecture_inventory_is_actual_oci_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.tar"
            archive_image(path, ("arm64", "amd64"))
            self.assertEqual(
                [item["platform"] for item in images.oci_images(path)],
                ["linux/amd64", "linux/arm64"],
            )

    def test_duplicate_platform_cannot_qualify_an_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.tar"
            archive_image(path, ("amd64", "amd64"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                images.oci_images(path)

    def test_identity_rejects_stale_source_and_changed_archive(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(images, "source_sha", return_value="a" * 40),
        ):
            directory = Path(tmp)
            archive_image(directory / "image.tar")
            manifest = {
                "source_sha": "a" * 40,
                "id": "vllm-sr",
                "context": ".",
                "dockerfile": "src/vllm-sr/Dockerfile",
                "sha256": images.sha256(directory / "image.tar"),
                "images": images.oci_images(directory / "image.tar"),
            }
            receipt = directory / "manifest.json"
            receipt.write_text(json.dumps(manifest))
            images.verify(directory, "vllm-sr")
            manifest["source_sha"] = "b" * 40
            receipt.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source revision"):
                images.verify(directory, "vllm-sr")
            manifest["source_sha"] = "a" * 40
            receipt.write_text(json.dumps(manifest))
            with (directory / "image.tar").open("ab") as archive:
                archive.write(b"changed")
            with self.assertRaisesRegex(ValueError, "content differs"):
                images.verify(directory, "vllm-sr")

    def test_pr_cannot_be_promoted_and_tags_remain_compatible(self):
        with self.assertRaisesRegex(ValueError, "cannot be published"):
            images.publication_tags("vllm-sr", "pr", "", False, "20260917")
        for mode in ("pr", "nightly", "release"):
            with self.assertRaises(ValueError):
                images.publication_tags("provider-mocker", mode, "", False, "20260917")
        self.assertEqual(
            images.publication_tags("vllm-sr", "release", "v1.2.3", True, ""),
            ["v1.2.3", "latest"],
        )

    def test_promotion_records_the_copied_digest_for_each_tag(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                sys,
                "argv",
                [
                    "image_artifacts.py",
                    "promote",
                    "--image",
                    "dashboard",
                    "--directory",
                    tmp,
                    "--mode",
                    "release",
                    "--tag",
                    "v1.2.3",
                    "--latest",
                ],
            ),
            patch.dict(images.os.environ, {"GITHUB_REPOSITORY_OWNER": "Example"}),
            patch.object(
                images,
                "verify",
                return_value={
                    "mode": "release",
                    "tag": "v1.2.3",
                    "date": "",
                },
            ),
            patch.object(images.subprocess, "run") as run,
        ):
            images.main()
            self.assertEqual(run.call_count, 2)
            for call, tag in zip(run.call_args_list, ["v1.2.3", "latest"], strict=True):
                self.assertEqual(
                    call.args[0],
                    [
                        "skopeo",
                        "copy",
                        "--all",
                        "--preserve-digests",
                        "--digestfile",
                        str(Path(tmp) / "published-digest.txt"),
                        f"oci-archive:{tmp}/image.tar",
                        f"docker://ghcr.io/example/semantic-router/dashboard:{tag}",
                    ],
                )
                self.assertTrue(call.kwargs["check"])

    def test_published_archive_preserves_registry_and_checkout_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ci-build-image-provider-mocker"
            directory.mkdir()
            archive_image(directory / "image.tar")
            record = {
                "id": "provider-mocker",
                "source": "published",
                "inputs_sha256": "e" * 64,
                "registry_digest": "sha256:" + "f" * 64,
                "image_source_sha": "a" * 40,
                "ref": images.mocker.REGISTRY + "@sha256:" + "f" * 64,
                "images": images.oci_images(directory / "image.tar"),
            }
            with (
                patch.dict(os.environ, PUBLISHED_IMAGE=json.dumps(record)),
                patch.object(images.subprocess, "run") as copy_image,
                patch.object(images, "source_sha", return_value="b" * 40),
                patch.object(images.mocker, "input_fingerprint", return_value="e" * 64),
            ):
                images.acquire_published(directory)
                command = copy_image.call_args.args[0]
                self.assertIn("docker://" + record["ref"], command)
                self.assertIn("--preserve-digests", command)
                manifest = images.verify(directory, "provider-mocker")
                self.assertEqual(manifest["source_sha"], "b" * 40)
                self.assertEqual(manifest["image_source_sha"], "a" * 40)
                builds = load_builds(Path(tmp))
                self.assertEqual(builds[0]["id"], "image:provider-mocker")
                self.assertEqual(
                    builds[0]["registry_digest"], record["registry_digest"]
                )
                record["images"][0]["config"] = "sha256:" + "9" * 64
                os.environ["PUBLISHED_IMAGE"] = json.dumps(record)
                with self.assertRaisesRegex(ValueError, "differs from planned"):
                    images.acquire_published(directory)

    def test_import_verifies_content_for_both_docker_image_id_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "image.tar"
            archive_image(archive)
            expected = images.oci_images(archive)[0]
            actual = {
                "Os": "linux",
                "Architecture": "amd64",
                "RootFS": {"Type": "layers", "Layers": ["sha256:" + "a" * 64]},
                "Config": {
                    "Cmd": ["router"],
                    "Labels": {"empty-label": ""},
                    "AttachStdin": False,
                    "Volumes": None,
                },
            }
            for image_id in (expected["config"], expected["manifest"]):
                with self.subTest(image_id=image_id):
                    images.verify_loaded_image(
                        archive, expected, {**actual, "Id": image_id}
                    )
            for field, value in (
                ("Architecture", "arm64"),
                ("RootFS", {"Type": "layers", "Layers": ["sha256:" + "b" * 64]}),
                ("Config", {"Cmd": ["another-command"], "Labels": {"empty-label": ""}}),
            ):
                with self.subTest(field=field):
                    changed = copy.deepcopy(actual)
                    changed[field] = value
                    with self.assertRaisesRegex(ValueError, "differ"):
                        images.verify_loaded_image(archive, expected, changed)


if __name__ == "__main__":
    unittest.main()
