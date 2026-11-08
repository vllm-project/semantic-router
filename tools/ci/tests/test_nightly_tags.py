from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nightly_tags as tags

AMD64, ARM64 = "sha256:" + "a" * 64, "sha256:" + "b" * 64


def receipt(*images: tuple[str, str], date: str = "20261105") -> dict:
    return {
        "date": date,
        "images": [
            {"platform": platform, "manifest": digest, "config": "sha256:" + "c" * 64}
            for platform, digest in images
        ],
    }


def index(*images: tuple[str, str]) -> bytes:
    return json.dumps(
        {
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": digest,
                    "platform": dict(
                        zip(("os", "architecture"), platform.split("/"), strict=True)
                    ),
                }
                for platform, digest in images
            ],
        }
    ).encode()


BARE = json.dumps({"mediaType": "application/vnd.oci.image.manifest.v1+json"}).encode()
BARE_DIGEST = "sha256:" + hashlib.sha256(BARE).hexdigest()


class OwnershipTests(unittest.TestCase):
    def test_tags_are_absent_ours_or_another_qualifications(self) -> None:
        both = receipt(("linux/amd64", AMD64), ("linux/arm64", ARM64))
        cases = [
            (None, both, tags.ABSENT),
            (index(("linux/amd64", AMD64), ("linux/arm64", ARM64)), both, tags.OURS),
            (index(("linux/amd64", AMD64), ("linux/arm64", AMD64)), both, tags.OTHER),
            (index(("linux/amd64", AMD64)), both, tags.OTHER),
            (BARE, receipt(("linux/amd64", BARE_DIGEST)), tags.OURS),
            (BARE, receipt(("linux/amd64", AMD64)), tags.OTHER),
        ]
        for raw, built, expected in cases:
            with self.subTest(expected=expected, raw=raw):
                self.assertEqual(tags.ownership(raw, built), expected)

    def test_publishes_only_a_free_date_or_its_own_partial_publication(self) -> None:
        cases = [
            ({"vllm-sr": tags.ABSENT, "dashboard": tags.ABSENT}, False, True),
            ({"vllm-sr": tags.OURS, "dashboard": tags.ABSENT}, True, True),
            ({"vllm-sr": tags.OURS, "dashboard": tags.OTHER}, True, False),
            ({"vllm-sr": tags.ABSENT, "dashboard": tags.ABSENT}, True, False),
        ]
        for states, chart, expected in cases:
            with self.subTest(states=states, chart=chart):
                self.assertEqual(tags.decide(states, chart)[0], expected)

    def test_only_a_missing_tag_reads_as_absent(self) -> None:
        def answer(code: int, out: bytes = b"", err: bytes = b""):
            return mock.patch.object(
                tags.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], code, out, err),
            )

        with answer(0, out=BARE):
            self.assertEqual(tags.registry_manifest("ghcr.io/o/i:t"), BARE)
        with answer(1, err=b"reading manifest t: manifest unknown"):
            self.assertIsNone(tags.registry_manifest("ghcr.io/o/i:t"))
        with (
            answer(1, err=b"unauthorized: authentication required"),
            self.assertRaisesRegex(RuntimeError, "unauthorized"),
        ):
            tags.registry_manifest("ghcr.io/o/i:t")


class CommandTests(unittest.TestCase):
    def run_check(self, registry: dict[str, bytes], date: str = "20261105") -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for image, digest in (("vllm-sr", AMD64), ("dashboard", ARM64)):
                path = root / f"ci-build-image-{image}" / "manifest.json"
                path.parent.mkdir()
                path.write_text(json.dumps(receipt(("linux/amd64", digest))))
            output = root / "output"
            argv = [
                "nightly_tags.py",
                "--date",
                date,
                "--images",
                json.dumps(["vllm-sr", "dashboard"]),
                "--receipts",
                str(root),
                "--registry",
                "ghcr.io/o",
                "--github-output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(tags, "registry_manifest", side_effect=registry.get),
                mock.patch("builtins.print"),
            ):
                tags.main()
            return output.read_text()

    def test_a_new_date_publishes(self) -> None:
        self.assertEqual(self.run_check({}), "publish=true\n")

    def test_another_commits_tag_on_the_same_date_stops_publication(self) -> None:
        registry = {
            "ghcr.io/o/semantic-router/vllm-sr:nightly-20261105": index(
                ("linux/amd64", ARM64)
            )
        }
        self.assertEqual(self.run_check(registry), "publish=false\n")

    def test_a_rerun_finishes_its_own_publication(self) -> None:
        registry = {
            "ghcr.io/o/semantic-router/vllm-sr:nightly-20261105": index(
                ("linux/amd64", AMD64)
            ),
            "ghcr.io/o/charts/semantic-router:0.0.0-nightly.20261105": b"{}",
        }
        self.assertEqual(self.run_check(registry), "publish=true\n")

    def test_images_must_carry_the_runs_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "built for 20261105, not 20261106"):
            self.run_check({}, date="20261106")


if __name__ == "__main__":
    unittest.main()
