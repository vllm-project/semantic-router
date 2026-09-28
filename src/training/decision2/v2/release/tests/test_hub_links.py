"""Hub link readback (stdlib, stubbed network)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.release import hub_links

README = """---
license: apache-2.0
---

![banner](assets/b.png)

# DEV2.0-0.6B

[Collection](https://huggingface.co/collections/org/decision-20-abc) · [Download](#download-and-decide)

## Download and decide

Origin: [Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base). [License](LICENSE) · [License](LICENSE)
"""


class HubLinksTest(unittest.TestCase):
    def test_links_and_anchors(self):
        self.assertEqual(
            hub_links.links(README),
            [
                (True, "assets/b.png"),
                (False, "https://huggingface.co/collections/org/decision-20-abc"),
                (False, "#download-and-decide"),
                (False, "https://huggingface.co/Qwen/Qwen3-0.6B-Base"),
                (False, "LICENSE"),
            ],
        )
        self.assertIn("download-and-decide", hub_links.anchors(README))

    def test_check_hashes_files_and_uses_the_api_for_hub_pages(self):
        with tempfile.TemporaryDirectory() as scratch:
            package = Path(scratch)
            files = {
                "README.md": README,
                "ATTRIBUTIONS.md": "- [x](LICENSE)\n",
                "LICENSE": "L\n",
            }
            for name, text in files.items():
                (package / name).write_text(text, encoding="utf-8")
            (package / "assets").mkdir()
            (package / "assets/b.png").write_bytes(b"png")
            manifest = {
                "model_name": "DEV2.0-0.6B",
                "files_sha256": {
                    name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in (*files, "assets/b.png")
                },
            }
            (package / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
            calls = []

            def hub(tamper: str | None):
                def fake(url, token):
                    calls.append(url)
                    if "/resolve/" in url:
                        name = url.split("/resolve/rev/", 1)[1]
                        return 200, (
                            b"x" if name == tamper else (package / name).read_bytes()
                        )
                    if "/api/" in url:
                        return (
                            200,
                            json.dumps({"private": "collections" in url}).encode(),
                        )
                    return 200, b"<h1>DEV2.0-0.6B</h1> owl-banner.png"

                return fake

            with mock.patch.object(hub_links, "fetch", side_effect=hub(None)):
                result = hub_links.check("org/DEV2.0-0.6B", "rev", package, "t")
            self.assertTrue(result["passed"], result["failed"])
            self.assertEqual(result["checked"], 6)
            self.assertIn(
                "https://huggingface.co/api/models/Qwen/Qwen3-0.6B-Base", calls
            )
            self.assertIn(
                "https://huggingface.co/api/collections/org/decision-20-abc", calls
            )
            self.assertTrue(result["rendered_page"]["title_present"])

            with mock.patch.object(hub_links, "fetch", side_effect=hub("LICENSE")):
                result = hub_links.check("org/DEV2.0-0.6B", "rev", package, "t")
            self.assertFalse(result["passed"])
            self.assertEqual(result["failed"], ["LICENSE", "LICENSE"])


if __name__ == "__main__":
    unittest.main()
