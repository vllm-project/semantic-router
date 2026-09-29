from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.htdev import fetch


class ReuseTest(unittest.TestCase):
    def test_reuse_links_by_sha_under_the_isolation_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw" / "claimstance" / "nested"
            raw.mkdir(parents=True)
            (raw / "renamed.csv").write_bytes(b"topic,claim\n")
            (raw / "other.csv").write_bytes(b"unrelated\n")
            digest = hashlib.sha256(b"topic,claim\n").hexdigest()
            pins = {
                "sources": {
                    "claim_stance": {
                        "kind": "url",
                        "repo": "r",
                        "revision": "x",
                        "licence": "l",
                        "licence_evidence": "e",
                        "urls": {"test.csv": "https://invalid.example/never"},
                        "files": {"test.csv": digest},
                    }
                }
            }
            (tmp / "pins.json").write_text(json.dumps(pins))
            argv = ["fetch", "--pins", str(tmp / "pins.json"), "--dest"]
            argv += [str(tmp / "dest"), "--reuse", str(tmp / "raw")]
            self.assertEqual(fetch.main(argv), 0)
            target = tmp / "dest" / "claim_stance" / "test.csv"
            self.assertEqual(target.read_bytes(), b"topic,claim\n")
            snapshot = json.loads(
                (tmp / "dest" / "claim_stance" / "SNAPSHOT.json").read_text()
            )
            self.assertEqual(snapshot["files"], {"test.csv": digest})


if __name__ == "__main__":
    unittest.main()
