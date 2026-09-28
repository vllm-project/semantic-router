"""Slow regression test: the frozen A2/A6 builds must stay byte-identical (about 10 s)."""

import tempfile
import unittest
from pathlib import Path

from training.model.data import file_sha256
from v2.data.verifiable import build

FROZEN = {
    ("a2", "decision2-a2-v1"): {
        "a2.train.jsonl": "4a1703ab8c484baca1232615b796b2147078771eab2128fb301c94272c3a1b1f",
        "a2.aho.jsonl": "13d81b6d6456d7df809d9142eab28755d0a460a352144e661036343155c7f6c7",
    },
    ("a6", "decision2-a6-v1"): {
        "a6.train.jsonl": "5c14863a1ef40f5e7315c6a9eb51ae2e7388bccf997357b359ec7d9536e9b231",
        "a6.aho.jsonl": "e431c72bd1e2c572fb04289ce903fbbe1490ae8bb7974613067758963a49b3df",
    },
}


class FrozenArmsTest(unittest.TestCase):
    def test_default_a2_and_a6_builds_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for (arm, seed), expected in FROZEN.items():
                files, manifest = build.generate(arm, seed, None, 0.3)
                build.write(out, arm, files, manifest)
                for name, digest in expected.items():
                    self.assertEqual(file_sha256(out / name), digest, name)


if __name__ == "__main__":
    unittest.main()
