"""Training-tree index includes the KV mapper family."""

from __future__ import annotations

import unittest
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]


class KvMapperFamilyTest(unittest.TestCase):
    def test_family_layout(self) -> None:
        family = TRAINING_ROOT / "kv_mapper"
        self.assertTrue((family / "README.md").is_file())
        self.assertTrue((family / "requirements.txt").is_file())
        self.assertTrue((family / "artifact.py").is_file())
        self.assertTrue((family / "fit.py").is_file())
        self.assertTrue((family / "collect.py").is_file())
        self.assertTrue((family / "hooks.py").is_file())
        self.assertTrue((family / "mapper_id.py").is_file())
        self.assertTrue((family / "tests" / "test_artifact.py").is_file())
        self.assertTrue((family / "tests" / "test_collect.py").is_file())
        self.assertTrue((family / "tests" / "test_fit.py").is_file())


if __name__ == "__main__":
    unittest.main()
