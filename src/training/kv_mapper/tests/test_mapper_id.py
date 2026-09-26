from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.mapper_id import make_mapper_id


class MapperIdTests(unittest.TestCase):
    def test_mapper_id_includes_revisions_dtype_and_heads(self) -> None:
        mapper_id = make_mapper_id(
            pair_slug="qwen3-14b-32b",
            variant="full_head",
            precision="fp16",
            source_revision="abc123def",
            target_revision="fed987cba",
            source_tp=1,
            target_tp=1,
            n_kv_heads=8,
            bundle_version=1,
        )
        self.assertIn("fp16", mapper_id)
        self.assertIn("h8", mapper_id)
        self.assertIn("sabc123def", mapper_id)
        self.assertIn("tfed987cba", mapper_id)
        self.assertTrue(mapper_id.endswith("-b1"))


if __name__ == "__main__":
    unittest.main()
