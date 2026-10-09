from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.mapper_id import make_mapper_id, resolve_weight_commit

SOURCE_SHA = "a" * 28 + "123456789012"
TARGET_SHA = "b" * 40


class MapperIdTests(unittest.TestCase):
    def test_mapper_id_includes_revisions_dtype_and_heads(self) -> None:
        mapper_id = make_mapper_id(
            pair_slug="qwen3-14b-32b",
            variant="full_head",
            precision="fp16",
            source_revision=SOURCE_SHA,
            target_revision=TARGET_SHA,
            source_tp=1,
            target_tp=1,
            n_kv_heads=8,
            bundle_version=1,
        )
        self.assertIn("fp16", mapper_id)
        self.assertIn("h8", mapper_id)
        self.assertIn(f"s{SOURCE_SHA}", mapper_id)
        self.assertIn(f"t{TARGET_SHA}", mapper_id)
        self.assertTrue(mapper_id.endswith("-b1"))

    def test_alias_drift_and_truncated_token_collision(self) -> None:
        replies = iter((SOURCE_SHA, "c" * 28 + "123456789012"))
        revisions = []

        def model_info(model_id, *, revision):
            revisions.append((model_id, revision))
            return SimpleNamespace(sha=next(replies))

        first = resolve_weight_commit("source", "main", model_info)
        second = resolve_weight_commit("source", "main", model_info)
        self.assertEqual(revisions, [("source", "main"), ("source", "main")])
        self.assertNotEqual(first, second)
        first_id = make_mapper_id(
            pair_slug="pair",
            variant="full_head",
            precision="bf16",
            source_revision=first,
            target_revision=TARGET_SHA,
            source_tp=1,
            target_tp=1,
            n_kv_heads=8,
        )
        second_id = make_mapper_id(
            pair_slug="pair",
            variant="full_head",
            precision="bf16",
            source_revision=second,
            target_revision=TARGET_SHA,
            source_tp=1,
            target_tp=1,
            n_kv_heads=8,
        )
        self.assertNotEqual(first_id, second_id)
        for alias in ("main", "alpha-123456789012", "beta-123456789012"):
            with self.subTest(alias=alias), self.assertRaisesRegex(
                ValueError, "commit SHA"
            ):
                make_mapper_id(
                    pair_slug="pair",
                    variant="full_head",
                    precision="bf16",
                    source_revision=alias,
                    target_revision=TARGET_SHA,
                    source_tp=1,
                    target_tp=1,
                    n_kv_heads=8,
                )


if __name__ == "__main__":
    unittest.main()
