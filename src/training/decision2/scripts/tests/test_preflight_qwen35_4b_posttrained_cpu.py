"""Metadata compatibility gates for the official 4B initialization ablation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_qwen35_4b_posttrained_cpu import compare_metadata


def write_metadata(
    path: Path, *, added: list[dict], config: dict | None = None
) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(config or {"architectures": ["Qwen3_5ForConditionalGeneration"]})
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": {"w": "a"}})
    )
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"a": 0}, "merges": []},
                "added_tokens": added,
            }
        )
    )


class MetadataComparisonTests(unittest.TestCase):
    def test_allows_only_new_special_tokens_with_same_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_metadata(root / "base", added=[{"content": "old", "id": 1}])
            write_metadata(
                root / "post",
                added=[{"content": "old", "id": 1}, {"content": "new", "id": 2}],
            )
            result = compare_metadata(root / "base", root / "post")
            self.assertEqual(result["new_special_tokens"], {"new": 2})

    def test_rejects_reassigned_existing_special_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_metadata(root / "base", added=[{"content": "old", "id": 1}])
            write_metadata(root / "post", added=[{"content": "old", "id": 2}])
            with self.assertRaisesRegex(ValueError, "Existing special-token IDs"):
                compare_metadata(root / "base", root / "post")

    def test_rejects_weight_tensor_roster_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_metadata(root / "base", added=[])
            write_metadata(root / "post", added=[])
            (root / "post" / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"metadata": {"total_size": 8}, "weight_map": {"other": "a"}}
                )
            )
            with self.assertRaisesRegex(ValueError, "Weight tensor roster"):
                compare_metadata(root / "base", root / "post")


if __name__ == "__main__":
    unittest.main()
