"""Synthetic end-to-end tests; no real private prompt or answer is read."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.data.materialize_goldfree_inventory import (
    project_core,
    write_private_inventory,
)
from training.data.plan_goldfree_inventory import (
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
)
from training.model.data import INPUT_FIELDS, digest, file_sha256


class MaterializeCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _jsonl(self, name: str, row: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def _sources(self) -> tuple[Path, dict[str, tuple[Path, str]]]:
        entries = []
        for role in NATIVE_ROLE_COUNTS:
            source = self._jsonl(
                f"{role}.source.jsonl",
                {
                    "id": f"native-{role}",
                    "state": {"target": {"entity": "widget", "item": "certificate"}},
                    "questions": {"label": {"criteria": "Choose valid evidence"}},
                },
            )
            entries.append(
                {"role": role, "path": str(source), "sha256": file_sha256(source)}
            )
        entries.append({"role": "optional-unattested", "path": "never-opened"})
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps(entries), encoding="utf-8")
        partitions = {}
        for role, (split, _) in PARTITION_ROLE_COUNTS.items():
            row = {
                "id": f"partition-{role}",
                "split": split,
                "state": "Synthetic evidence state",
                "instructions": "Select one option",
                "options": [{"key": "a", "description": "Verified"}],
                "task_type": "choice",
                "label": "PRIVATE_TARGET_SENTINEL",
            }
            row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
            source = self._jsonl(f"{role}.source.jsonl", row)
            partitions[role] = (source, file_sha256(source))
        return manifest, partitions

    def test_projection_is_pinned_input_only_and_never_overwrites(self) -> None:
        with patch.dict(NATIVE_ROLE_COUNTS, dict.fromkeys(NATIVE_ROLE_COUNTS, 1)):
            with patch.dict(
                PARTITION_ROLE_COUNTS,
                {
                    role: (split, 1)
                    for role, (split, _) in PARTITION_ROLE_COUNTS.items()
                },
            ):
                source, partitions = self._sources()
                roles, identity = project_core(source, file_sha256(source), partitions)
                self.assertEqual(len(roles), 8)
                self.assertEqual(identity["excluded_optional_role_count"], 1)
                output = self.root / "private-projection"
                receipt = write_private_inventory(output, roles, identity)
                self.assertTrue(receipt.exists())
                manifest = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(set(manifest["roles"]), set(roles))
                for role, entry in manifest["roles"].items():
                    projected = output / entry["path"]
                    self.assertEqual(file_sha256(projected), entry["sha256"])
                    self.assertNotIn("PRIVATE_TARGET_SENTINEL", projected.read_text())
                with self.assertRaisesRegex(ValueError, "already exists"):
                    write_private_inventory(output, roles, identity)
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    project_core(source, "0" * 64, partitions)

    def test_missing_native_or_partition_fails_before_output(self) -> None:
        with patch.dict(NATIVE_ROLE_COUNTS, dict.fromkeys(NATIVE_ROLE_COUNTS, 1)):
            with patch.dict(
                PARTITION_ROLE_COUNTS,
                {
                    role: (split, 1)
                    for role, (split, _) in PARTITION_ROLE_COUNTS.items()
                },
            ):
                source, partitions = self._sources()
                absent = next(iter(partitions))
                del partitions[absent]
                with self.assertRaisesRegex(ValueError, "partition roles"):
                    project_core(source, file_sha256(source), partitions)
                self.assertFalse((self.root / "private-projection").exists())


if __name__ == "__main__":
    unittest.main()
