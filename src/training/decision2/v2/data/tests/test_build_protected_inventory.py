from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.build_protected_inventory import build, project_row, sha_file


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return sha_file(path)


class BuildProtectedInventoryTest(unittest.TestCase):
    def test_projection_drops_answers_and_metadata(self) -> None:
        row = {
            "id": "x1",
            "state": "s",
            "questions": {"q": {"type": "choice", "gold": "a", "criteria": {"a": "A"}}},
            "gold": {"q": "a"},
            "notes": "private",
        }
        self.assertEqual(
            project_row(row),
            {
                "id": "x1",
                "state": "s",
                "questions": {"q": {"type": "choice", "criteria": {"a": "A"}}},
            },
        )

    def test_build_rejects_unprojected_answers_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.jsonl"
            keyed = root / "keyed.jsonl"
            clean_sha = _write(
                clean, [{"id": "a", "state": "alpha", "instructions": "q"}]
            )
            keyed_sha = _write(keyed, [{"id": "b", "state": "beta", "gold": 1}])
            spec = root / "spec.json"
            spec.write_text(
                json.dumps([{"role": "r1", "origin": str(keyed), "sha256": keyed_sha}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build(spec, root / "out-bad")
            spec.write_text(
                json.dumps(
                    [
                        {"role": "r1", "origin": str(clean), "sha256": clean_sha},
                        {"role": "r2", "origin": str(clean), "sha256": clean_sha},
                        {
                            "role": "r3",
                            "origin": str(keyed),
                            "sha256": keyed_sha,
                            "project": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            receipt = build(spec, root / "out")
            roles = {entry["role"]: entry for entry in receipt["roles"]}
            self.assertEqual(roles["r2"]["duplicate_of"], "r1")
            self.assertEqual(receipt["rows_total"], 2)
            manifest = json.loads(
                (root / "out" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual([entry["role"] for entry in manifest], ["r1", "r3"])
            with self.assertRaises(FileExistsError):
                build(spec, root / "out")


if __name__ == "__main__":
    unittest.main()
