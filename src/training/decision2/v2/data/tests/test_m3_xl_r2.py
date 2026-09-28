from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.data.m3 import xl_r2


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _r1_dir(root: Path, recipes: dict[str, list[dict]]) -> tuple[Path, str]:
    r1 = root / "r1"
    r1.mkdir()
    manifest = {"recipes": {}}
    for name in xl_r2.R1_RECIPES:
        path = _jsonl(r1 / f"{name}.ids.jsonl", recipes.get(name, []))
        manifest["recipes"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    raw = json.dumps(manifest).encode()
    (r1 / "mx-xl.manifest.json").write_bytes(raw)
    return r1, hashlib.sha256(raw).hexdigest()


class RowsTest(unittest.TestCase):
    def test_resolves_the_union_with_pool_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_a = root / "a.jsonl"
            lines = [
                '{"id": "a1", "group_id": "g1", "state": "x"}\n',
                '{"state": "y",   "group_id": "g1", "id": "a2"}\n',
                '{"id": "a3", "group_id": "g2", "state": "z"}\n',
            ]
            pool_a.write_text("".join(lines), encoding="utf-8")
            _jsonl(root / "b.jsonl", [{"id": "b1", "group_id": "g1", "state": "w"}])
            specs = {
                "A": {"rows": [str(pool_a)]},
                "V1:B": {"rows": [str(root / "b.jsonl")]},
            }
            (root / "pools.json").write_text(json.dumps(specs))
            ids = lambda *pairs: [  # noqa: E731
                {"id": i, "pool": p, "native": 5} for i, p in pairs
            ]
            r1, digest = _r1_dir(
                root,
                {
                    "mx-xl-full": ids(("a2", "A")),
                    "cx-xl-v2v1-short": ids(("a1", "A"), ("b1", "V1:B")),
                },
            )
            argv = [
                "rows",
                "--pools",
                str(root / "pools.json"),
                "--r1-dir",
                str(r1),
                "--r1-manifest-sha256",
                digest,
                "--out-dir",
                str(root / "out"),
            ]
            xl_r2.main(argv)
            out = root / "out"
            self.assertEqual(
                (out / "rows" / "A.jsonl").read_text(), lines[0] + lines[1]
            )
            self.assertTrue((out / "rows" / "V1-B.jsonl").is_file())
            index = [json.loads(line) for line in open(out / "index.jsonl")]
            self.assertEqual(
                [(r["pool"], r["id"], r["group"]) for r in index],
                [("A", "a1", "g1"), ("A", "a2", "g1"), ("V1:B", "b1", "g1")],
            )
            receipt = json.loads((out / "rows.receipt.json").read_text())
            self.assertEqual(receipt["union"]["rows"], 3)
            self.assertEqual(receipt["union"]["group_ids_in_several_pools"], 1)
            with self.assertRaises(ValueError):
                xl_r2.main(argv[:-3] + ["0" * 64, "--out-dir", str(root / "o2")])

    def test_an_id_missing_from_its_pool_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _jsonl(root / "a.jsonl", [{"id": "a1", "group_id": "g1", "state": "x"}])
            specs = {"A": {"rows": [str(root / "a.jsonl")]}}
            recipes = {"mx-xl-full": [{"id": "a9", "pool": "A", "native": 1}]}
            with self.assertRaises(ValueError):
                xl_r2.resolve_rows(specs, recipes, root / "out")


if __name__ == "__main__":
    unittest.main()
