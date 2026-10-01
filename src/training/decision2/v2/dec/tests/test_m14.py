"""CPU tests for decoder M14: the per-row loss weights of the upweighted arms (m14_weights)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m14" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(rid: str, kind: str) -> bytes:
    return json.dumps({"id": rid, "task_type": kind, "family": "f"}).encode() + b"\n"


def ids(rid: str, block: str) -> bytes:
    return json.dumps({"id": rid, "block": block, "tokens": 3}).encode() + b"\n"


class WeightsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.w = load("m14_weights")
        self.train = [
            row("a", "choice"),
            row("b", "score"),
            row("ib1", "noul"),
            row("ib1~c2", "noul"),
        ]
        self.ids = [
            ids("a", "base"),
            ids("b", "base"),
            ids("ib1", "ib1"),
            ids("ib1~c2", "ib1"),
        ]
        self.released_sha = hashlib.sha256(b"".join(self.train[:2])).hexdigest()
        self.typed = self.w.parse_weights("choice=1.5,noul=1.5,score=2")

    def test_released_rows_take_their_type_weight(self) -> None:
        records, report = self.w.build(
            self.train, self.ids, 2, self.released_sha, self.typed, 1.0
        )
        self.assertEqual([r["weight"] for r in records], [1.5, 2.0, 1.0, 1.0])
        self.assertEqual([r["id"] for r in records], ["a", "b", "ib1", "ib1~c2"])
        self.assertAlmostEqual(report["released_weight_share"], 3.5 / 5.5)
        self.assertAlmostEqual(report["released_row_share"], 0.5)
        self.assertAlmostEqual(report["type_weight_share"]["score"], 2.0 / 5.5)
        self.assertEqual(
            report["by_block_type"]["ib"]["noul"], {"rows": 2, "weight": 2.0}
        )

    def test_rejects_inconsistent_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.w.build(self.train, self.ids, 3, self.released_sha, self.typed, 1.0)
        with self.assertRaises(ValueError):
            self.w.build(self.train, self.ids, 2, "0" * 64, self.typed, 1.0)
        with self.assertRaises(ValueError):
            self.w.build(
                self.train, self.ids[:3], 2, self.released_sha, self.typed, 1.0
            )
        swapped = [self.ids[0], self.ids[2], self.ids[1], self.ids[3]]
        with self.assertRaises(ValueError):
            self.w.build(self.train, swapped, 2, self.released_sha, self.typed, 1.0)
        late = self.ids[:2] + [ids("ib1", "ib1"), ids("ib1~c2", "base")]
        with self.assertRaises(ValueError):
            self.w.build(self.train, late, 2, self.released_sha, self.typed, 1.0)
        for spec in (
            "choice=1.5,noul=1.5",
            "choice=1,noul=1,score=0",
            "choice=1,noul=1,score=1,x=2",
        ):
            with self.assertRaises(ValueError):
                self.w.parse_weights(spec)


if __name__ == "__main__":
    unittest.main()
