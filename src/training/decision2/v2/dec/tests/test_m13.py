"""CPU tests for decoder M13: the family-upweighted TRAIN (m13_data)."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m13" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line(rid: str, family: str) -> bytes:
    return json.dumps({"id": rid, "family": family, "label": "1"}).encode() + b"\n"


class UpweightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load("m13_data")
        self.arm = [
            line("a1", "gate"),
            line("a2", "other"),
            line("a3", "gate"),
            line("ib1~c2", "ib_family"),
        ]
        self.released = {"a1", "a2", "a3"}

    def test_keeps_arm_then_appends_copies(self) -> None:
        lines, info = self.data.upweighted(self.arm, self.released, {"gate"}, 3)
        self.assertEqual(lines[:4], self.arm)
        ids = [json.loads(x)["id"] for x in lines[4:]]
        self.assertEqual(ids, ["a1~a2", "a3~a2", "a1~a3", "a3~a3"])
        self.assertEqual(info["upweighted_rows"], 2)
        self.assertEqual(info["added_rows"], 4)
        copy, original = json.loads(lines[4]), json.loads(self.arm[0])
        self.assertEqual(copy.pop("id"), "a1~a2")
        original.pop("id")
        self.assertEqual(copy, original)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, self.released, {"gate"}, 1)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, self.released, {"gate", "absent"}, 2)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm, {"a1", "a2"}, {"gate"}, 2)
        with self.assertRaises(ValueError):
            self.data.upweighted(self.arm + [self.arm[0]], self.released, {"gate"}, 2)


if __name__ == "__main__":
    unittest.main()
