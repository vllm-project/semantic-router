"""CPU tests for decoder M12: additive TRAIN arms (m12_data) and the transfer-breadth gate (m12_rules)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_m11 import arm, hs1, s5  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m12" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(
    block: str, families: tuple[str, ...], groups: int, tokens: int = 10
) -> list[dict]:
    out = []
    for f in families:
        for g in range(groups):
            for k in range(2):
                rid = f"{block}-{f}{g}-{k}"
                line = (
                    json.dumps({"id": rid, "family": f, "group_id": f"{f}{g}"}).encode()
                    + b"\n"
                )
                out.append(
                    {
                        "block": block,
                        "family": f,
                        "group_id": f"{f}{g}",
                        "id": rid,
                        "tokens": tokens,
                        "line": line,
                    }
                )
    return out


class DataTest(unittest.TestCase):
    def test_parse_arm(self):
        md = load("m12_data")
        self.assertEqual(md.parse_arm("4b-LHA=all:0.25"), ("4b-LHA", "all", 0.25, None))
        self.assertEqual(md.parse_arm("08b-RA=all:x3"), ("08b-RA", "all", None, 3))
        self.assertEqual(md.parse_arm("4b-LHAx=transfer:0.10")[1], "transfer")
        for bad in ("a=some:0.1", "a=all:1.5", "a=all:x0"):
            with self.assertRaises(ValueError):
                md.parse_arm(bad)

    def test_share_is_additive(self):
        md = load("m12_data")
        base = rows("base", ("t",), 20)
        ib = rows("ib1", ("a", "b"), 10)
        total = sum(r["tokens"] for r in base)
        out, info = md.build_arm(base, ib, total, 0.25, None, 7)
        self.assertEqual(
            [r["id"] for r, _ in out[: len(base)]], [r["id"] for r in base]
        )
        ib_rows = out[len(base) :]
        ib_tokens = sum(r["tokens"] for r, _ in ib_rows)
        # whole groups never exceed a family's quota (50 tokens each); the shortfall is under one group per family
        self.assertTrue(100 - 2 * 20 < ib_tokens <= 100)
        self.assertEqual(info["ib_share_of_T"], ib_tokens / total)
        self.assertTrue(all(k == 1 for _, k in out))
        self.assertEqual(len({r["group_id"] for r, _ in ib_rows}) * 2, len(ib_rows))
        with self.assertRaises(ValueError):
            md.build_arm(base, ib[:10], total, 0.5, None, 7)

    def test_copies_get_suffixed_ids(self):
        md = load("m12_data")
        base = rows("base", ("t",), 5)
        ib = rows("ib2", ("a",), 2)
        total = sum(r["tokens"] for r in base)
        out, info = md.build_arm(base, ib, total, None, 3, 7)
        self.assertEqual(len(out), len(base) + 3 * len(ib))
        self.assertEqual(info["ib_tokens"], 3 * sum(r["tokens"] for r in ib))
        ids = [json.loads(md.copy_line(r["line"], k))["id"] for r, k in out]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(f"{ib[0]['id']}~c3", ids)
        self.assertEqual(md.copy_line(ib[0]["line"], 1), ib[0]["line"])
        copied = json.loads(md.copy_line(ib[0]["line"], 2))
        self.assertEqual(
            (copied["group_id"], copied["family"]), (ib[0]["group_id"], "a")
        )


class RulesTest(unittest.TestCase):
    def run_rules(self, tier: str, points: dict, ref: str) -> dict:
        mr = load("m12_rules")
        clean = s5([], 0.4, 0.45)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "diag").mkdir()
            readout = root / "readout.json"
            readout.write_text(
                json.dumps(
                    {"arms": {ref: arm(0.7, 300)} | {p: arm(0.7, 300) for p in points}}
                )
            )
            (root / "diag" / f"{ref}.score5t.json").write_text(json.dumps(clean))
            for p, spec in points.items():
                d = root / "diag"
                (d / f"{p}.score5t.json").write_text(json.dumps(clean))
                (d / f"{p}.htdev2.json").write_text(
                    json.dumps(
                        {
                            "delta": spec["ht"],
                            "ci95": [0, 0],
                            "verdict": spec["verdict"],
                        }
                    )
                )
                (d / f"{p}.probes.json").write_text(
                    json.dumps(
                        {
                            "macro_mmlu_arc_gsm8k": 0.5,
                            "macro_delta": 0.0,
                            "macro_delta_ci95": [-0.02, 0.02],
                        }
                    )
                )
                (d / f"{p}.hs1.json").write_text(json.dumps(hs1(p, ref, 0.2, 0.2)))
                if "tr" in spec:
                    block = {
                        "family_macro": 0.9,
                        "delta_family_macro": spec.get("ib", 0.05),
                        "paired": {"ci95": [0, 0]},
                    }
                    transfer = block | {"delta_family_macro": spec["tr"]}
                    (d / f"{p}.ibdev.json").write_text(
                        json.dumps(
                            {
                                "reference_name": spec.get("refname", ref),
                                "ib_dev": block,
                                "transfer": transfer,
                            }
                        )
                    )
            out = root / "sel.json"
            args = [
                "--tier",
                tier,
                "--lines-root",
                str(root),
                "--readout",
                str(readout),
                "--output",
                str(out),
            ]
            for p in points:
                args += ["--point", f"{p}={ref}"]
            mr.main(args)
            with self.assertRaises(FileExistsError):
                mr.main(args)
            return json.loads(out.read_text())

    def test_transfer_gate_and_order(self):
        tie = {"ht": 0.0, "verdict": "TIE"}
        result = self.run_rules(
            "4b",
            {
                "4b-A": tie | {"tr": 0.02},
                "4b-B": tie | {"tr": -0.01, "ib": 0.09},
                "4b-C": tie | {"tr": 0.05},
                "4b-D": tie,
                "4b-E": tie | {"tr": 0.04, "refname": "4b-other"},
                "4b-F": {"ht": 0.02, "verdict": "GAIN", "tr": 0.001},
            },
            "4b-LH-f",
        )
        self.assertEqual(result["finalists"], ["4b-F", "4b-C"])
        reasons = {r["point"]: r["reasons"] for r in result["points"]}
        self.assertTrue(
            any(r.startswith("breadth: transfer macro") for r in reasons["4b-B"])
        )
        self.assertTrue(
            any(r.startswith("breadth: no IB DEV") for r in reasons["4b-D"])
        )
        self.assertTrue(
            any(r.startswith("breadth: no IB DEV") for r in reasons["4b-E"])
        )
        self.assertEqual(reasons["4b-A"], [])

    def test_08b_keeps_conservative_rule(self):
        result = self.run_rules(
            "08b",
            {
                "08b-A": {"ht": -0.004, "verdict": "TIE", "tr": 0.03},
                "08b-B": {"ht": 0.01, "verdict": "TIE", "tr": 0.03},
            },
            "08b-C0-e",
        )
        self.assertEqual(result["finalists"], ["08b-B"])
        reasons = {r["point"]: r["reasons"] for r in result["points"]}
        self.assertTrue(any(r.startswith("0.8B rule") for r in reasons["08b-A"]))


if __name__ == "__main__":
    unittest.main()
