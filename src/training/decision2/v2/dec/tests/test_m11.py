"""CPU tests for decoder M11: the NT max-length rule and the per-tier development gates (ops/m11)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m11" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LengthRuleTest(unittest.TestCase):
    def test_nt_max_length(self):
        ml = load("m11_lengths")
        self.assertEqual(ml.nt_max_length(7000), 8448)
        self.assertEqual(ml.nt_max_length(8448), 8448)
        self.assertEqual(ml.nt_max_length(8449), 8704)
        self.assertEqual(ml.nt_max_length(9000), 9216)


def arm(t: float, rp: int) -> dict:
    return {
        "T": t,
        "H_mean": 0.5,
        "proxy": 60.0,
        "by_type": {"choice": {"n": 100, "correct": 50}},
        "by_family": {
            "rule_precedence": {"n": 400, "correct": rp},
            "f": {"n": 10, "correct": 5},
        },
    }


def s5(flags: list[str], top: float, upper: float) -> dict:
    return {
        "check": {"flags": flags, "top_share": top, "top_share_wilson95": [0, upper]}
    }


def hs1(point: str, ref: str, fy_point: float, fy_ref: float) -> dict:
    diag = {point: {"f3_false_yes_rate": fy_point}, ref: {"f3_false_yes_rate": fy_ref}}
    return {"families": {"hs1_unmet_condition": {"diagnostics": diag}}}


class RulesTest(unittest.TestCase):
    def run_rules(self, tier: str, points: dict, ref_s5: dict) -> dict:
        mr = load("m11_rules")
        ref = f"{tier}-C0"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "diag").mkdir()
            arms = {ref: arm(0.7, 300)} | {p: arm(0.7, 300) for p in points}
            readout = root / "readout.json"
            readout.write_text(json.dumps({"arms": arms}))
            (root / "diag" / f"{ref}.score5t.json").write_text(json.dumps(ref_s5))
            for p, spec in points.items():
                (root / "diag" / f"{p}.score5t.json").write_text(json.dumps(spec["s5"]))
                (root / "diag" / f"{p}.htdev2.json").write_text(
                    json.dumps(
                        {
                            "delta": spec["ht"],
                            "ci95": [0, 0],
                            "verdict": spec["verdict"],
                        }
                    )
                )
                (root / "diag" / f"{p}.probes.json").write_text(
                    json.dumps(
                        {
                            "macro_mmlu_arc_gsm8k": 0.5,
                            "macro_delta": spec["ret"],
                            "macro_delta_ci95": [
                                spec["ret"] - 0.02,
                                spec["ret"] + 0.02,
                            ],
                        }
                    )
                )
                (root / "diag" / f"{p}.hs1.json").write_text(
                    json.dumps(hs1(p, ref, spec["fy"], 0.20))
                )
                if "ib" in spec:
                    block = {
                        "family_macro": 0.6,
                        "delta_family_macro": spec["ib"],
                        "paired": {"ci95": [0, 0]},
                    }
                    (root / "diag" / f"{p}.ibdev.json").write_text(
                        json.dumps(
                            {"reference_name": ref, "ib_dev": block, "transfer": block}
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
            ]
            for p in points:
                args += ["--point", f"{p}={ref}"]
            mr.main(args + ["--output", str(out)])
            return json.loads(out.read_text())

    def test_2b_score_collapse_and_yes_bias(self):
        clean = s5([], 0.4, 0.45)
        result = self.run_rules(
            "2b",
            {
                "2b-A": {
                    "s5": clean,
                    "ht": 0.01,
                    "verdict": "TIE",
                    "ret": 0.03,
                    "fy": 0.25,
                },
                "2b-B": {
                    "s5": s5(["COLLAPSE"], 0.95, 0.97),
                    "ht": 0.03,
                    "verdict": "GAIN",
                    "ret": 0.05,
                    "fy": 0.2,
                },
                "2b-C": {
                    "s5": clean,
                    "ht": 0.03,
                    "verdict": "GAIN",
                    "ret": 0.01,
                    "fy": 0.35,
                },
            },
            clean,
        )
        self.assertEqual(result["finalists"], ["2b-A"])
        reasons = {r["point"]: r["reasons"] for r in result["points"]}
        self.assertTrue(any(r.startswith("Score floor") for r in reasons["2b-B"]))
        self.assertTrue(any(r.startswith("yes-bias") for r in reasons["2b-C"]))

    def test_08b_collapsed_reference_and_conservative_rule(self):
        ref = s5(["COLLAPSE", "NO-GAIN"], 0.31, 0.357)
        result = self.run_rules(
            "08b",
            {
                "08b-A": {
                    "s5": s5(["COLLAPSE"], 0.33, 0.37),
                    "ht": 0.005,
                    "verdict": "TIE",
                    "ret": 0.02,
                    "fy": 0.2,
                },
                "08b-B": {
                    "s5": s5(["COLLAPSE"], 0.93, 0.95),
                    "ht": 0.03,
                    "verdict": "GAIN",
                    "ret": 0.02,
                    "fy": 0.2,
                },
                "08b-C": {
                    "s5": s5([], 0.3, 0.35),
                    "ht": -0.005,
                    "verdict": "TIE",
                    "ret": 0.04,
                    "fy": 0.2,
                },
            },
            ref,
        )
        self.assertEqual(result["finalists"], ["08b-A"])
        reasons = {r["point"]: r["reasons"] for r in result["points"]}
        self.assertEqual(reasons["08b-A"], [])
        self.assertTrue(any("top share" in r for r in reasons["08b-B"]))
        self.assertTrue(any(r.startswith("0.8B rule") for r in reasons["08b-C"]))

    def test_4b_breadth_gate_and_order(self):
        clean = s5([], 0.4, 0.45)
        base = {"s5": clean, "ht": 0.0, "verdict": "TIE", "ret": 0.0, "fy": 0.2}
        result = self.run_rules(
            "4b",
            {
                "4b-A": base | {"ib": 0.02},
                "4b-B": base | {"ib": -0.01},
                "4b-C": base | {"ib": 0.05},
                "4b-D": base,
            },
            clean,
        )
        self.assertEqual(result["finalists"], ["4b-C", "4b-A"])
        reasons = {r["point"]: r["reasons"] for r in result["points"]}
        self.assertTrue(
            any(r.startswith("breadth: IB DEV macro") for r in reasons["4b-B"])
        )
        self.assertTrue(
            any(r.startswith("breadth: no IB DEV") for r in reasons["4b-D"])
        )


class StageTwoDataTest(unittest.TestCase):
    def test_sample_groups_whole_groups_and_quota(self):
        sd = load("m11_s2data")
        rows = [
            {"family": f, "group_id": f"{f}{g}", "tokens": 10}
            for f in ("a", "b")
            for g in range(10)
            for _ in range(2)
        ]
        kept, info = sd.sample_groups(rows, 200, 7)
        self.assertEqual(info["fraction"], 0.5)
        self.assertEqual(sum(rows[i]["tokens"] for i in kept), 200)
        groups = {rows[i]["group_id"] for i in kept}
        self.assertEqual(len(kept), 2 * len(groups))
        self.assertEqual({rows[i]["family"] for i in kept}, {"a", "b"})
        self.assertEqual(kept, sd.sample_groups(rows, 200, 7)[0])
        self.assertEqual(len(sd.sample_groups(rows, 999, 7)[0]), len(rows))


class FormalLibTest(unittest.TestCase):
    """ops/m6/m6-formal-lib.sh tier_setup: the M6_SMALL_NODE=E|F option (decoder M11) and the unchanged defaults."""

    def setup(self, tier: str, **env: str) -> subprocess.CompletedProcess:
        lib = HERE / "ops" / "m6" / "m6-formal-lib.sh"
        script = (
            f". {lib}; tier_setup; "
            'echo "$NODE|$IMAGE|$MASTER|$MASTER_SHA|$MASTER_MLX|$SOURCE|${ISOLATE[*]}|$TLABEL|$INCUMBENT|${ENVX[*]}"'
        )
        with tempfile.TemporaryDirectory() as tmp:
            full = {
                "PATH": os.environ["PATH"],
                "S": str(HERE.parents[1]),
                "TIER": tier,
                "M6_FORMAL_ROOT": tmp,
                **env,
            }
            return subprocess.run(
                ["bash", "-c", script], env=full, capture_output=True, text=True
            )

    def fields(self, tier: str, **env: str) -> list[str]:
        out = self.setup(tier, **env)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip().split("|")

    def test_small_node_f(self):
        for tier, label, one in (
            ("2b", "2B", "Decision-1.0-Sol-2B"),
            ("08b", "0.8B", "Decision-1.0-Eos-0.8B"),
        ):
            node, image, master, pin, mlx, source, isolate, tlabel, _, envx = (
                self.fields(tier, M6_SMALL_NODE="F", M6_SMALL_MASTER_DIR="/m")
            )
            self.assertEqual(
                (node, pin, isolate, tlabel, envx),
                ("F", "frozen", "--isolate", label, ""),
            )
            self.assertTrue(image.startswith("sha256:dbe5f32b"))
            self.assertEqual(
                (master, mlx),
                (f"/m/cache-frozen-{tier}", f"/m/cache-frozen-{tier}-mlx"),
            )
            self.assertTrue(source.startswith(f"/data/dev2/models/{one}/"))

    def test_defaults_unchanged(self):
        node, image, master, pin, *_ = self.fields("08b")
        self.assertEqual((node, pin), ("A", "live"))
        self.assertTrue(master.endswith("formal/m2/m2-E8F-soup-nodeA-triton"))
        node, _, master, pin, *_ = self.fields("2b")
        self.assertEqual((node, pin), ("B", "frozen"))
        self.assertTrue(master.endswith("/cache-frozen-2b"))
        node, *_ = self.fields("2b", M6_2B_NODE="A")
        self.assertEqual(node, "A")

    def test_bad_small_node(self):
        self.assertEqual(
            self.setup("2b", M6_SMALL_NODE="B", M6_SMALL_MASTER_DIR="/m").returncode, 2
        )


if __name__ == "__main__":
    unittest.main()
