from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lux9b import m9_data, m9_rules


def pn1(clean_no, lo, hop):
    return {
        "delta": {"clean_no": clean_no, "hop": hop, "pawsx6": 0.0, "all8": 0.0},
        "delta_ci95": {"clean_no": [lo, lo + 0.04], "hop": [hop - 0.01, hop + 0.01]},
        "candidate": {"clean_no": {"yes": 0.3}, "hop": {"yes": 0.98}},
        "reference_summary": {"clean_no": {"yes": 0.26}, "hop": {"yes": 0.98}},
    }


def hs1(point, ref):
    return {
        "families": {
            "hs1_unmet_condition": {
                "diagnostics": {
                    "X": {"f3_false_yes_rate": point},
                    "C0": {"f3_false_yes_rate": ref},
                }
            }
        }
    }


class YesBiasGuardTest(unittest.TestCase):
    def test_level_point_passes(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.01, -0.01, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_margin_and_significance_both_bind(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.03, -0.005, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("> +0.02", reasons[0])
        _, reasons = m9_rules.yes_bias(
            pn1(0.015, 0.001, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("CI lower", reasons[0])

    def test_margin_edge_is_inclusive(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.02, -0.001, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_hop_and_hs1(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.0, -0.02, -0.031), hs1(0.21, 0.15), "X", "C0"
        )
        self.assertEqual([r[:2] for r in reasons], ["Y2", "Y3"])
        _, reasons = m9_rules.yes_bias(
            pn1(0.0, -0.02, -0.03), hs1(0.20, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_missing_readouts_fail(self):
        _, reasons = m9_rules.yes_bias(None, None, "X", "C0")
        self.assertEqual(len(reasons), 2)


class MultilingualTest(unittest.TestCase):
    def test_upper_bounds(self):
        mlx = {
            "metrics": {
                "noul_ml": {"diff": -0.01, "ci95": [-0.02, -0.001]},
                "choice_ml": {"diff": 0.0, "ci95": [-0.01, 0.01]},
                "score_ml": {"diff": -0.05, "ci95": [-0.08, -0.02]},
            }
        }
        _, reasons = m9_rules.multilingual(mlx)
        self.assertEqual(len(reasons), 1)
        self.assertIn("noul_ml", reasons[0])
        self.assertEqual(m9_rules.multilingual(None)[1], ["MLX-DEV-9B readout missing"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class Stage2BuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        x60 = self.tmp / "x60"
        x60.mkdir()
        write_jsonl(
            x60 / "train.jsonl",
            [
                {"id": "x1", "group_id": "gx", "input_sha256": "hx1"},
                {"id": "x2", "group_id": "gx", "input_sha256": "hx2"},
            ],
        )
        write_jsonl(x60 / "teacher.jsonl", [{"id": "x1"}, {"id": "x2"}])
        (x60 / "manifest.json").write_text(json.dumps({"recipe": {"tokens": 10}}))
        self.x60 = x60

    def ib(self, name, rows):
        path = self.tmp / f"{name}.train.jsonl"
        write_jsonl(path, rows)
        write_jsonl(
            self.tmp / f"{name}.train.tokens.jsonl",
            [{"id": r["id"], "native": 7} for r in rows],
        )
        return path

    def test_concatenates_excludes_and_keeps_x60_bytes(self):
        ib1 = self.ib(
            "ib1",
            [
                {"id": "a1", "group_id": "g1", "input_sha256": "h1", "family": "args"},
                {"id": "a2", "group_id": "g1", "input_sha256": "h2", "family": "isarc"},
            ],
        )
        ib2 = self.ib(
            "ib2",
            [{"id": "b1", "group_id": "g2", "input_sha256": "h3", "family": "hover"}],
        )
        out = self.tmp / "out"
        m9_data.main(
            [
                "--x60-dir",
                str(self.x60),
                "--ib1",
                str(ib1),
                "--ib2",
                str(ib2),
                "--exclude-family",
                "isarc",
                "--output",
                str(out),
            ]
        )
        lines = (out / "train.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(x)["id"] for x in lines], ["x1", "x2", "a1", "b1"])
        self.assertEqual(
            (out / "teacher.jsonl").read_bytes(),
            (self.x60 / "teacher.jsonl").read_bytes(),
        )
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["ib_rows_dropped"], {"ib1/isarc": 1})
        self.assertEqual(manifest["ib_native_tokens_kept"], 14)

    def test_refuses_x60_group_or_repeated_input(self):
        ib1 = self.ib(
            "ib1",
            [{"id": "a1", "group_id": "gx", "input_sha256": "h1", "family": "args"}],
        )
        ib2 = self.ib(
            "ib2",
            [{"id": "b1", "group_id": "g2", "input_sha256": "h3", "family": "hover"}],
        )
        with self.assertRaises(ValueError):
            m9_data.main(
                [
                    "--x60-dir",
                    str(self.x60),
                    "--ib1",
                    str(ib1),
                    "--ib2",
                    str(ib2),
                    "--output",
                    str(self.tmp / "o1"),
                ]
            )
        ib1 = self.ib(
            "ib1",
            [{"id": "a1", "group_id": "g1", "input_sha256": "h3", "family": "args"}],
        )
        with self.assertRaises(ValueError):
            m9_data.main(
                [
                    "--x60-dir",
                    str(self.x60),
                    "--ib1",
                    str(ib1),
                    "--ib2",
                    str(ib2),
                    "--output",
                    str(self.tmp / "o2"),
                ]
            )


class Stage3MatchedBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        x60 = self.tmp / "x60"
        x60.mkdir()
        rows, ids = [], []
        for g in range(20):
            for k in range(2):
                rid = f"x{g:02d}{k}"
                rows.append(
                    {
                        "id": rid,
                        "group_id": f"g{g}",
                        "input_sha256": f"h{rid}",
                        "source": "s1" if g < 10 else "s2",
                        "task_type": "noul" if g % 2 else "choice",
                        "language": "en",
                    }
                )
                ids.append(
                    {"id": rid, "source": rows[-1]["source"], "pool": "P", "native": 5}
                )
        write_jsonl(x60 / "train.jsonl", rows)
        write_jsonl(x60 / "teacher.jsonl", [{"id": r["id"], "p": 1} for r in rows])
        (x60 / "manifest.json").write_text(json.dumps({"recipe": {"tokens": 200}}))
        write_jsonl(self.tmp / "ids.jsonl", ids)
        self.x60 = x60
        self.ib1 = self.tmp / "ib1.train.jsonl"
        write_jsonl(
            self.ib1,
            [
                {
                    "id": "a1",
                    "group_id": "ga",
                    "input_sha256": "ha1",
                    "family": "args",
                    "task_type": "choice",
                },
                {
                    "id": "a2",
                    "group_id": "ga",
                    "input_sha256": "ha2",
                    "family": "isarc",
                    "task_type": "noul",
                },
            ],
        )
        self.ib2 = self.tmp / "ib2.train.jsonl"
        write_jsonl(
            self.ib2,
            [
                {
                    "id": "b1",
                    "group_id": "gb",
                    "input_sha256": "hb1",
                    "family": "hover",
                    "task_type": "noul",
                }
            ],
        )
        self.tokens(self.ib1, {"a1": 20, "a2": 20})
        self.tokens(self.ib2, {"b1": 20})

    def tokens(self, path, counts):
        write_jsonl(
            path.with_name(path.name.replace(".jsonl", ".tokens.jsonl")),
            [{"id": k, "native": v} for k, v in counts.items()],
        )

    def build(self, out, *extra):
        return m9_data.main(
            [
                "--x60-dir",
                str(self.x60),
                "--ib1",
                str(self.ib1),
                "--ib2",
                str(self.ib2),
                "--match-tokens",
                "200",
                "--x60-ids",
                str(self.tmp / "ids.jsonl"),
                "--keep-seed",
                "t:keep",
                "--keep-tolerance",
                "0.2",
                *extra,
                "--output",
                str(out),
            ]
        )

    def test_cuts_x60_to_the_matched_budget_in_whole_groups(self):
        out = self.tmp / "kib"
        self.build(out)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["ib_native_tokens_kept"], 60)
        self.assertEqual(manifest["x60_keep"]["budget_tokens"], 140)
        kept_tokens = manifest["x60_keep"]["native_tokens"]
        self.assertTrue(140 <= kept_tokens <= 168)
        self.assertEqual(manifest["train_native_tokens"], kept_tokens + 60)
        lines = (out / "train.jsonl").read_bytes().splitlines(keepends=True)
        x60_lines = (self.x60 / "train.jsonl").read_bytes().splitlines(keepends=True)
        kept = lines[: manifest["x60_rows_kept"]]
        self.assertEqual(kept, [x for x in x60_lines if x in set(kept)])
        groups = {json.loads(x)["group_id"] for x in kept}
        self.assertEqual(len(kept), 2 * len(groups))
        self.assertEqual(
            [json.loads(x)["id"] for x in lines[len(kept) :]], ["a1", "a2", "b1"]
        )
        teacher = (out / "teacher.jsonl").read_text().splitlines()
        self.assertEqual(
            [json.loads(t)["id"] for t in teacher], [json.loads(x)["id"] for x in kept]
        )

    def test_ablation_keeps_more_x60_and_nests(self):
        self.build(self.tmp / "kib")
        self.build(
            self.tmp / "kibx", "--exclude-family", "isarc", "--exclude-family", "hover"
        )
        a = json.loads((self.tmp / "kib" / "manifest.json").read_text())
        b = json.loads((self.tmp / "kibx" / "manifest.json").read_text())
        self.assertEqual(b["ib_native_tokens_kept"], 20)
        self.assertEqual(b["x60_keep"]["budget_tokens"], 180)
        ids = lambda d, n: {
            json.loads(x)["id"]
            for x in (self.tmp / d / "train.jsonl").read_text().splitlines()[:n]
        }
        self.assertTrue(
            ids("kib", a["x60_rows_kept"]) <= ids("kibx", b["x60_rows_kept"])
        )

    def test_cut_language_keeps_every_other_language_group_whole(self):
        rows = [
            json.loads(x) for x in (self.x60 / "train.jsonl").read_text().splitlines()
        ]
        for r in rows:
            if r["group_id"] in {"g0", "g1", "g2"} and r["id"].endswith("1"):
                r["language"] = "de"
        write_jsonl(self.x60 / "train.jsonl", rows)
        out = self.tmp / "ksw"
        self.build(out, "--cut-language", "en")
        manifest = json.loads((out / "manifest.json").read_text())
        keep = manifest["x60_keep"]
        self.assertEqual((keep["fixed_groups"], keep["fixed_rows"]), (3, 6))
        self.assertEqual(keep["fixed_native_tokens"], 30)
        self.assertEqual(keep["budget_tokens"], 110)
        self.assertTrue(140 <= keep["native_tokens"] <= 162)
        kept = (
            (out / "train.jsonl").read_text().splitlines()[: manifest["x60_rows_kept"]]
        )
        kept_ids = {json.loads(x)["id"] for x in kept}
        self.assertTrue({"x000", "x001", "x010", "x011", "x020", "x021"} <= kept_ids)
        self.assertEqual(manifest["train_native_tokens"], keep["native_tokens"] + 60)

    def test_refuses_missing_counts_and_missing_ids(self):
        self.tokens(self.ib2, {})
        with self.assertRaises(ValueError):
            self.build(self.tmp / "o1")
        self.tokens(self.ib2, {"b1": 20})
        write_jsonl(
            self.tmp / "ids.jsonl",
            [{"id": "x000", "source": "s1", "pool": "P", "native": 5}],
        )
        with self.assertRaises(ValueError):
            self.build(self.tmp / "o2")


if __name__ == "__main__":
    unittest.main()
