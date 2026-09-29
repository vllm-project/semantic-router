import contextlib
import importlib
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

rules = importlib.import_module("v2.27b.m4b.m4b_rules")
contrast = importlib.import_module("v2.27b.contrast")
fixture = importlib.import_module("v2.27b.tests.test_m3_contrast")

THIRD, HALF, TWO_THIRDS, ONE = (
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(1),
)


def metrics(types=(80, 40, 40), families=(9, 9), h3=0.60, t=None):
    """Rule metrics on n_t = 100 per type and two families of 10 questions."""
    fam = {"fa": [families[0], 10], "fb": [families[1], 10]}
    return {
        "types": {k: [c, 100] for k, c in zip(("choice", "noul", "score"), types)},
        "families": fam,
        "H3": h3,
        "T": (
            Fraction(t)
            if t is not None
            else sum(Fraction(c, n) for c, n in fam.values()) / 2
        ),
    }


REF = metrics(families=(8, 8), t=Fraction(80, 100))


class AlphaRuleTest(unittest.TestCase):
    def line(self, **gains):
        """alpha -> eligible metrics with T = F1's T + gain (None: no readout)."""
        out = {}
        for key, alpha in (
            ("a13", THIRD),
            ("a12", HALF),
            ("a23", TWO_THIRDS),
            ("a1", ONE),
        ):
            gain = gains.get(key)
            out[alpha] = None if gain is None else metrics(t=REF["T"] + Fraction(gain))
        return out

    def test_picks_the_smallest_alpha_within_three_quarters_of_g_star(self):
        got = rules.alpha_rule(
            REF, self.line(a13="0.02", a12="0.035", a23="0.04", a1="0.03")
        )
        self.assertTrue(got["pick"])
        self.assertEqual(got["G_star"], 0.04)
        self.assertEqual(got["alpha_star"], "1/2")

    def test_exact_three_quarters_qualifies(self):
        got = rules.alpha_rule(
            REF, self.line(a13="0.03", a12="0.01", a23="0.01", a1="0.04")
        )
        self.assertEqual(got["alpha_star"], "1/3")

    def test_no_alpha_eligible(self):
        bad = metrics(h3=0.59)
        got = rules.alpha_rule(REF, {THIRD: bad, HALF: bad, TWO_THIRDS: None, ONE: bad})
        self.assertFalse(got["pick"])
        self.assertEqual(got["reason"], "no alpha is eligible")
        self.assertFalse(got["alphas"]["2/3"]["readout"])

    def test_g_star_below_one_point(self):
        got = rules.alpha_rule(REF, self.line(a13="0.005", a12="0.009", a1="0.0099"))
        self.assertFalse(got["pick"])
        self.assertEqual(got["reason"], "G* < 0.01")

    def test_g_star_of_exactly_one_point_is_enough(self):
        got = rules.alpha_rule(REF, self.line(a13="0.01"))
        self.assertEqual(got["alpha_star"], "1/3")

    def test_only_alpha_one_qualifies(self):
        got = rules.alpha_rule(
            REF, self.line(a13="0.01", a12="0.02", a23="0.029", a1="0.04")
        )
        self.assertFalse(got["pick"])
        self.assertEqual(got["reason"], "only alpha = 1 qualifies")
        got = rules.alpha_rule(
            REF,
            {
                THIRD: None,
                HALF: None,
                TWO_THIRDS: None,
                ONE: metrics(t=Fraction(9, 10)),
            },
        )
        self.assertEqual(got["reason"], "only alpha = 1 qualifies")

    def test_ineligible_alpha_does_not_set_g_star_or_pick(self):
        line = self.line(a13="0.02", a1="0.02")
        line[HALF] = {**metrics(types=(80, 40, 36)), "T": REF["T"] + Fraction(1, 10)}
        got = rules.alpha_rule(REF, line)
        self.assertEqual(got["alphas"]["1/2"]["failed_types"], ["score"])
        self.assertEqual(got["G_star"], 0.02)
        self.assertEqual(got["alpha_star"], "1/3")

    def test_eligibility_boundaries_are_exact(self):
        self.assertTrue(
            rules.eligibility(REF, metrics(types=(77, 37, 37), families=(7, 7)))[
                "eligible"
            ]
        )
        for cand, failed in (
            (metrics(types=(76, 40, 40)), "types_within_0.03n"),
            (metrics(families=(6, 9)), "families_within_0.10"),
            (metrics(h3=0.5999999), "h3_at_least_f1"),
        ):
            check = rules.eligibility(REF, cand)
            self.assertFalse(check["eligible"])
            self.assertEqual(
                [k for k, ok in check["checks"].items() if not ok], [failed]
            )
        self.assertEqual(
            rules.eligibility(REF, metrics(families=(6, 9)))["failed_families"], ["fa"]
        )


class FinalistTest(unittest.TestCase):
    ORDER = [("A2", "A2-soup"), ("A1", "A1-s2"), ("theta", "T-13"), ("A3", "A3-s1")]

    def names(self, order, usable):
        return [(f["slot"], f["candidate"]) for f in rules.finalist_list(order, usable)]

    def test_priority_and_cap(self):
        got = self.names(self.ORDER, {"A2-soup", "A1-s2", "T-13", "A3-s1"})
        self.assertEqual(got, [("F-a", "A2-soup"), ("F-b", "A1-s2"), ("F-c", "T-13")])

    def test_no_pick_gives_a3(self):
        order = [*self.ORDER[:2], ("theta", None), self.ORDER[3]]
        got = self.names(order, {"A2-soup", "A1-s2", "A3-s1"})
        self.assertEqual(got, [("F-a", "A2-soup"), ("F-b", "A1-s2"), ("F-c", "A3-s1")])

    def test_dropped_or_failed_slots_pass_on(self):
        got = self.names(self.ORDER, {"A1-s2", "T-13", "A3-s1"})
        self.assertEqual(got, [("F-a", "A1-s2"), ("F-b", "T-13"), ("F-c", "A3-s1")])
        got = self.names(
            [self.ORDER[0], ("A1", None), ("theta", None), ("A3", "A3-s1")], {"A2-soup"}
        )
        self.assertEqual(got, [("F-a", "A2-soup")])


class RulesIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.world = fixture.World(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **overrides):
        w = self.world
        spec = {
            "F1": dict(dev_wrong=("d2",)),
            "F1M": dict(dev_wrong=("d2",)),
            "A1-s1": dict(dev_wrong=("d2",), best=5, select=(0.80, 0.1)),
            "A1-s2": dict(dev_wrong=("d2",), best=6, select=(0.82, 0.1)),
            "A1-soup": dict(dev_wrong=("d0", "d2")),
            "A2-s1": dict(best=5, select=(0.85, 0.1)),
            "A2-s2": dict(dev_wrong=("d4",), best=5, select=(0.84, 0.1)),
            "A2-soup": dict(),
            "A3-s1": dict(dev_wrong=("d0", "d1", "d2", "d3", "d4"), best=7),
            "T-13": dict(),
            "T-12": dict(),
            "T-23": dict(),
        }
        spec.update(overrides)
        return {name: w.candidate(name, **kw) for name, kw in spec.items()}

    def run_rules(self, dirs, extra=(), draws=40):
        f1 = dirs.pop("F1")
        argv = [
            "rules",
            *self.world.args(dirs, draws=draws),
            "--f1",
            str(f1),
            "--f1m",
            "F1M",
        ]
        argv += [
            "--arm",
            "A1=A1-soup:A1-s1+A1-s2",
            "--arm",
            "A2=A2-soup:A2-s1+A2-s2",
            "--arm",
            "A3=A3-s1",
        ]
        argv += [
            "--line-alpha",
            "1/3=T-13",
            "--line-alpha",
            "1/2=T-12",
            "--line-alpha",
            "2/3=T-23",
            *extra,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            rules.main(argv)
        return json.loads((self.root / "out.json").read_text(encoding="utf-8"))

    def test_end_to_end(self):
        out = self.run_rules(self.build())
        self.assertEqual(out["soup_rule"]["A1"]["artifact"], "A1-s2")
        self.assertEqual(out["soup_rule"]["A2"]["artifact"], "A2-soup")
        self.assertEqual(out["soup_rule"]["A3"]["artifact"], "A3-s1")
        self.assertFalse(out["proxy_screen"]["candidates"]["A3-s1"]["kept"])
        self.assertNotIn("F1M", out["proxy_screen"]["pool"])
        self.assertTrue(out["line"]["f1m_agreement"]["passes"])
        self.assertEqual(out["line"]["f1m_agreement"]["rate"], 1.0)
        self.assertEqual(out["line"]["S"], "A2-soup")
        self.assertEqual(out["line"]["alpha_rule"]["alpha_star"], "1/3")
        slots = [(f["slot"], f["candidate"]) for f in out["finalists"]["slots"]]
        self.assertEqual(slots, [("F-a", "A2-soup"), ("F-b", "A1-s2"), ("F-c", "T-13")])
        self.assertEqual(
            sorted(out["contrasts"]),
            sorted(
                [
                    "A2-s1:A1-s1",
                    "A2-s2:A1-s2",
                    "A2-s1+A2-s2:A1-s1+A1-s2",
                    "A1-s1:F1",
                    "A1-s2:F1",
                    "A1-s1+A1-s2:F1",
                    "A3-s1:A2-s1",
                ]
            ),
        )
        entry = out["contrasts"]["A2-s1:A1-s1"]
        for key in (
            "T_dev",
            "choice_accuracy",
            "noul_accuracy",
            "score_accuracy",
            "H_pilot",
            "H3",
            "P_dev",
            "cal_brier",
        ):
            self.assertIn(key, entry["bootstrap"])
            self.assertIn(key, entry["delta"])
        self.assertAlmostEqual(entry["delta"]["score_accuracy"], 0.5)
        self.assertIn("all_pass", out["retention_floors_vs_F1"]["A2-soup"])
        self.assertEqual(out["rule_metrics"]["F1"]["types"]["score"], [1, 2])

    def test_f1m_disagreement_stops_the_line_and_a3_takes_f_c(self):
        dirs = self.build(
            **{"F1M": dict(dev_wrong=("d2", "d0")), "A3-s1": dict(best=7)}
        )
        out = self.run_rules(dirs)
        self.assertFalse(out["line"]["f1m_agreement"]["passes"])
        self.assertFalse(out["line"]["alpha_rule"]["pick"])
        slots = [f["candidate"] for f in out["finalists"]["slots"]]
        self.assertEqual(slots, ["A2-soup", "A1-s2", "A3-s1"])

    def test_failed_soup_and_seed(self):
        dirs = self.build()
        for name in ("A2-soup", "A1-s2"):
            dirs.pop(name)
        out = self.run_rules(
            dirs, ["--failed", "A2-soup", "--failed", "A1-s2", "--contrast", "A2-s1:F1"]
        )
        self.assertEqual(out["soup_rule"]["A2"]["artifact"], "A2-s1")
        self.assertIn("soup failed", out["soup_rule"]["A2"]["reason"])
        self.assertEqual(out["soup_rule"]["A1"]["artifact"], "A1-s1")
        self.assertEqual(out["line"]["S"], "A2-s1")
        self.assertEqual(list(out["contrasts"]), ["A2-s1:F1"])

    def test_paired_matches_contrast_paired(self):
        w = self.world
        panels = contrast.Panels(w.dev_gold, w.css_gold)
        dirs = self.build()
        cal = fixture.m3.CalRows(w.cal)
        aho = {"A7": w.aho}
        inputs = fixture.m3.Inputs()
        left = [fixture.m3.Candidate("A2-s1", dirs["A2-s1"], panels, cal, aho, inputs)]
        right = [fixture.m3.Candidate("A1-s1", dirs["A1-s1"], panels, cal, aho, inputs)]
        ours = rules.paired(panels, left, right, 60, 7)
        theirs = contrast.paired(panels, left, right, 60, 7)
        self.assertEqual({k: v for k, v in ours.items() if k != "H3"}, theirs)
        self.assertIn("H3", ours)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def paired(self, name, v3, delta, ci, h):
        return self.write(name, {
            "point": {"left": {"score": v3}, "delta": {"score": delta, "H": 0.0, "T": 0.0}},
            "ci95": {"low": ci[0], "high": ci[1]},
            "axis_ci95": {"H": {"delta": {"low": h[0], "high": h[1]}}},
        })  # fmt: skip

    def verdicts(self, f1_ci=(0.5, 3.0), aj=(73.0, (0.1, 2.0), (-0.01, 0.02)), collapsed=False,
                 groups=(), mlx=None):  # fmt: skip
        args = [
            "verdicts", "--label", "DEV2.0-27B (M4b F-a)",
            "--paired-f1", str(self.paired("f1.json", aj[0], 1.0, f1_ci, (-0.01, 0.03))),
            "--paired-peer", f"AutoJev-27B={self.paired('aj.json', aj[0], 0.9, aj[1], aj[2])}",
            "--paired-peer", f"Eikos-27B={self.paired('ei.json', aj[0], 3.0, (1, 5), (0.0, 0.1))}",
            "--paired-peer", f"Jebadiah={self.paired('je.json', aj[0], 4.0, (2, 6), (-0.2, 0.0))}",
            "--types", str(self.write("types.json", {"types": {
                "choice": {"verdict": "OK"}, "noul": {"verdict": "COLLAPSED: x" if collapsed else "OK"},
                "score": {"verdict": "OK"}}})),
            "--exposure", str(self.write("exp.json", {"groups": list(groups), "matched_rows": {}})),
            "--report", str(self.write("report.json", {"panels": {"public231": {"correct": 200, "items": 231}}})),
            "--output", str(self.root / "verdicts.json"),
        ]  # fmt: skip
        if mlx is not None:
            key = "DEV2.0-27B (M4b F-a) - DEV2.0-27B (F1)"
            args += [
                "--mlx-overlap",
                str(
                    self.write(
                        "ov.json", {"pairs": {key: {"mlx": {"full": {"ci95": mlx}}}}}
                    )
                ),
            ]
        (self.root / "verdicts.json").unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rules.main(args)
        return json.loads((self.root / "verdicts.json").read_text(encoding="utf-8"))

    def test_pending_without_mlx_then_pass(self):
        out = self.verdicts()
        items = out["successor_rule"]["items"]
        self.assertEqual(
            items["4_mlx_diag_not_significantly_below_F1"]["status"], "PENDING"
        )
        self.assertEqual(out["successor_rule"]["verdict"], "PENDING")
        self.assertEqual(items["7_public231_reported"]["status"], "REPORTED")
        self.assertEqual(out["beats_autojev"]["verdict"], "PASS")
        out = self.verdicts(mlx=[-0.01, 0.02])
        self.assertEqual(out["successor_rule"]["verdict"], "PASS")

    def test_failures(self):
        out = self.verdicts(f1_ci=(-0.1, 2.0), mlx=[-0.03, -0.001])
        items = out["successor_rule"]["items"]
        self.assertEqual(items["1_v3_ci_lower_vs_F1_above_0"]["status"], "FAIL")
        self.assertEqual(
            items["4_mlx_diag_not_significantly_below_F1"]["status"], "FAIL"
        )
        self.assertEqual(out["successor_rule"]["verdict"], "FAIL")
        out = self.verdicts(
            collapsed=True, groups=["g"], aj=(72.133, (0.1, 2.0), (-0.03, -0.001))
        )
        items = out["successor_rule"]["items"]
        self.assertEqual(items["3_no_type_collapsed"]["status"], "FAIL")
        self.assertEqual(items["5_no_1_0_tier_gates"]["status"], "FAIL")
        self.assertEqual(items["6_no_overlap_exposure"]["status"], "FAIL")
        beats = out["beats_autojev"]["items"]
        self.assertEqual(beats["v3_above_72.133"]["status"], "FAIL")
        self.assertEqual(beats["H_not_significantly_below_AutoJev"]["status"], "FAIL")
        self.assertEqual(out["beats_autojev"]["verdict"], "FAIL")


class OverlapSpecTest(unittest.TestCase):
    def test_spec_from_template(self):
        committed = json.loads(
            (Path(rules.__file__).parent / "overlap-spec-27b-m4b.json").read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = json.loads(json.dumps(committed))
            for slot in ("F-a", "F-b", "F-c"):
                template["m4b"][slot]["run"] = str(root / slot / "formal")
            for slot in ("F-a", "F-b"):
                run = root / slot / "formal"
                run.mkdir(parents=True)
                (run / "SEAL.json").write_text("{}")
            (root / "F-b" / "formal" / "PAIRED-vs-AutoJev-27B.json").write_text("{}")
            spec = rules.overlap_spec(template, "F-b", root / "exp.json")
            label = template["m4b"]["F-b"]["label"]
            self.assertEqual(spec["tiers"]["27B"]["candidate"], label)
            self.assertIn(
                template["m4b"]["F-a"]["label"], spec["tiers"]["27B"]["internal_peers"]
            )
            self.assertNotIn(template["m4b"]["F-c"]["label"], spec["models"])
            self.assertEqual(
                spec["models"][label]["exposure"], [str(root / "exp.json")]
            )
            self.assertNotIn(
                "exposure", spec["models"][template["m4b"]["F-a"]["label"]]
            )
            self.assertEqual([r["right"] for r in spec["reproduce"]], ["AutoJev-27B"])
            for peer in (
                spec["tiers"]["27B"]["peers"] + spec["tiers"]["27B"]["internal_peers"]
            ):
                self.assertIn(peer, spec["models"])
            with self.assertRaises(ValueError):
                rules.overlap_spec(template, "F-c", root / "exp.json")


class SummaryTest(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tasks = {"discourse": 0.6, "implicit_hate": 0.7, "semeval_stance": 0.5}
            readout = {
                "label": "x",
                "typed_dev": {"T_dev": 0.81, "by_type": {"choice": {"correct": 8, "n": 10}},
                              "by_family": {"f": 0.81}, "invalid_or_missing": 0},
                "css_pilot": {"H_pilot": 0.6, "tasks": tasks, "invalid_or_missing": 1},
                "development_proxy": 100 * (0.81 * 0.6) ** 0.5,
                "select": {"n": 700, "correct": 600, "family_macro_accuracy": 0.8, "family_macro_brier": 0.2},
            }  # fmt: skip
            (d / "READOUT.json").write_text(json.dumps(readout))
            out = rules.summary(d, "/ckpt")
            self.assertAlmostEqual(out["H3"], 0.6)
            self.assertEqual(out["select700"]["correct"], 600)
            self.assertIsNone(out["cal698"])
            readout["development_proxy"] += 0.1
            (d / "READOUT.json").write_text(json.dumps(readout))
            with self.assertRaises(ValueError):
                rules.summary(d)


if __name__ == "__main__":
    unittest.main()
