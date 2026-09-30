import collections
import unittest

from v2.data.hr2 import build
from v2.data.hr2 import families as fam

SCREEN = (set(), set())


def turn(text):
    return [{"role": "user", "content": text}]


def pref(prompt, first, second, overall, scores):
    return {
        "domain": "general",
        "language": "english",
        "context": turn(prompt),
        "response1": first,
        "response2": second,
        "overall_preference": overall,
        "individual_preference": [{"score": score} for score in scores],
    }


def feedback(prompt, levels_one, levels_two, first="r1", second="r2"):
    def texts(levels):
        return [f"The response is {level} helpful. More detail." for level in levels]

    return {
        "domain": "general",
        "language": "english",
        "context": turn(prompt),
        "response1": first,
        "response2": second,
        "feedback1": texts(levels_one),
        "feedback2": texts(levels_two),
    }


def step(*completions, chosen=None):
    return {
        "completions": [
            {"text": text, "rating": rating} for text, rating in completions
        ],
        "chosen_completion": chosen,
        "human_completion": None,
    }


def prm(problem, *steps):
    return {
        "question": {"problem": problem},
        "label": {"steps": list(steps), "finish_reason": "found_error"},
    }


class HelpSteer3Test(unittest.TestCase):
    def test_key_ignores_response_order(self):
        one = pref("q", "x", "y", -3, [-3, -2])
        two = pref("q", "y", "x", 3, [3, 2])
        self.assertEqual(fam.hs3_key(one), fam.hs3_key(two))

    def test_pref_keeps_agreeing_repeats_once_and_drops_every_conflicting_copy(self):
        records = [
            pref("q1", "good!", "bad", -3, [-3, -2]),
            pref("q1", "bad", "good!", 2, [2, 3]),
            pref("q2", "p", "q", -2, [-2, -3]),
            pref("q2", "p", "q", 1, [1, 1]),
            pref("q3", "m", "n", -3, [-3, 1]),
            pref("q4", "ok", "longer", -2, [-2, -2]),
        ]
        report = collections.Counter()
        rows = fam.hs3_pref(records, SCREEN, report)
        self.assertEqual(report["drop_exact_duplicates"], 1)
        self.assertEqual(report["drop_conflicting_duplicates"], 1)
        self.assertEqual(report["drop_tie_or_slight"], 1)
        self.assertEqual(report["drop_annotator_split"], 1)
        self.assertEqual(len(rows), 2)
        for row in rows:
            gold = row["state"]["response_a" if row["label"] == 0 else "response_b"]
            self.assertIn(gold, ("good!", "ok"))

    def test_help_requires_three_parsed_close_levels_agreeing_across_copies(self):
        self.assertEqual(fam.help_levels(["The response is mostly helpful."]), [3])
        self.assertIsNone(fam.help_levels(["Helpful overall."]))
        records = [
            feedback("a", ["mostly", "mostly", "perfectly"], ["mostly"] * 3),
            feedback(
                "b", ["not", "mostly", "perfectly"], ["not", "mostly", "perfectly"]
            ),
            feedback("c", ["mostly"] * 3, ["mostly"] * 3),
            feedback("c", ["slightly"] * 3, ["slightly"] * 3),
            feedback("d", ["partially"] * 2, ["partially"] * 2),
        ]
        report = collections.Counter()
        fam.hs3_help(records, SCREEN, report)
        self.assertEqual(report["eligible"], 1)
        self.assertEqual(report["drop_disagreement"], 1)
        self.assertEqual(report["drop_conflicting_duplicates"], 2)
        self.assertEqual(report["drop_unparsed_or_not_three"], 1)


class PrmTest(unittest.TestCase):
    def test_walk_records_alternative_ratings(self):
        record = prm("p", step(("a", 1), ("b", -1), chosen=0), step(("c", -1)))
        yes, no, rated = fam.prm_walk(record)
        self.assertEqual(yes, [(0, [], "a")])
        self.assertEqual(no, (1, ["a"], "c"))
        self.assertIn(((0, [], "b"), -1), rated)

    def test_conflicting_ratings_of_one_state_drop_the_row(self):
        records = [
            prm("p1", step(("s1", 1)), step(("s2", -1))),
            prm("p1", step(("s1", 0))),
            prm("p2", step(("a", 1), ("b", -1), chosen=0), step(("c", -1))),
            prm("p2", step(("x", 1), ("a", 0), chosen=0)),
        ]
        report = collections.Counter()
        fam.prm_step(records, report)
        self.assertEqual(report["drop_conflicting_ratings"], 2)
        self.assertEqual(report["eligible"], 3)


class CommonRulesTest(unittest.TestCase):
    def test_resolve_and_dedup(self):
        row = {"id": "x", "input_sha256": "h1", "label": 0, "family": "f"}
        report = collections.Counter()
        kept = fam.resolve([row, dict(row), dict(row, id="y", label=1)], report)
        self.assertEqual(sorted(r["id"] for r in kept), ["x", "y"])
        self.assertEqual(report["drop_exact_duplicates"], 1)
        kept = fam.resolve([row, dict(row, label=1)], collections.Counter())
        self.assertEqual(kept, [])
        rows = [
            row,
            dict(row),
            dict(row, id="w", family="g"),
            dict(row, id="y", input_sha256="h2", label=1),
            dict(row, id="z", input_sha256="h2", family="g"),
        ]
        unique, dup = build.dedup(rows)
        self.assertEqual([r["id"] for r in unique], ["w"])
        self.assertEqual(dup["conflicting"], {"f": 1, "g": 1})

    def test_dev_slice_is_deterministic_and_about_a_tenth(self):
        groups = [f"hr2:src:{i}" for i in range(20000)]
        share = sum(build.is_dev(g) for g in groups) / len(groups)
        self.assertTrue(0.09 < share < 0.11, share)
        self.assertEqual(
            [build.is_dev(g) for g in groups[:50]],
            [build.is_dev(g) for g in groups[:50]],
        )

    def test_balancers(self):
        rows = [
            {"id": str(i), "label": int(i < 70), "audit_metadata": {"hr2": {}}}
            for i in range(100)
        ]
        for row in rows:
            row["audit_metadata"]["hr2"]["hash_key"] = row["id"]
            row["audit_metadata"]["hr2"]["cell"] = str(
                row["label"] + int(row["id"]) % 3
            )
        kept = fam.balance_yes_no(rows, None, "t")
        self.assertEqual(collections.Counter(r["label"] for r in kept), {0: 30, 1: 30})
        capped = fam.cap_share(rows, 100, 0.30, "t")
        cells = collections.Counter(r["audit_metadata"]["hr2"]["cell"] for r in capped)
        self.assertLessEqual(max(cells.values()), 0.30 * len(capped))


if __name__ == "__main__":
    unittest.main()
