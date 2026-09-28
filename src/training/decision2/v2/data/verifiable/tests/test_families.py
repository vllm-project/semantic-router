import collections
import unittest

from v2.data.verifiable import core
from v2.data.verifiable.build import (
    A2_FAMILIES,
    A6_FAMILIES,
    BuildStats,
    a2_task,
    a4_rows,
    counterfactual_rows,
)
from v2.data.verifiable.tests import checks

SEED = "unit-test-families"
MIN_ROWS = 300


class CounterfactualFamilyTest(unittest.TestCase):
    def _run(self, arm, module, lang, task_for, levels_for):
        stats = BuildStats()
        rows, index = [], 0
        while len(rows) < MIN_ROWS:
            rows += counterfactual_rows(
                arm,
                module,
                lang,
                index,
                SEED,
                task_for(index),
                levels_for(index),
                stats,
            )
            index += 1
        self.assertEqual(
            stats.disagreements, 0, f"{module.FAMILY}/{lang} oracle disagreements"
        )
        self.assertEqual(stats.dropped_rows, 0)
        for row in rows:
            checks.assert_row(self, row)
        checks.assert_counterfactual_groups(self, rows)
        return rows

    def test_a2_families_agree_on_every_row(self):
        for module in A2_FAMILIES:
            for lang in core.LANGS:
                with self.subTest(family=module.FAMILY, lang=lang):
                    rows = self._run("a2", module, lang, a2_task, lambda _: None)
                    tasks = collections.Counter(r["task_type"] for r in rows)
                    self.assertEqual(set(tasks), {"choice", "noul"})
                    noul = [r["label"] for r in rows if r["task_type"] == "noul"]
                    self.assertEqual(sum(noul) * 2, len(noul))

    def test_a6_families_agree_on_every_row(self):
        for module in A6_FAMILIES:
            for lang in core.LANGS:
                with self.subTest(family=module.FAMILY, lang=lang):
                    if module.FAMILY == "a6_evidence_status":
                        levels_for = lambda _: 3
                    else:
                        levels_for = lambda i: 2 + i % 9
                    rows = self._run("a6", module, lang, lambda _: "score", levels_for)
                    for row in rows:
                        levels = len(row["options"])
                        self.assertEqual(
                            [o["key"] for o in row["options"]],
                            [str(k) for k in range(levels)],
                        )
                        self.assertEqual(row["audit_metadata"]["levels"], levels)

    def test_evidence_status_is_three_level_only(self):
        with self.assertRaises(ValueError):
            A6_FAMILIES[-1].build_group(core.make_rng("x"), "en", "score", 4)


class HardNegativeFamilyTest(unittest.TestCase):
    def test_a4_scenarios_agree_for_near_and_random_distractors(self):
        for module in A2_FAMILIES:
            for lang in core.LANGS:
                with self.subTest(family=module.FAMILY, lang=lang):
                    stats, counters = BuildStats(), collections.Counter()
                    hard, rand = [], []
                    for index in range(80):
                        h, r = a4_rows(module, lang, index, SEED, counters, stats)
                        hard += h
                        rand += r
                    self.assertEqual(stats.disagreements, 0)
                    self.assertEqual(stats.identical_a4_distractors, 0)
                    self.assertEqual(len(hard), 160)
                    for row in hard + rand:
                        checks.assert_row(self, row)


class CoreHelperTest(unittest.TestCase):
    def test_rng_is_keyed_by_every_part(self):
        a = core.make_rng("seed", "fam", "en", 1).random()
        self.assertEqual(a, core.make_rng("seed", "fam", "en", 1).random())
        self.assertNotEqual(a, core.make_rng("seed", "fam", "zh", 1).random())
        self.assertNotEqual(a, core.make_rng("seed", "fam", "en", 2).random())

    def test_language_interleave_hits_share(self):
        from fractions import Fraction

        share = Fraction(3, 10)
        zh = sum(core.is_zh(i, share) for i in range(1000))
        self.assertEqual(zh, 300)

    def test_presence_rule(self):
        options = core.choice_options(["5 crates", "6 crates"])
        self.assertTrue(core.presence_ok("nothing here", options))
        self.assertFalse(core.presence_ok("we counted 5 crates", options))
        self.assertTrue(core.presence_ok("15 crates", options))


if __name__ == "__main__":
    unittest.main()
