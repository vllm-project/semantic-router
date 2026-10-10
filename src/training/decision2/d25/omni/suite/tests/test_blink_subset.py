"""BLINK subset screening on the board's values and recovery of a hidden subset from synthetic models."""

import random
import unittest

from d25.omni.suite import blink_subset as bs
from d25.omni.suite import lattice


class ScreenTest(unittest.TestCase):
    def test_43_candidates_and_named_sets(self):
        pool = bs.candidates()
        self.assertEqual(len(pool), 43)
        for name, members in bs.NAMED.items():
            self.assertIn(tuple(sorted(members)), pool, name)
            self.assertEqual(bs.name_of(members), name)

    def test_board_lattice_and_single_image_engines_leave_e_and_f(self):
        board = lattice.board_values()
        single = [-2.13, -4.55]
        screened = bs.screen(board["BLINK"], single)
        survivors = sorted(
            c["name"] for c in screened if c["lattice_ok"] and c["single_image_ok"]
        )
        self.assertEqual(survivors, ["E", "F"])
        by_name = {c["name"]: c for c in screened}
        for name in "ABC":
            self.assertFalse(by_name[name]["lattice_ok"], name)
        self.assertAlmostEqual(by_name["A"]["chance_sum"], 339.75)
        self.assertTrue(by_name["D"]["lattice_ok"])
        self.assertFalse(by_name["D"]["single_image_ok"])
        self.assertEqual(by_name["E"]["single_image_rows"], 387)
        self.assertIn(bs.DEFAULT, survivors)


class FitTest(unittest.TestCase):
    def synthetic(self, hidden, n_models=8, seed=7):
        rng = random.Random(seed)
        models, official = {}, {}
        for m in range(n_models):
            level = rng.uniform(0.35, 0.85)
            per = {}
            for s, (rows, options, _) in bs.SUBTASKS.items():
                p = min(0.98, max(1 / options, level + rng.uniform(-0.25, 0.25)))
                per[s] = (sum(rng.random() < p for _ in range(rows)), rows)
            models[f"m{m}"] = per
            official[f"m{m}"] = round(bs.predicted_skill(per, bs.NAMED[hidden]), 2)
        return models, official

    def test_fit_recovers_the_hidden_subset(self):
        for hidden in ("E", "F", "A"):
            models, official = self.synthetic(hidden)
            ranking = bs.fit(models, official)
            self.assertEqual(ranking[0]["name"], hidden)
            self.assertLess(ranking[0]["rmse"], 0.01)
            self.assertGreater(ranking[1]["rmse"], 0.3)

    def test_fit_tolerates_reproduction_noise(self):
        models, official = self.synthetic("E", n_models=10, seed=11)
        rng = random.Random(3)
        noisy = {m: v + rng.gauss(0, 0.5) for m, v in official.items()}
        ranking = bs.fit(models, noisy, [tuple(bs.NAMED[k]) for k in "DEF"])
        self.assertEqual(ranking[0]["name"], "E")

    def test_row_count_mismatch_is_rejected(self):
        models, official = self.synthetic("E", n_models=2)
        models["m0"]["Counting"] = (10, 119)
        with self.assertRaises(ValueError):
            bs.fit(models, official)


if __name__ == "__main__":
    unittest.main()
