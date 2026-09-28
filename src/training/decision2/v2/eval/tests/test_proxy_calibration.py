from __future__ import annotations

import math
import unittest

from v2.eval import proxy_calibration as pc


def model(key, tier, v3, group="comparator", track="eval", **values):
    base = {
        "T_dev": 0.5,
        "H_pilot": 0.3,
        "H_mean3": 0.3,
        "dev_choice": 0.5,
        "dev_noul": 0.5,
        "dev_score": 0.5,
        "pilot_micro": 0.4,
    }
    base.update(values)
    t, h, h3 = base["T_dev"], base["H_pilot"], base["H_mean3"]
    c, n, s = base["dev_choice"], base["dev_noul"], base["dev_score"]
    base.update(
        {
            "P": 100 * math.sqrt(t * h),
            "P_mean3": 100 * math.sqrt(t * h3),
            "P_type": 100 * math.sqrt((c + n + s) / 3 * h),
            "P_type_mean3": 100 * math.sqrt((c + n + s) / 3 * h3),
            "P_CS_mean3": 100 * math.sqrt((c + s) / 2 * h3),
            "A_med": 100 * (t + h) / 2,
            "A_mean3": 100 * (t + h3) / 2,
        }
    )
    return {
        "key": key,
        "tier": tier,
        "v3": v3,
        "group": group,
        "track": track,
        "features": base,
        "noise_sd": {k: 0.01 for k in base},
    }


class FeaturesTest(unittest.TestCase):
    def test_median_and_mean_of_three_pilot_tasks(self):
        typed = [("f1", "g1", {"choice": (1, 2)}), ("f2", "g2", {"score": (1, 1)})]
        labels = {"a": ["x", "y"], "b": ["x", "y"], "c": ["x", "y"]}
        tasks = {
            "a": [("x", "x"), ("y", "y")],  # F1 1.0
            "b": [("x", "y"), ("y", "x")],  # F1 0.0
            "c": [("x", "x"), ("y", "x")],  # F1 (2/3 + 0) / 2
        }
        f = pc.features(typed, tasks, labels)
        self.assertAlmostEqual(f["T_dev"], 0.75)
        self.assertAlmostEqual(f["H_pilot"], 1 / 3)
        self.assertAlmostEqual(f["H_mean3"], (1 + 0 + 1 / 3) / 3)
        self.assertAlmostEqual(f["P"], 100 * math.sqrt(0.75 / 3))
        self.assertAlmostEqual(f["P_mean3"], 100 * math.sqrt(0.75 * f["H_mean3"]))
        self.assertAlmostEqual(f["A_mean3"], 100 * (0.75 + f["H_mean3"]) / 2)
        self.assertAlmostEqual(f["P_CS_mean3"], 100 * math.sqrt(0.75 * f["H_mean3"]))
        self.assertAlmostEqual(f["pilot_b"], 0.0)


class FitTest(unittest.TestCase):
    def test_ols_recovers_exact_plane_and_loo_is_exact(self):
        a = [0.1, 0.4, 0.2, 0.9, 0.5, 0.7]
        b = [0.3, 0.1, 0.8, 0.2, 0.6, 0.4]
        y = [1 + 2 * u + 3 * v for u, v in zip(a, b)]
        beta = pc.ols([a, b], y)
        for got, want in zip(beta, (1, 2, 3)):
            self.assertAlmostEqual(got, want)
        for got, want in zip(pc.loo_predictions([a, b], y), y):
            self.assertAlmostEqual(got, want)

    def test_within_tier_r_removes_tier_offsets(self):
        tiers = ["a", "a", "b", "b"]
        x = [1.0, 2.0, 11.0, 12.0]
        y = [2.0, 1.0, 12.0, 11.0]
        self.assertAlmostEqual(pc.within_tier_r(x, y, tiers), -1.0)


class PairsTest(unittest.TestCase):
    def test_pair_classes_and_strata(self):
        models = [
            model("c1", "4B", 50.0, "candidate", "dec"),
            model("c2", "4B", 53.0, "candidate", "dec"),
            model("r1", "4B", 60.0),
            model("r2", "4B", 61.0),
            model("c3", "9B", 70.0, "candidate", "9b"),
        ]
        pairs = {(p["i"], p["j"]): p for p in pc.model_pairs(models)}
        self.assertTrue(pairs[(0, 1)]["finalist"])
        self.assertEqual(pairs[(0, 1)]["stratum"], "close_2to5")
        self.assertTrue(pairs[(0, 2)]["decision"])
        self.assertFalse(pairs[(0, 2)]["finalist"])
        self.assertFalse(pairs[(2, 3)]["decision"])
        self.assertEqual(pairs[(2, 3)]["stratum"], "tie_lt2")
        self.assertFalse(pairs[(0, 4)]["same_tier"])

    def test_tie_band_picks_smallest_stable_gap(self):
        pairs, x = [], []
        # 40 pairs with gap 1 (half reversed by 3 points), 40 with gap 6 (all agree)
        for k in range(80):
            gap = 1.0 if k < 40 else 6.0
            dy = -3.0 if (k < 40 and k % 2) else 3.0
            x.extend([gap, 0.0])
            pairs.append({"i": 2 * k, "j": 2 * k + 1, "dy": dy})
        band = pc.tie_band(pairs, x)
        self.assertEqual(band["band"], 2)
        self.assertGreater(band["model_check"]["beta_within"], 0)

    def test_tie_band_none_when_proxy_has_no_signal(self):
        pairs, x = [], []
        for k in range(60):
            x.extend([float(k % 7 + 1), 0.0])
            pairs.append({"i": 2 * k, "j": 2 * k + 1, "dy": 3.0 if k % 2 else -3.0})
        self.assertIsNone(pc.tie_band(pairs, x)["band"])


class AnalyzeTest(unittest.TestCase):
    def synthetic(self):
        models = []
        for t, tier in enumerate(("0.6B", "4B", "27B")):
            for k in range(6):
                quality = 0.2 * t + 0.03 * k
                models.append(
                    model(
                        f"{tier}-{k}",
                        tier,
                        30 + 60 * quality + (1.5 if k % 2 else -1.5),
                        "candidate" if k >= 3 else "comparator",
                        "trk" if k >= 3 else "eval",
                        T_dev=0.3 + quality,
                        H_pilot=0.2 + quality + (0.05 if k % 3 == 0 else 0),
                        H_mean3=0.2 + quality + (0.02 if k % 2 else -0.01),
                        dev_choice=0.3 + quality,
                        dev_noul=0.5,
                        dev_score=0.2 + quality,
                    )
                )
        return {"models": models, "sensitivity": {"S1_exclude": ["0.6B-3"]}}

    def test_identical_proxies_never_beat_baseline(self):
        data = self.synthetic()
        models = data["models"]
        pairs = pc.model_pairs(models)
        x = [m["features"]["P"] for m in models]
        boot = pc.cluster_bootstrap(models, pairs, {"P": x, "copy": list(x)}, 50, 1)
        self.assertEqual(boot["decision_resolvable"]["copy"]["prob_better_than_P"], 0.0)
        self.assertEqual(
            boot["decision_resolvable"]["copy"]["diff_vs_P_ci95"], [0.0, 0.0]
        )

    def test_analyze_runs_and_selection_is_a_candidate(self):
        result = pc.analyze(self.synthetic(), cluster_draws=40)
        self.assertIn(result["recommended"]["proxy"], pc.CANDIDATES)
        self.assertEqual(set(result["sets"]), {"main", "S1"})
        main = result["sets"]["main"]
        self.assertEqual(main["n_models"], 18)
        self.assertEqual(main["proxies"]["L2"]["fitted_parameters"], 3)
        self.assertIn("decision_resolvable", main["cluster_bootstrap"])

    def test_selection_keeps_baseline_without_bootstrap_evidence(self):
        result = pc.analyze(self.synthetic(), cluster_draws=40)
        selection = result["selection"]
        for name, check in selection["checks"].items():
            if check["prob_better_primary"] < pc.SELECTION["min_prob_better"]:
                self.assertFalse(check["qualifies"], name)
        if not selection["qualified"]:
            self.assertEqual(selection["recommended"], pc.BASELINE)


if __name__ == "__main__":
    unittest.main()
